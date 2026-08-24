#!/usr/bin/env python3
"""Whole-seed Kv2.1 breadth and RMSF positional-dispersion inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_kv21_sampling_breadth_uncertainty import (
    S6_PATHS,
    _selected_rmsf_rows,
)
from analysis.statistics_revision.scripts.run_seed_block_revision import add_metadata, read_csv
from scripts.ensemble_rmsf_analysis.io import read_csv_resolving_lfs, resolve_local_lfs_object
from shared.plotting import add_s6_cross_pore_columns
from shared.seed_block_statistics import seed_distribution_metrics_matrix, weighted_quantile


BASE_SEED = 20260824


def expanded_draw(frame: pd.DataFrame, draw: np.ndarray) -> pd.DataFrame:
    groups = {seed: part for seed, part in frame.groupby("seed")}
    pieces = []
    for draw_id, seed in enumerate(draw):
        if seed in groups:
            part = groups[seed].copy(); part["seed"] = draw_id; pieces.append(part)
    if not pieces:
        return frame.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def weighted_mad(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    rows, weights = [], []
    seeds = frame.seed.nunique()
    for _, seed in frame.groupby("seed"):
        models = seed.model_number.nunique()
        rows.append(seed[columns].to_numpy(float))
        weights.append(np.full(len(seed), 1 / (seeds * models)))
    values, mass = np.vstack(rows), np.concatenate(weights)
    medians = np.array([weighted_quantile(values[:, i], .5, mass) for i in range(len(columns))])
    return np.array([
        weighted_quantile(np.abs(values[:, i] - medians[i]), .5, mass)
        for i in range(len(columns))
    ])


def breadth(output: Path, bootstrap: int) -> None:
    rows = []
    for condition_index, (condition, paths) in enumerate(S6_PATHS.items()):
        tables = {}
        aliases = None
        for protocol, filename in paths.items():
            frame, current = add_s6_cross_pore_columns(read_csv(ROOT / "kv21/dataDistances" / filename))
            frame = add_metadata(frame)
            aliases = current
            columns = list(current.values())
            tables[protocol] = frame.groupby(["seed", "model_number"], as_index=False)[columns].median()
        columns = list(aliases.values())
        point = seed_distribution_metrics_matrix(tables["vanilla"], tables["masked"], columns)
        point_mad_a = weighted_mad(tables["vanilla"], columns)
        point_mad_b = weighted_mad(tables["masked"], columns)
        labels = np.array(sorted(set(tables["vanilla"].seed) | set(tables["masked"].seed)))
        rng = np.random.default_rng(BASE_SEED + 7000 + condition_index)
        samples = np.empty((bootstrap, 4), dtype=float)
        for iteration in range(bootstrap):
            draw = rng.choice(labels, len(labels), replace=True)
            a, b = expanded_draw(tables["vanilla"], draw), expanded_draw(tables["masked"], draw)
            metrics = seed_distribution_metrics_matrix(a, b, columns)
            mad_a, mad_b = weighted_mad(a, columns), weighted_mad(b, columns)
            samples[iteration] = [
                np.median(metrics.weighted_IQR_ratio_B_over_A),
                np.median((mad_b + 1e-12) / (mad_a + 1e-12)),
                np.median(metrics.seed_balanced_W1_A),
                np.median(metrics.delta_weighted_median_A),
            ]
        intervals = np.quantile(samples, [.025, .975], axis=0)
        rows.append({
            "sequence_background": condition,
            "outcome_family": "six chain-label-invariant S6 cross-pore coordinates",
            "median_masked_over_vanilla_IQR_ratio": point.weighted_IQR_ratio_B_over_A.median(),
            "IQR_ratio_CI_low": intervals[0, 0], "IQR_ratio_CI_high": intervals[1, 0],
            "median_masked_over_vanilla_MAD_ratio": np.median((point_mad_b + 1e-12) / (point_mad_a + 1e-12)),
            "MAD_ratio_CI_low": intervals[0, 1], "MAD_ratio_CI_high": intervals[1, 1],
            "median_raw_W1_A": point.seed_balanced_W1_A.median(),
            "W1_CI_low_A": intervals[0, 2], "W1_CI_high_A": intervals[1, 2],
            "median_signed_change_A": point.delta_weighted_median_A.median(),
            "signed_change_CI_low_A": intervals[0, 3], "signed_change_CI_high_A": intervals[1, 3],
            "SD_status": "secondary; robust IQR and MAD are primary breadth measures",
            "bootstrap_unit": "joint nominal seed label; trajectory medians and equal model weights",
            "bootstrap_replicates": bootstrap,
            "random_seed": BASE_SEED + 7000 + condition_index,
        })
    pd.DataFrame(rows).to_csv(output / "kv21_breadth_whole_seed.csv", index=False)


def seed_moments(coords, present, group: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    labels, means, seconds, availability = [], [], [], []
    for seed, seed_rows in group.groupby("seed", sort=True):
        trajectory_means, trajectory_seconds, trajectory_present = [], [], []
        for _, trajectory in seed_rows.groupby("model_number", sort=True):
            idx = trajectory.manifest_index.astype(int).to_numpy()
            xyz = np.asarray(coords[idx], dtype=np.float64)
            valid = np.asarray(present[idx], dtype=bool) & np.isfinite(xyz).all(axis=-1)
            count = valid.sum(axis=0)
            total = np.where(valid[..., None], xyz, 0).sum(axis=0)
            total_sq = np.where(valid, np.square(xyz).sum(axis=-1), 0).sum(axis=0)
            trajectory_means.append(np.divide(total, count[..., None], out=np.zeros_like(total), where=count[..., None] > 0))
            trajectory_seconds.append(np.divide(total_sq, count, out=np.zeros_like(total_sq), where=count > 0))
            trajectory_present.append(count > 0)
        valid_models = np.stack(trajectory_present)
        denominator = valid_models.sum(axis=0)
        mean = np.divide(
            np.where(valid_models[..., None], np.stack(trajectory_means), 0).sum(axis=0),
            denominator[..., None], out=np.zeros_like(trajectory_means[0]), where=denominator[..., None] > 0,
        )
        second = np.divide(
            np.where(valid_models, np.stack(trajectory_seconds), 0).sum(axis=0),
            denominator, out=np.zeros_like(trajectory_seconds[0]), where=denominator > 0,
        )
        labels.append(int(seed)); means.append(mean); seconds.append(second); availability.append(denominator > 0)
    return np.array(labels), np.stack(means), np.stack(seconds), np.stack(availability)


def rmsf_profile(moment, draw: np.ndarray) -> np.ndarray:
    labels, means, seconds, available = moment
    index = {label: i for i, label in enumerate(labels)}
    selected = [index[label] for label in draw if label in index]
    m, s, present = means[selected], seconds[selected], available[selected]
    denominator = present.sum(axis=0)
    mean = np.divide(
        np.where(present[..., None], m, 0).sum(axis=0), denominator[..., None],
        out=np.zeros_like(m[0]), where=denominator[..., None] > 0,
    )
    second = np.divide(
        np.where(present, s, 0).sum(axis=0), denominator,
        out=np.zeros_like(s[0]), where=denominator > 0,
    )
    rmsf = np.sqrt(np.maximum(second - np.square(mean).sum(axis=-1), 0))
    rmsf[denominator == 0] = np.nan
    return np.nanmean(rmsf, axis=0)


def rmsf(output: Path, bootstrap: int) -> None:
    merged = ROOT / "kv21/dataRMSF/merged"
    metadata = read_csv_resolving_lfs(merged / "kv21_alignment_metadata.csv", ROOT)
    coords = np.load(resolve_local_lfs_object(merged / "kv21_aligned_ca_coordinates.npy", ROOT), mmap_mode="r")
    present = np.load(resolve_local_lfs_object(merged / "kv21_aligned_ca_present.npy", ROOT), mmap_mode="r")
    selected = _selected_rmsf_rows(metadata)
    profile = read_csv_resolving_lfs(
        ROOT / "kv21/dataRMSF/profiles/kv21_all_ok_3_symmetry_averaged_profiles.csv", ROOT
    )
    direct_mask = profile.loc[profile.dataset.eq("wt_masked")].sort_values("raw_residue_number").directly_masked.to_numpy(bool)
    rows = []
    for condition_index, condition in enumerate(("wt", "l403a", "f412l")):
        moments = {
            protocol: seed_moments(
                coords, present,
                selected[(selected.sequence_condition == condition) & (selected.protocol == protocol)],
            )
            for protocol in ("vanilla", "masked")
        }
        labels = np.array(sorted(set(moments["vanilla"][0]) | set(moments["masked"][0])))
        observed = rmsf_profile(moments["masked"], labels) - rmsf_profile(moments["vanilla"], labels)
        rng = np.random.default_rng(BASE_SEED + 8000 + condition_index)
        samples = np.empty((bootstrap, 2))
        for iteration in range(bootstrap):
            draw = rng.choice(labels, len(labels), replace=True)
            delta = rmsf_profile(moments["masked"], draw) - rmsf_profile(moments["vanilla"], draw)
            samples[iteration] = [np.nanmedian(delta[direct_mask]), np.nanmedian(delta[~direct_mask])]
        intervals = np.quantile(samples, [.025, .975], axis=0)
        for index, region in enumerate(("directly_masked_positions", "outside_direct_mask")):
            rows.append({
                "sequence_background": condition.upper(), "region": region,
                "observed_median_masked_minus_vanilla_RMSF_A": np.nanmedian(observed[direct_mask if index == 0 else ~direct_mask]),
                "whole_seed_CI_low_A": intervals[0, index], "whole_seed_CI_high_A": intervals[1, index],
                "bootstrap_unit": "joint nominal seed label; equal surviving models within seed and equal retained snapshots within model",
                "interpretation": "positional dispersion among AlphaFold2 outputs, not molecular flexibility",
                "bootstrap_replicates": bootstrap, "random_seed": BASE_SEED + 8000 + condition_index,
            })
    pd.DataFrame(rows).to_csv(output / "kv21_rmsf_whole_seed.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    parser.add_argument("--mode", choices=("exploratory", "publication"), default="publication")
    args = parser.parse_args()
    bootstrap = 250 if args.mode == "exploratory" else 2000
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    breadth(output, bootstrap)
    rmsf(output, bootstrap)
    (output / "breadth_rmsf_run_summary.json").write_text(json.dumps({
        "mode": args.mode, "bootstrap_replicates": bootstrap, "base_seed": BASE_SEED,
        "breadth_primary": "IQR and MAD", "RMSF_interpretation": "positional dispersion among predictions",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
