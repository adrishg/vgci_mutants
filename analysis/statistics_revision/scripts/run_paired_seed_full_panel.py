#!/usr/bin/env python3
"""Paired-seed full distance-panel effect estimates and rank stability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_seed_block_distance_panel import (
    _comparison_summary,
    _plot_top_distance_heatmaps,
    _trajectory_median_tables,
    comparison_registry,
)
from shared.distribution_statistics import candidate_distance_columns
from shared.paired_seed_statistics import low_pooled_iqr_flags, select_paired_estimand
from shared.seed_block_statistics import seed_distribution_metrics_matrix


BASE_SEED = 20260824


def parse_ranges(text: str) -> set[int]:
    result = set()
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start, stop = map(int, token.split("-"))
            result.update(range(start, stop + 1))
        else:
            result.add(int(token))
    return result


def masks() -> dict[str, set[int]]:
    text = (ROOT / "scripts/ensemble_rmsf_analysis/config/authoritative_mask_definitions.yaml").read_text()
    block = text.split("mask_definitions:\n", 1)[1].split("\ndatasets:\n", 1)[0]
    definitions: dict[str, dict[str, str]] = {}
    current = None
    for line in block.splitlines():
        match = re.match(r"  ([A-Za-z0-9_]+):\s*$", line)
        if match:
            current = match.group(1); definitions[current] = {}; continue
        prop = re.match(r"    ([A-Za-z0-9_]+):\s*\"?([^\"]*)\"?\s*$", line)
        if prop and current:
            definitions[current][prop.group(1)] = prop.group(2).strip()
    resolved = {}
    while len(resolved) < len(definitions):
        for name, definition in definitions.items():
            if name in resolved:
                continue
            if "base" in definition and definition["base"] not in resolved:
                continue
            values = set(resolved.get(definition.get("base"), set()))
            values |= parse_ranges(definition.get("ranges", ""))
            values |= parse_ranges(definition.get("add_ranges", ""))
            values -= parse_ranges(definition.get("remove_ranges", ""))
            resolved[name] = values
    return resolved


def mask_id_for_spec(spec) -> str:
    if spec.protocol_A == spec.protocol_B == "vanilla":
        return "unmasked"
    if spec.channel == "kv21":
        return "kv21_common"
    mapping = {
        ("nav15", "WT", "masked"): "nav15_standard",
        ("nav15", "WT", "masked_v2"): "nav15_v2",
        ("nav15", "WT", "masked_v2_noIFM"): "nav15_v2_noIFM",
        ("nav15", "QQQ", "masked"): "nav15_standard_plus_IFM",
        ("nav15", "QQQ", "masked_v2"): "nav15_v2",
        ("cav12", "WT", "masked"): "cav12_wt_common",
        ("cav12", "G402S", "masked"): "cav12_g402s",
        ("cav12", "G406R", "masked"): "cav12_g406r",
    }
    ids = {
        mapping.get((spec.channel, spec.condition_A, spec.protocol_A)),
        mapping.get((spec.channel, spec.condition_B, spec.protocol_B)),
    } - {None}
    return next(iter(ids)) if len(ids) == 1 else "nonidentical_or_unresolved_masks"


def endpoint_classification(distance: str, mask_positions: set[int] | None) -> str:
    if mask_positions is None:
        return "not_classifiable_without_equivalent_mask"
    positions = [int(value) for value in re.findall(r"(?<=[A-Z])(\d+)", distance)]
    if len(positions) < 2:
        return "endpoint_parse_unresolved"
    count = sum(position in mask_positions for position in positions[:2])
    return ("neither_endpoint_masked", "one_endpoint_masked", "both_endpoints_masked")[count]


def bootstrap_rank_stability(
    a: pd.DataFrame,
    b: pd.DataFrame,
    columns: list[str],
    *,
    replicates: int,
    random_seed: int,
) -> pd.DataFrame:
    labels = sorted(set(a.seed) | set(b.seed))
    by_a = {seed: part for seed, part in a.groupby("seed")}
    by_b = {seed: part for seed, part in b.groupby("seed")}
    counts = np.zeros((len(columns), 3), dtype=int)
    rng = np.random.default_rng(random_seed)
    valid = 0
    for _ in range(replicates):
        draw = rng.choice(labels, len(labels), replace=True)
        pieces_a, pieces_b = [], []
        for draw_id, seed in enumerate(draw):
            if seed in by_a:
                part = by_a[seed].copy(); part["seed"] = draw_id; pieces_a.append(part)
            if seed in by_b:
                part = by_b[seed].copy(); part["seed"] = draw_id; pieces_b.append(part)
        if not pieces_a or not pieces_b:
            continue
        metrics = seed_distribution_metrics_matrix(
            pd.concat(pieces_a, ignore_index=True), pd.concat(pieces_b, ignore_index=True), columns
        )
        order = np.argsort(-metrics.seed_balanced_W1_A.to_numpy())
        for column, fraction in enumerate((.01, .05, .10)):
            top = max(1, int(np.ceil(len(columns) * fraction)))
            counts[order[:top], column] += 1
        valid += 1
    return pd.DataFrame({
        "distance": columns,
        "top_1pct_frequency": counts[:, 0] / valid,
        "top_5pct_frequency": counts[:, 1] / valid,
        "top_10pct_frequency": counts[:, 2] / valid,
        "rank_bootstrap_valid_replicates": valid,
        "rank_bootstrap_requested_replicates": replicates,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2/full_panel")
    parser.add_argument("--mode", choices=("exploratory", "publication"), default="publication")
    args = parser.parse_args()
    rank_replicates = 25 if args.mode == "exploratory" else 200
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    registry = comparison_registry()
    tables, source_audit = _trajectory_median_tables(registry)
    mask_sets = masks()
    results = []
    for comparison_index, spec in enumerate(registry.itertuples(index=False)):
        print(f"paired full panel: {spec.comparison_id}", flush=True)
        key_a = f"{spec.channel}|{spec.condition_A}|{spec.protocol_A}"
        key_b = f"{spec.channel}|{spec.condition_B}|{spec.protocol_B}"
        a, b = tables[key_a], tables[key_b]
        columns = sorted(set(candidate_distance_columns(a)) & set(candidate_distance_columns(b)))
        primary = seed_distribution_metrics_matrix(a, b, columns)
        primary["estimand_id"] = "primary_joint_nominal_seed"
        loo_w1, loo_delta, loo_iqr = [], [], []
        for model in sorted(set(a.model_number) | set(b.model_number)):
            loo = seed_distribution_metrics_matrix(
                a[a.model_number.ne(model)], b[b.model_number.ne(model)], columns
            )
            primary[f"W1_A_omit_AF2_model_{model}"] = loo.seed_balanced_W1_A
            primary[f"delta_median_A_omit_AF2_model_{model}"] = loo.delta_weighted_median_A
            primary[f"log2_IQR_ratio_omit_AF2_model_{model}"] = loo.weighted_log2_IQR_ratio
            loo_w1.append(loo.seed_balanced_W1_A.to_numpy())
            loo_delta.append(loo.delta_weighted_median_A.to_numpy())
            loo_iqr.append(loo.weighted_log2_IQR_ratio.to_numpy())
        w1_matrix, delta_matrix, iqr_matrix = map(np.vstack, (loo_w1, loo_delta, loo_iqr))
        primary["W1_A_leave_one_model_out_min"] = w1_matrix.min(axis=0)
        primary["W1_A_leave_one_model_out_max"] = w1_matrix.max(axis=0)
        primary["delta_median_A_leave_one_model_out_min"] = delta_matrix.min(axis=0)
        primary["delta_median_A_leave_one_model_out_max"] = delta_matrix.max(axis=0)
        primary["log2_IQR_ratio_leave_one_model_out_min"] = iqr_matrix.min(axis=0)
        primary["log2_IQR_ratio_leave_one_model_out_max"] = iqr_matrix.max(axis=0)
        primary["delta_median_direction_stable_leave_one_model_out"] = np.all(
            np.sign(delta_matrix) == np.sign(primary.delta_weighted_median_A.to_numpy())[None, :], axis=0
        )
        primary["IQR_direction_stable_leave_one_model_out"] = np.all(
            np.sign(iqr_matrix) == np.sign(primary.weighted_log2_IQR_ratio.to_numpy())[None, :], axis=0
        )
        for sensitivity in ("common_seed", "common_model_seed"):
            sa, sb, audit = select_paired_estimand(a, b, columns[0], estimand=sensitivity)
            metrics = seed_distribution_metrics_matrix(sa, sb, columns).set_index("distance")
            for metric in ("seed_balanced_W1_A", "delta_weighted_median_A", "weighted_log2_IQR_ratio"):
                primary[f"{metric}_{sensitivity}"] = primary.distance.map(metrics[metric])
            for key, value in audit.items():
                primary[f"{key}_{sensitivity}"] = value
        flags = primary.weighted_pooled_IQR_A.map(low_pooled_iqr_flags).apply(pd.Series)
        primary = pd.concat([primary, flags], axis=1)
        rank = bootstrap_rank_stability(
            a, b, columns, replicates=rank_replicates,
            random_seed=BASE_SEED + 10000 + comparison_index,
        )
        primary = primary.merge(rank, on="distance", validate="one_to_one")
        mask_id = mask_id_for_spec(spec)
        mask_positions = mask_sets.get(mask_id)
        primary["mask_id_for_endpoint_classification"] = mask_id
        primary["mask_endpoint_class"] = primary.distance.map(
            lambda value: endpoint_classification(value, mask_positions)
        )
        for name in (
            "comparison_id", "channel", "condition_A", "protocol_A", "condition_B",
            "protocol_B", "comparison_class", "inferential_role", "design_note",
        ):
            primary[name] = getattr(spec, name)
        primary["panel_description"] = "coordinate-by-comparison effect estimates"
        primary["actual_random_seed_pairing_status"] = "not verified; nominal seed labels only"
        results.append(primary)
    effects = pd.concat(results, ignore_index=True)
    effects.to_csv(output / "paired_seed_full_panel_effects.csv.gz", index=False, compression="gzip")
    source_audit.to_csv(output / "source_audit.csv", index=False)
    registry.to_csv(output / "comparison_registry.csv", index=False)
    summary = _comparison_summary(effects)
    summary.to_csv(output / "comparison_summary.csv", index=False)
    region = effects.groupby(
        ["channel", "comparison_id", "mask_endpoint_class"], as_index=False
    ).agg(
        registered_effect_estimates=("distance", "size"),
        median_raw_W1_A=("seed_balanced_W1_A", "median"),
        median_absolute_signed_change_A=("delta_weighted_median_A", lambda x: np.abs(x).median()),
        median_log2_IQR_ratio=("weighted_log2_IQR_ratio", "median"),
        median_top_5pct_rank_frequency=("top_5pct_frequency", "median"),
    )
    region.to_csv(output / "region_balanced_endpoint_summary.csv", index=False)
    _plot_top_distance_heatmaps(effects, output)
    report = {
        "mode": args.mode, "rank_bootstrap_replicates": rank_replicates,
        "random_seed_base": BASE_SEED, "comparisons": len(registry),
        "coordinate_by_comparison_effect_estimates": len(effects),
        "normalization_low_spread_flags_A": [0.05, 0.10, 0.25],
        "mass_univariate_p_or_q_values": "not calculated",
        "rank_interpretation": "recurrence under whole-nominal-seed resampling; not independent validation",
    }
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
