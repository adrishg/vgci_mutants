#!/usr/bin/env python3
"""Repeated common-seed reduced-depth sensitivity with paired inner bootstrap."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_paired_seed_v2 import trajectory_summary
from analysis.statistics_revision.scripts.run_seed_block_revision import (
    L403_COLUMNS, add_metadata, derive_l403a_threshold, read_csv,
)


BASE_SEED = 20260824


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--inner-bootstrap", type=int, default=2000)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    threshold, _, _ = derive_l403a_threshold(output)
    frame = add_metadata(read_csv(ROOT / "kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv"))
    frame[L403_COLUMNS] = frame[L403_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame["max_distance_A"] = frame[L403_COLUMNS].max(axis=1)
    frame["any_shifted"] = frame[L403_COLUMNS].ge(threshold).any(axis=1).astype(float)
    definitions = {
        "continuous_max_distance_A": ("max_distance_A", "median"),
        "experiment_anchored_any_shifted_fraction": ("any_shifted", "mean"),
    }
    rng = np.random.default_rng(BASE_SEED + 9000)
    records, summaries = [], []
    for metric, (column, reduction) in definitions.items():
        series = {}
        for protocol in ("vanilla", "masked"):
            trajectories = trajectory_summary(frame[frame.condition.eq(protocol)], column, reduction)
            series[protocol] = trajectories.groupby("seed")[column].mean()
        common = np.array(sorted(set(series["vanilla"].index) & set(series["masked"].index)))
        paired_differences = series["masked"].loc[common] - series["vanilla"].loc[common]
        full_point = paired_differences.mean()
        metric_rows = []
        for draw_index in range(args.draws):
            chosen = rng.choice(common, 20, replace=False)
            differences = (series["masked"].loc[chosen] - series["vanilla"].loc[chosen]).to_numpy(float)
            effect = differences.mean()
            inner = differences[
                rng.integers(0, len(differences), size=(args.inner_bootstrap, len(differences)))
            ].mean(axis=1)
            low, high = np.quantile(inner, [.025, .975])
            row = {
                "metric": metric, "draw": draw_index + 1,
                "common_seed_labels_sampled": 20,
                "nominal_model_seed_trajectories_per_protocol": 100,
                "effect": effect, "inner_interval_low": low, "inner_interval_high": high,
                "full_ensemble_common_seed_point_estimate": full_point,
                "same_direction_as_full": np.sign(effect) == np.sign(full_point),
                "relative_error": abs(effect - full_point) / max(abs(full_point), 1e-12),
                "subset_interval_contains_full_ensemble_point_estimate": low <= full_point <= high,
                "paired_inner_bootstrap": True,
                "actual_random_seed_pairing_status": "not verified; common recorded numeric seed labels only",
            }
            records.append(row); metric_rows.append(row)
        table = pd.DataFrame(metric_rows)
        summaries.append({
            "metric": metric, "draws": args.draws,
            "fraction_same_direction": table.same_direction_as_full.mean(),
            "median_relative_error": table.relative_error.median(),
            "fraction_subset_interval_contains_full_ensemble_point_estimate": table.subset_interval_contains_full_ensemble_point_estimate.mean(),
            "interpretation": "retrospective stability; not frequentist coverage or a stopping rule",
            "inner_bootstrap_replicates": args.inner_bootstrap,
            "random_seed": BASE_SEED + 9000,
        })
    pd.DataFrame(records).to_csv(output / "reduced_depth_paired_draws.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "reduced_depth_paired_summary.csv", index=False)


if __name__ == "__main__":
    main()
