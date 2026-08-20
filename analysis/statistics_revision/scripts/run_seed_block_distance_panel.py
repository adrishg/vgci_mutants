#!/usr/bin/env python3
"""Generate the complete all-distance seed-block discovery panel.

The panel uses every final-QC survivor without conditioning on common
seed/model trajectories. Recycles are first reduced to one median per
trajectory, available AF2 model parameterizations receive equal weight within
each seed, and seeds receive equal weight. The panel deliberately reports
effect sizes rather than thousands of mass-univariate q-values; confirmatory
9,999-permutation/2,000-bootstrap inference is reserved for prespecified focal
outcomes in ``run_seed_block_revision.py``.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

_MPL_CACHE = Path(tempfile.gettempdir()) / "vgci_mutants_matplotlib"
_MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(_MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_seed_block_revision import (
    BASE_SEED,
    FINAL_DISTANCE_PATHS,
    add_metadata,
    read_csv,
)
from shared.distribution_statistics import candidate_distance_columns
from shared.seed_block_statistics import seed_distribution_metrics_matrix


COMPARISON_ROWS = [
    ("cav_wt_mask", "cav12", "WT", "vanilla", "WT", "masked", "protocol_effect", "primary", "within-sequence mask effect"),
    ("cav_g402s_mask", "cav12", "G402S", "vanilla", "G402S", "masked", "protocol_effect", "primary", "within-sequence mask effect"),
    ("cav_g406r_mask", "cav12", "G406R", "vanilla", "G406R", "masked", "protocol_effect", "primary", "within-sequence mask effect"),
    ("cav_wt_g402s_van", "cav12", "WT", "vanilla", "G402S", "vanilla", "sequence_effect", "primary", "sequence comparison under unmasked MSA"),
    ("cav_wt_g402s_mask", "cav12", "WT", "masked", "G402S", "masked", "sequence_effect", "protocol_specific_exploration", "condition-specific masks prevent a factorial sequence contrast"),
    ("cav_wtvan_g402smask", "cav12", "WT", "vanilla", "G402S", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("cav_wtmask_g402svan", "cav12", "WT", "masked", "G402S", "vanilla", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("cav_wt_g406r_van", "cav12", "WT", "vanilla", "G406R", "vanilla", "sequence_effect", "primary", "sequence comparison under unmasked MSA"),
    ("cav_wt_g406r_mask", "cav12", "WT", "masked", "G406R", "masked", "sequence_effect", "protocol_specific_exploration", "condition-specific masks prevent a factorial sequence contrast"),
    ("cav_wtvan_g406rmask", "cav12", "WT", "vanilla", "G406R", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("cav_wtmask_g406rvan", "cav12", "WT", "masked", "G406R", "vanilla", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("kv_wt_mask", "kv21", "WT", "vanilla", "WT", "masked", "protocol_effect", "primary", "shared Kv2.1 mask design"),
    ("kv_l403a_mask", "kv21", "L403A", "vanilla", "L403A", "masked", "protocol_effect", "primary", "shared Kv2.1 mask design"),
    ("kv_f412l_mask", "kv21", "F412L", "vanilla", "F412L", "masked", "protocol_effect", "primary", "shared Kv2.1 mask design"),
    ("kv_wt_l403a_van", "kv21", "WT", "vanilla", "L403A", "vanilla", "sequence_effect", "primary", "factorial Kv2.1 sequence contrast"),
    ("kv_wt_l403a_mask", "kv21", "WT", "masked", "L403A", "masked", "sequence_effect", "primary", "factorial Kv2.1 sequence contrast"),
    ("kv_wtvan_l403amask", "kv21", "WT", "vanilla", "L403A", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("kv_wtmask_l403avan", "kv21", "WT", "masked", "L403A", "vanilla", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("kv_wt_f412l_van", "kv21", "WT", "vanilla", "F412L", "vanilla", "sequence_effect", "primary", "factorial Kv2.1 sequence contrast"),
    ("kv_wt_f412l_mask", "kv21", "WT", "masked", "F412L", "masked", "sequence_effect", "primary", "factorial Kv2.1 sequence contrast"),
    ("kv_wtvan_f412lmask", "kv21", "WT", "vanilla", "F412L", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("kv_wtmask_f412lvan", "kv21", "WT", "masked", "F412L", "vanilla", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("nav_wt_van_maskv2", "nav15", "WT", "vanilla", "WT", "masked_v2", "protocol_effect", "primary", "WT v2 mask effect"),
    ("nav_wt_van_noifm", "nav15", "WT", "vanilla", "WT", "masked_v2_noIFM", "protocol_effect", "primary", "WT v2-noIFM mask effect"),
    ("nav_wt_maskv2_noifm", "nav15", "WT", "masked_v2", "WT", "masked_v2_noIFM", "mask_design_sensitivity", "sensitivity", "IFM inclusion sensitivity within WT"),
    ("nav_qqq_van_mask", "nav15", "QQQ", "vanilla", "QQQ", "masked", "protocol_effect", "primary", "QQQ standard-plus-IFM mask effect"),
    ("nav_qqq_van_maskv2", "nav15", "QQQ", "vanilla", "QQQ", "masked_v2", "protocol_effect", "primary", "QQQ v2 mask effect"),
    ("nav_qqq_mask_maskv2", "nav15", "QQQ", "masked", "QQQ", "masked_v2", "mask_design_sensitivity", "sensitivity", "mask design sensitivity within QQQ"),
    ("nav_wt_qqq_van", "nav15", "WT", "vanilla", "QQQ", "vanilla", "sequence_effect", "primary", "sequence comparison under unmasked MSA"),
    ("nav_wt_qqq_maskv2", "nav15", "WT", "masked_v2", "QQQ", "masked_v2", "sequence_effect", "protocol_specific_exploration", "same label does not establish identical masked columns"),
    ("nav_wtvan_qqqmask", "nav15", "WT", "vanilla", "QQQ", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent masks and sequence"),
    ("nav_wtvan_qqqmaskv2", "nav15", "WT", "vanilla", "QQQ", "masked_v2", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("nav_wtmaskv2_qqqvan", "nav15", "WT", "masked_v2", "QQQ", "vanilla", "cross_protocol_sequence", "sensitivity", "non-equivalent sequence and protocol"),
    ("nav_wtmaskv2_qqqmask", "nav15", "WT", "masked_v2", "QQQ", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent masks and sequence"),
    ("nav_wtnoifm_qqqvan", "nav15", "WT", "masked_v2_noIFM", "QQQ", "vanilla", "cross_protocol_sequence", "sensitivity", "non-equivalent masks and sequence"),
    ("nav_wtnoifm_qqqmask", "nav15", "WT", "masked_v2_noIFM", "QQQ", "masked", "cross_protocol_sequence", "sensitivity", "non-equivalent masks and sequence"),
    ("nav_wtnoifm_qqqmaskv2", "nav15", "WT", "masked_v2_noIFM", "QQQ", "masked_v2", "cross_protocol_sequence", "sensitivity", "non-equivalent masks and sequence"),
]

COMPARISON_COLUMNS = [
    "comparison_id", "channel", "condition_A", "protocol_A", "condition_B",
    "protocol_B", "comparison_class", "inferential_role", "design_note",
]


def comparison_registry() -> pd.DataFrame:
    """Return the auditable comparison design with non-factorial labels."""
    return pd.DataFrame(COMPARISON_ROWS, columns=COMPARISON_COLUMNS)


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unavailable"


def _trajectory_median_tables(
    registry: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    required = set()
    for row in registry.itertuples(index=False):
        required.add(f"{row.channel}|{row.condition_A}|{row.protocol_A}")
        required.add(f"{row.channel}|{row.condition_B}|{row.protocol_B}")
    tables: dict[str, pd.DataFrame] = {}
    audit_rows = []
    for ensemble_id in sorted(required):
        relative = FINAL_DISTANCE_PATHS[ensemble_id]
        frame = add_metadata(read_csv(ROOT / relative))
        columns = candidate_distance_columns(frame)
        reduced = frame.groupby(["seed", "model_number"], as_index=False)[columns].median()
        if reduced[columns].isna().any().any():
            raise ValueError(f"Non-finite trajectory medians in {ensemble_id}")
        tables[ensemble_id] = reduced
        audit_rows.append({
            "ensemble_id": ensemble_id,
            "source_path": relative,
            "source_filter": "registered final distance cohort after all_ok_3/available interface QC",
            "snapshot_rows": len(frame),
            "retained_seeds": reduced["seed"].nunique(),
            "retained_seed_model_trajectories": len(reduced),
            "distance_columns": len(columns),
            "within_trajectory_reduction": "median of retained recycle snapshots",
        })
    return tables, pd.DataFrame(audit_rows)


def _pairing_coverage(a: pd.DataFrame, b: pd.DataFrame) -> dict[str, float | int]:
    trajectories_a = set(zip(a["seed"], a["model_number"]))
    trajectories_b = set(zip(b["seed"], b["model_number"]))
    common = trajectories_a & trajectories_b
    return {
        "n_trajectories_A": len(trajectories_a),
        "n_trajectories_B": len(trajectories_b),
        "n_common_survivor_trajectories_descriptive": len(common),
        "paired_coverage_A_descriptive": len(common) / len(trajectories_a),
        "paired_coverage_B_descriptive": len(common) / len(trajectories_b),
    }


def _run_comparison(
    spec, tables: dict[str, pd.DataFrame], commit: str
) -> pd.DataFrame:
    key_a = f"{spec.channel}|{spec.condition_A}|{spec.protocol_A}"
    key_b = f"{spec.channel}|{spec.condition_B}|{spec.protocol_B}"
    a, b = tables[key_a], tables[key_b]
    columns = sorted(
        (set(candidate_distance_columns(a)) & set(candidate_distance_columns(b)))
    )
    result = seed_distribution_metrics_matrix(a, b, columns)
    models = sorted(set(a["model_number"]) | set(b["model_number"]))
    loo_w1, loo_delta, loo_log_iqr = [], [], []
    for model in models:
        sensitivity = seed_distribution_metrics_matrix(
            a[a["model_number"].ne(model)],
            b[b["model_number"].ne(model)],
            columns,
        )
        result[f"W1_A_omit_AF2_model_{model}"] = sensitivity["seed_balanced_W1_A"]
        result[f"delta_median_A_omit_AF2_model_{model}"] = sensitivity["delta_weighted_median_A"]
        result[f"log2_IQR_ratio_omit_AF2_model_{model}"] = sensitivity["weighted_log2_IQR_ratio"]
        loo_w1.append(sensitivity["seed_balanced_W1_A"].to_numpy())
        loo_delta.append(sensitivity["delta_weighted_median_A"].to_numpy())
        loo_log_iqr.append(sensitivity["weighted_log2_IQR_ratio"].to_numpy())
    w1_matrix = np.vstack(loo_w1)
    delta_matrix = np.vstack(loo_delta)
    log_iqr_matrix = np.vstack(loo_log_iqr)
    result["W1_A_leave_one_model_out_min"] = w1_matrix.min(axis=0)
    result["W1_A_leave_one_model_out_max"] = w1_matrix.max(axis=0)
    result["delta_median_A_leave_one_model_out_min"] = delta_matrix.min(axis=0)
    result["delta_median_A_leave_one_model_out_max"] = delta_matrix.max(axis=0)
    result["log2_IQR_ratio_leave_one_model_out_min"] = log_iqr_matrix.min(axis=0)
    result["log2_IQR_ratio_leave_one_model_out_max"] = log_iqr_matrix.max(axis=0)
    base_delta_sign = np.sign(result["delta_weighted_median_A"].to_numpy())
    base_iqr_sign = np.sign(result["weighted_log2_IQR_ratio"].to_numpy())
    result["delta_median_direction_stable_leave_one_model_out"] = np.all(
        np.sign(delta_matrix) == base_delta_sign[None, :], axis=0
    )
    result["IQR_direction_stable_leave_one_model_out"] = np.all(
        np.sign(log_iqr_matrix) == base_iqr_sign[None, :], axis=0
    )

    coverage = _pairing_coverage(a, b)
    metadata = {
        "comparison_id": spec.comparison_id,
        "channel": spec.channel,
        "comparison_class": spec.comparison_class,
        "inferential_role": spec.inferential_role,
        "design_note": spec.design_note,
        "condition_A": spec.condition_A,
        "protocol_A": spec.protocol_A,
        "condition_B": spec.condition_B,
        "protocol_B": spec.protocol_B,
        "source_path_A": FINAL_DISTANCE_PATHS[key_a],
        "source_path_B": FINAL_DISTANCE_PATHS[key_b],
        "source_filter": "all registered final-QC survivors; no common-survivor conditioning",
        "within_trajectory_reduction": "median",
        "within_seed_weighting": "equal available AF2 model parameterizations",
        "between_seed_weighting": "equal",
        "panel_status": "descriptive discovery; no mass-univariate p/q values",
        "confirmatory_inference": "prespecified focal outputs use 9999 seed permutations and 2000 seed bootstraps",
        "resampling_unit_for_confirmatory_outputs": "seed",
        "random_seed_for_point_estimates": "not applicable (deterministic)",
        "script": "analysis/statistics_revision/scripts/run_seed_block_distance_panel.py",
        "git_commit": commit,
        "n_seeds_A": a["seed"].nunique(),
        "n_seeds_B": b["seed"].nunique(),
        **coverage,
    }
    for key, value in reversed(list(metadata.items())):
        result.insert(0, key, value)
    return result


def _comparison_summary(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for comparison_id, group in effects.groupby("comparison_id", sort=False):
        top = group.loc[group["seed_balanced_W1_A"].idxmax()]
        rows.append({
            "comparison_id": comparison_id,
            "channel": group["channel"].iloc[0],
            "comparison_class": group["comparison_class"].iloc[0],
            "inferential_role": group["inferential_role"].iloc[0],
            "design_note": group["design_note"].iloc[0],
            "number_distances": len(group),
            "median_seed_balanced_W1_A": group["seed_balanced_W1_A"].median(),
            "median_normalized_W1": group[
                "seed_balanced_W1_normalized_by_weighted_pooled_IQR"
            ].median(),
            "median_weighted_IQR_ratio_B_over_A": group[
                "weighted_IQR_ratio_B_over_A"
            ].median(),
            "fraction_broader_B_descriptive": group["weighted_log2_IQR_ratio"].gt(0).mean(),
            "fraction_narrower_B_descriptive": group["weighted_log2_IQR_ratio"].lt(0).mean(),
            "fraction_delta_direction_stable_leave_one_model_out": group[
                "delta_median_direction_stable_leave_one_model_out"
            ].mean(),
            "fraction_IQR_direction_stable_leave_one_model_out": group[
                "IQR_direction_stable_leave_one_model_out"
            ].mean(),
            "maximum_seed_balanced_W1_A": top["seed_balanced_W1_A"],
            "distance_with_maximum_W1": top["distance"],
            "mass_univariate_p_or_q_values": "not calculated; discovery panel is effect-size only",
        })
    return pd.DataFrame(rows)


def _kv_interactions(effects: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for mutant, vanilla_id, masked_id in (
        ("L403A", "kv_wt_l403a_van", "kv_wt_l403a_mask"),
        ("F412L", "kv_wt_f412l_van", "kv_wt_f412l_mask"),
    ):
        vanilla = effects[effects["comparison_id"].eq(vanilla_id)].copy()
        masked = effects[effects["comparison_id"].eq(masked_id)].copy()
        keep = ["distance", "delta_weighted_median_A"] + [
            f"delta_median_A_omit_AF2_model_{model}" for model in range(1, 6)
        ]
        merged = vanilla[keep].merge(masked[keep], on="distance", suffixes=("_vanilla", "_masked"))
        merged.insert(0, "mutant", mutant)
        merged["interaction_delta_median_A"] = (
            merged["delta_weighted_median_A_masked"]
            - merged["delta_weighted_median_A_vanilla"]
        )
        loo = []
        for model in range(1, 6):
            column = f"interaction_delta_median_A_omit_AF2_model_{model}"
            merged[column] = (
                merged[f"delta_median_A_omit_AF2_model_{model}_masked"]
                - merged[f"delta_median_A_omit_AF2_model_{model}_vanilla"]
            )
            loo.append(merged[column].to_numpy())
        loo_matrix = np.vstack(loo)
        merged["interaction_leave_one_model_out_min_A"] = loo_matrix.min(axis=0)
        merged["interaction_leave_one_model_out_max_A"] = loo_matrix.max(axis=0)
        merged["interaction_direction_stable_leave_one_model_out"] = np.all(
            np.sign(loo_matrix) == np.sign(merged["interaction_delta_median_A"].to_numpy())[None, :],
            axis=0,
        )
        merged["estimand"] = "(mutant_masked-WT_masked)-(mutant_vanilla-WT_vanilla)"
        merged["status"] = "descriptive full-panel interaction; focal coordinates require seed-bootstrap CI"
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True)


def _plot_top_distance_heatmaps(effects: pd.DataFrame, output: Path) -> None:
    display_roles = {"primary", "protocol_specific_exploration"}
    data = effects[effects["inferential_role"].isin(display_roles)].copy()
    figure, axes = plt.subplots(1, 3, figsize=(28, 12), constrained_layout=True)
    channel_colors = {"cav12": "magma", "kv21": "viridis", "nav15": "cividis"}
    for axis, channel in zip(axes, ("cav12", "kv21", "nav15")):
        part = data[data["channel"].eq(channel)]
        top = part.groupby("distance")["seed_balanced_W1_A"].max().nlargest(20).index
        matrix = part[part["distance"].isin(top)].pivot_table(
            index="distance", columns="comparison_id", values="seed_balanced_W1_A"
        ).reindex(top)
        image = axis.imshow(matrix.to_numpy(), aspect="auto", cmap=channel_colors[channel])
        axis.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=70, ha="right")
        axis.set_yticks(np.arange(len(matrix.index)), matrix.index, fontsize=7)
        axis.set_title(f"{channel}: largest seed-balanced W1 coordinates\n(discovery panel)")
        axis.set_xlabel("Comparison")
        figure.colorbar(image, ax=axis, fraction=.035, pad=.02, label="W1 (Å)")
    figure.savefig(output / "all_distance_seed_block_top_effects.png", dpi=220, bbox_inches="tight")
    figure.savefig(output / "all_distance_seed_block_top_effects.pdf", bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "analysis/statistics_revision/seed_block/full_panel",
    )
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    commit = _git_commit()
    registry = comparison_registry()
    tables, source_audit = _trajectory_median_tables(registry)
    source_audit.to_csv(output / "all_distance_seed_block_source_audit.csv", index=False)
    registry.to_csv(output / "all_distance_seed_block_comparison_registry.csv", index=False)

    results = []
    for spec in registry.itertuples(index=False):
        print(f"run: {spec.comparison_id}", flush=True)
        results.append(_run_comparison(spec, tables, commit))
    effects = pd.concat(results, ignore_index=True)
    effects.to_csv(
        output / "all_distance_seed_block_effects.csv.gz",
        index=False,
        compression="gzip",
    )
    summary = _comparison_summary(effects)
    summary.to_csv(output / "all_distance_seed_block_comparison_summary.csv", index=False)
    interactions = _kv_interactions(effects)
    interactions.to_csv(
        output / "kv21_all_distance_masking_by_sequence_interactions.csv.gz",
        index=False,
        compression="gzip",
    )
    _plot_top_distance_heatmaps(effects, output)

    report = {
        "status": "completed_seed_block_all_distance_discovery_panel",
        "git_commit": commit,
        "deterministic_point_estimate_seed": None,
        "registered_ensembles": len(tables),
        "comparisons": len(registry),
        "distance_effect_rows": len(effects),
        "primary_estimand": "median retained recycle per seed-model trajectory; equal available AF2 models within seed; equal seeds",
        "survivor_policy": "all final-QC survivors; no common-survivor conditioning",
        "leave_one_AF2_model_out": "complete for every distance and comparison",
        "mass_univariate_p_or_q_values": "not calculated; panel is descriptive discovery and fraction-significant is intentionally removed",
        "confirmatory_focal_mode": {
            "permutations": 9999,
            "bootstrap_replicates": 2000,
            "random_seed": BASE_SEED,
            "location": "analysis/statistics_revision/seed_block",
        },
        "nonfactorial_design_policy": "Cav1.2/Nav1.5 masked WT-mutant contrasts are labeled protocol-specific explorations; vanilla sequence contrasts are primary",
        "elapsed_seconds": time.time() - started,
    }
    (output / "all_distance_seed_block_panel_run_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(f"completed: {output}", flush=True)


if __name__ == "__main__":
    main()
