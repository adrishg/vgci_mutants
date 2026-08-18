"""Audit and uncertainty estimates for manuscript-facing Kv2.1 breadth claims."""

from __future__ import annotations

from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from scripts.ensemble_rmsf_analysis.io import read_csv_resolving_lfs, resolve_local_lfs_object
from shared.plotting import add_s6_cross_pore_columns
from shared.sampling_depth_analysis import latest_qc_trajectory_representatives


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "analysis/statistics_revision/tables"
DIST = ROOT / "kv21/dataDistances"
ALL_DISTANCE = ROOT / "kv21/dataExtra/conformation_analysis/all_distance_sampling/tables"
BOOTSTRAP_SEED = 20260803
BOOTSTRAP_REPLICATES = 10_000

S6_PATHS = {
    "WT": {
        "vanilla": "26-02-11_Kv2.1_wt_vanillaAF2test_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        "masked": "26-02-11_Kv2.1_wt_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    },
    "L403A": {
        "vanilla": "26-02-11_Kv2.1_l403a_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        "masked": "26-02-11_Kv2.1_l403a_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    },
    "F412L": {
        "vanilla": "26-02-11_Kv2.1_f412l_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        "masked": "26-02-11_Kv2.1_f412l_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    },
}
REPORTED_S6 = {"WT": 3.50, "L403A": 2.08, "F412L": 1.92}


def _trajectory_counts(frame: pd.DataFrame) -> tuple[int, int]:
    names = frame.pdb_file.astype(str)
    seeds = names.str.extract(r"_seed_(\d+)", expand=False)
    models = names.str.extract(r"_model_(\d+)", expand=False)
    return pd.DataFrame({"seed": seeds, "model": models}).drop_duplicates().shape[0], seeds.nunique()


def _distance_trajectory_moments(frame: pd.DataFrame, columns: list[str]):
    keys = pd.DataFrame({
        "seed": frame.pdb_file.astype(str).str.extract(r"_seed_(\d+)", expand=False),
        "model": frame.pdb_file.astype(str).str.extract(r"_model_(\d+)", expand=False),
    }, index=frame.index)
    blocks = []
    for indices in keys.groupby(["seed", "model"], sort=True).groups.values():
        values = frame.loc[indices, columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        safe = np.where(valid, values, 0.0)
        blocks.append((valid.sum(axis=0), safe.sum(axis=0), np.square(safe).sum(axis=0)))
    return tuple(np.stack([block[i] for block in blocks]).astype(float) for i in range(3))


def _sd_from_weighted_blocks(blocks, weights):
    counts, sums, sumsq = blocks
    count = weights @ counts
    total = weights @ sums
    total_sq = weights @ sumsq
    numerator = total_sq - np.divide(np.square(total), count, out=np.zeros_like(total), where=count > 0)
    variance = np.divide(numerator, count - 1, out=np.full_like(total, np.nan), where=count > 1)
    return np.sqrt(np.maximum(variance, 0.0))


def _s6_block_bootstrap(vanilla: pd.DataFrame, masked: pd.DataFrame, columns: list[str]):
    blocks = {
        "vanilla": _distance_trajectory_moments(vanilla, columns),
        "masked": _distance_trajectory_moments(masked, columns),
    }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    ratios = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for iteration in range(BOOTSTRAP_REPLICATES):
        sd = {}
        for protocol in ("vanilla", "masked"):
            n = blocks[protocol][0].shape[0]
            weights = np.bincount(rng.integers(0, n, n), minlength=n)
            sd[protocol] = _sd_from_weighted_blocks(blocks[protocol], weights)
        ratios[iteration] = np.nanmedian(sd["masked"] / sd["vanilla"])
    return ratios


def s6_source_audit() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit the current chain-label-invariant six-level S6 estimator.

    The repository contains no executable provenance for the manuscript values
    3.50/2.08/1.92. This function therefore labels the current estimates as
    revised values and bootstraps both the all-snapshot trajectory-block estimator
    and a one-representative-per-trajectory sensitivity estimator.
    """
    rows, status, replicate_rows = [], [], []
    for condition, paths in S6_PATHS.items():
        frames = {}
        aliases = None
        for protocol, filename in paths.items():
            raw = pd.read_csv(DIST / filename)
            frames[protocol], current = add_s6_cross_pore_columns(raw)
            aliases = current if aliases is None else aliases
        columns = list(aliases.values())
        trajectory_counts = {p: _trajectory_counts(f) for p, f in frames.items()}
        representatives = {p: latest_qc_trajectory_representatives(f) for p, f in frames.items()}
        ratios = frames["masked"][columns].std(ddof=1) / frames["vanilla"][columns].std(ddof=1)
        representative_ratios = (
            representatives["masked"][columns].std(ddof=1)
            / representatives["vanilla"][columns].std(ddof=1)
        )
        point = float(ratios.median())
        representative_point = float(representative_ratios.median())
        reproduced = bool(np.isclose(point, REPORTED_S6[condition], atol=0.005))
        bootstrap_sets = {
            "all_retained_snapshot_blocks": _s6_block_bootstrap(
                frames["vanilla"], frames["masked"], columns
            ),
            "latest_representative_per_trajectory": _s6_block_bootstrap(
                representatives["vanilla"], representatives["masked"], columns
            ),
        }
        for estimator, values in bootstrap_sets.items():
            replicate_rows.extend({
                "sequence_background": condition,
                "estimator": estimator,
                "replicate": i + 1,
                "median_masked_over_vanilla_SD_ratio": value,
                "bootstrap_seed": BOOTSTRAP_SEED,
            } for i, value in enumerate(values))
        primary_boot = bootstrap_sets["all_retained_snapshot_blocks"]
        representative_boot = bootstrap_sets["latest_representative_per_trajectory"]
        for alias, column in aliases.items():
            rows.append({
                "sequence_background": condition,
                "protocol_pair": "masked vs vanilla",
                "vanilla_source_file": str((DIST / paths["vanilla"]).relative_to(ROOT)),
                "masked_source_file": str((DIST / paths["masked"]).relative_to(ROOT)),
                "source_dataset_selector": "all_ok_rmsd_3A_structural_interface_alignment_qc",
                "estimator": "SD across all retained recycle snapshots; median ratio across six chain-label-invariant S6 cross-pore maxima",
                "s6_coordinate_alias": alias,
                "s6_coordinate_name": column,
                "number_s6_coordinates": len(columns),
                "vanilla_retained_rows": len(frames["vanilla"]),
                "masked_retained_rows": len(frames["masked"]),
                "vanilla_trajectories": trajectory_counts["vanilla"][0],
                "masked_trajectories": trajectory_counts["masked"][0],
                "vanilla_seeds": trajectory_counts["vanilla"][1],
                "masked_seeds": trajectory_counts["masked"][1],
                "vanilla_SD_A": frames["vanilla"][column].std(ddof=1),
                "masked_SD_A": frames["masked"][column].std(ddof=1),
                "masked_over_vanilla_SD_ratio": ratios[column],
                "representative_only_SD_ratio": representative_ratios[column],
                "median_masked_over_vanilla_SD_ratio": point,
                "reported_manuscript_ratio": REPORTED_S6[condition],
                "reported_value_reproduced": reproduced,
            })
        status.append({
            "sequence_background": condition,
            "reported_median_SD_ratio": REPORTED_S6[condition],
            "closest_current_reproducible_median_SD_ratio": point,
            "representative_only_median_SD_ratio": representative_point,
            "reported_value_reproduced": reproduced,
            "bootstrap_status": "completed for revised current estimator; not provenance for reported manuscript ratio",
            "bootstrap_median_ratio": np.nanmedian(primary_boot),
            "bootstrap_95CI_low": np.nanquantile(primary_boot, .025),
            "bootstrap_95CI_high": np.nanquantile(primary_boot, .975),
            "fraction_bootstrap_ratio_gt_1": np.nanmean(primary_boot > 1),
            "representative_bootstrap_median_ratio": np.nanmedian(representative_boot),
            "representative_bootstrap_95CI_low": np.nanquantile(representative_boot, .025),
            "representative_bootstrap_95CI_high": np.nanquantile(representative_boot, .975),
            "representative_fraction_bootstrap_ratio_gt_1": np.nanmean(representative_boot > 1),
            "representative_percent_difference_from_primary": 100 * (representative_point / point - 1),
            "representative_sensitivity_exceeds_10pct": abs(representative_point / point - 1) > .10,
            "qualitative_conclusion_agrees_ratio_gt_1": (point > 1) == (representative_point > 1),
            "bootstrap_unit": "complete seed-model trajectory; all retained recycles remain together",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        })
    return pd.DataFrame(rows), pd.DataFrame(status), pd.DataFrame(replicate_rows)


def _selected_rmsf_rows(metadata: pd.DataFrame) -> pd.DataFrame:
    selected = []
    paths = sorted(DIST.glob("*_all_ok_rmsd_3A_structural_interface_qc.csv"))
    if len(paths) != 6:
        raise FileNotFoundError(f"Expected six Kv2.1 RMSF QC allowlists, found {len(paths)}")
    lookup = metadata[["manifest_index", "pdb_file", "dataset", "sequence_condition", "protocol",
                       "trajectory_id", "seed", "model_number"]].copy()
    lookup["basename"] = lookup.pdb_file.astype(str).map(lambda x: Path(x).name)
    for path in paths:
        names = read_csv_resolving_lfs(path, ROOT, usecols=["pdb_file"])
        names["basename"] = names.pdb_file.astype(str).map(lambda x: Path(x).name)
        mapped = names[["basename"]].merge(lookup, on="basename", how="left", validate="one_to_one")
        if mapped.manifest_index.isna().any():
            raise KeyError(f"Unmapped RMSF QC rows in {path.name}")
        selected.append(mapped)
    return pd.concat(selected, ignore_index=True)


def _trajectory_moments(coords, present, group: pd.DataFrame):
    blocks = []
    for _, trajectory in group.groupby(["seed", "model_number"], sort=True):
        idx = trajectory.manifest_index.astype(int).to_numpy()
        xyz = np.asarray(coords[idx], dtype=np.float64)
        valid = np.asarray(present[idx], dtype=bool) & np.isfinite(xyz).all(axis=-1)
        safe = np.where(valid[..., None], xyz, 0.0)
        blocks.append((valid.sum(axis=0), safe.sum(axis=0), np.square(safe).sum(axis=(0, 3))))
    counts = np.stack([x[0] for x in blocks]).astype(np.float64)
    sums = np.stack([x[1] for x in blocks])
    sumsq = np.stack([x[2] for x in blocks])
    return counts, sums, sumsq


def _rmsf_from_weighted_blocks(blocks, weights):
    counts, sums, sumsq = blocks
    count = np.tensordot(weights, counts, axes=(0, 0))
    total = np.tensordot(weights, sums, axes=(0, 0))
    total_sq = np.tensordot(weights, sumsq, axes=(0, 0))
    mean = np.divide(total, count[..., None], out=np.zeros_like(total), where=count[..., None] > 0)
    variance = np.divide(total_sq, count, out=np.zeros_like(total_sq), where=count > 0)
    variance -= np.square(mean).sum(axis=-1)
    rmsf = np.sqrt(np.maximum(variance, 0.0))
    rmsf[count == 0] = np.nan
    return np.nanmean(rmsf, axis=0)


def rmsf_block_bootstrap() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recompute RMSF after resampling complete model/seed trajectories."""
    merged = ROOT / "kv21/dataRMSF/merged"
    metadata = read_csv_resolving_lfs(merged / "kv21_alignment_metadata.csv", ROOT)
    coords = np.load(resolve_local_lfs_object(merged / "kv21_aligned_ca_coordinates.npy", ROOT), mmap_mode="r")
    present = np.load(resolve_local_lfs_object(merged / "kv21_aligned_ca_present.npy", ROOT), mmap_mode="r")
    selected = _selected_rmsf_rows(metadata)
    profile = read_csv_resolving_lfs(
        ROOT / "kv21/dataRMSF/profiles/kv21_all_ok_3_symmetry_averaged_profiles.csv", ROOT
    )
    mask = profile.loc[profile.dataset.eq("wt_masked")].sort_values("raw_residue_number").directly_masked.to_numpy(bool)
    rows, audit = [], []
    for condition in ("wt", "l403a", "f412l"):
        blocks = {}
        for protocol in ("vanilla", "masked"):
            group = selected[(selected.sequence_condition == condition) & (selected.protocol == protocol)]
            blocks[protocol] = _trajectory_moments(coords, present, group)
        observed = {p: _rmsf_from_weighted_blocks(b, np.ones(b[0].shape[0])) for p, b in blocks.items()}
        observed_delta = observed["masked"] - observed["vanilla"]
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        boot = np.empty((BOOTSTRAP_REPLICATES, 2), dtype=float)
        for iteration in range(BOOTSTRAP_REPLICATES):
            profiles = {}
            for protocol in ("vanilla", "masked"):
                n = blocks[protocol][0].shape[0]
                weights = np.bincount(rng.integers(0, n, n), minlength=n)
                profiles[protocol] = _rmsf_from_weighted_blocks(blocks[protocol], weights)
            delta = profiles["masked"] - profiles["vanilla"]
            boot[iteration] = [np.nanmedian(delta[mask]), np.nanmedian(delta[~mask])]
        for index, region in enumerate(("directly_masked_positions", "outside_direct_mask")):
            values = boot[:, index]
            rows.append({
                "sequence_background": condition.upper(), "region": region,
                "observed_median_masked_minus_vanilla_RMSF_A": np.nanmedian(observed_delta[mask if index == 0 else ~mask]),
                "bootstrap_median_A": np.nanmedian(values),
                "bootstrap_95CI_low_A": np.quantile(values, .025),
                "bootstrap_95CI_high_A": np.quantile(values, .975),
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "bootstrap_unit": "complete model-seed trajectory",
            })
        audit.append({"sequence_background": condition.upper(),
                      "vanilla_trajectories": blocks["vanilla"][0].shape[0],
                      "masked_trajectories": blocks["masked"][0].shape[0],
                      "coordinate_source": "kv21/dataRMSF/merged/kv21_aligned_ca_coordinates.npy",
                      "selection_source": "six *_all_ok_rmsd_3A_structural_interface_qc.csv allowlists"})
    return pd.DataFrame(rows), pd.DataFrame(audit)


def manuscript_statistics(s6_status: pd.DataFrame, rmsf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    breadth = pd.read_csv(ALL_DISTANCE / "l403a_first100_seed_level_global_breadth_summary.csv").iloc[0]
    retention = pd.read_csv(ALL_DISTANCE / "l403a_first100_nominal_trajectory_qc_summary.csv").set_index("protocol")
    fisher = pd.read_csv(ALL_DISTANCE / "l403a_first100_qc_retention_fisher_test.csv").iloc[0]
    rows.append({"section": "First100 L403A", "metric": "global normalized seed breadth",
                 **breadth.to_dict(), "nominal_trajectories_per_protocol": 100,
                 "retained_vanilla": retention.loc["vanilla", "retained_trajectories"],
                 "retained_masked": retention.loc["masked", "retained_trajectories"],
                 "retention_fisher_exact_p": fisher.p})
    first = pd.read_csv(ALL_DISTANCE / "l403a_first100_seed_block_distribution_statistics_summary.csv").iloc[0]
    first_detail = pd.read_csv(ALL_DISTANCE / "l403a_first100_seed_block_distribution_statistics.csv")
    first_descriptive = first_detail.IQR_ratio_masked_over_vanilla.gt(1)
    by_type = pd.read_csv(ALL_DISTANCE / "l403a_first100_vs_full_breadth_by_distance_type.csv")
    rows.append({"section": "First100 distance-wise breadth", "metric": "all 546 distances", **first.to_dict(),
                 "n_descriptively_broader_masked": int(first_descriptive.sum()),
                 "fraction_descriptively_broader_masked": float(first_descriptive.mean()),
                 "CA_median_IQR_ratio": by_type.query("cohort == 'Nominal first 100' and distance_type == 'Cα'").median_IQR_ratio.iloc[0],
                 "CA_fraction_significantly_broader": by_type.query("cohort == 'Nominal first 100' and distance_type == 'Cα'").fraction_significantly_broader.iloc[0],
                 "shortest_heavy_median_IQR_ratio": by_type.query("cohort == 'Nominal first 100' and distance_type == 'Shortest-heavy'").median_IQR_ratio.iloc[0],
                 "shortest_heavy_fraction_significantly_broader": by_type.query("cohort == 'Nominal first 100' and distance_type == 'Shortest-heavy'").fraction_significantly_broader.iloc[0]})
    full = pd.read_csv(ALL_DISTANCE / "l403a_full_seed_block_distribution_statistics_summary.csv").iloc[0]
    concordance = pd.read_csv(ALL_DISTANCE / "l403a_first100_vs_full_broadening_concordance.csv", index_col=0)
    agreement = float(np.trace(concordance.to_numpy()))
    rows.append({"section": "Full-depth robustness", "metric": "all 546 distances", **full.to_dict(),
                 "first100_full_concordant_count": agreement,
                 "first100_full_concordant_fraction": agreement / concordance.to_numpy().sum()})
    saturation = pd.read_csv(ALL_DISTANCE / "l403a_random_seed_saturation_summary.csv")
    for _, row in saturation.iterrows():
        rows.append({"section": "Random-seed saturation", "metric": f"{int(row.seeds_sampled)} seeds",
                     "median_masked_over_vanilla_normalized_breadth_ratio": row.breadth_gain_masked_over_vanilla_median,
                     "empirical_2.5pct": row.breadth_gain_masked_over_vanilla_CI_low,
                     "empirical_97.5pct": row.breadth_gain_masked_over_vanilla_CI_high,
                     "fraction_random_draws_ratio_gt_1": row.fraction_draws_gain_gt_1})
    for _, row in s6_status.iterrows():
        rows.append({"section": "Full-ensemble S6", "metric": row.sequence_background,
                     **row.to_dict()})
    for _, row in rmsf.iterrows():
        rows.append({"section": "RMSF", "metric": f"{row.sequence_background} {row.region}", **row.to_dict()})
    return pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    audit, s6_status, s6_replicates = s6_source_audit()
    audit.to_csv(OUT / "kv21_s6_breadth_source_audit.csv", index=False)
    s6_status.to_csv(OUT / "kv21_s6_masked_vs_vanilla_breadth_bootstrap.csv", index=False)
    s6_replicates.to_csv(OUT / "kv21_s6_breadth_bootstrap_replicates.csv", index=False)
    rmsf, rmsf_audit = rmsf_block_bootstrap()
    rmsf.to_csv(OUT / "kv21_rmsf_trajectory_block_bootstrap.csv", index=False)
    rmsf_audit.to_csv(OUT / "kv21_rmsf_bootstrap_source_audit.csv", index=False)
    manuscript_statistics(s6_status, rmsf).to_csv(
        OUT / "kv21_sampling_breadth_manuscript_statistics.csv", index=False
    )
    report = {
        "first100_ratio_summary": str(ALL_DISTANCE / "l403a_first100_seed_level_global_breadth_summary.csv"),
        "s6_all_reported_values_reproduced": bool(s6_status.reported_value_reproduced.all()),
        "s6_bootstrap_status": "completed for revised current estimator; historical manuscript values remain unreproduced",
        "rmsf_bootstrap_status": "completed from aligned C-alpha arrays with trajectory-block resampling",
    }
    (OUT / "kv21_sampling_breadth_uncertainty_run_summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(s6_status.to_string(index=False))
    print(rmsf.to_string(index=False))
    return report


if __name__ == "__main__":
    run()
