#!/usr/bin/env python3
"""Run seed-block inference for the structural-ensemble analysis.

Outputs are additive; source coordinates, tables, notebooks, and figures are
not overwritten.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shared.distribution_statistics import parse_trajectory_metadata
from shared.seed_block_statistics import analyze_seed_distance, leave_one_model_out
from scripts.ensemble_rmsf_analysis.io import resolve_local_lfs_object


BASE_SEED = 20260819
L403_COLUMNS = [f"CA_CA_{chain}_GLU425_CA-{chain}_ASN181_CA" for chain in "ABCD"]
GATE_COLUMNS = [
    "CA_MET415_CA-ALA742_CA", "CA_MET415_CA-ILE1154_CA",
    "CA_MET415_CA-ILE1455_CA", "CA_ALA742_CA-ILE1154_CA",
    "CA_ALA742_CA-ILE1455_CA", "CA_ILE1154_CA-ILE1455_CA",
]

FINAL_DISTANCE_PATHS = {
    "kv21|WT|vanilla": "kv21/dataDistances/26-02-11_Kv2.1_wt_vanillaAF2test_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    "kv21|WT|masked": "kv21/dataDistances/26-02-11_Kv2.1_wt_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    "kv21|L403A|vanilla": "kv21/dataDistances/26-02-11_Kv2.1_l403a_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    "kv21|L403A|masked": "kv21/dataDistances/26-02-11_Kv2.1_l403a_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    "kv21|F412L|vanilla": "kv21/dataDistances/26-02-11_Kv2.1_f412l_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    "kv21|F412L|masked": "kv21/dataDistances/26-02-11_Kv2.1_f412l_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    "nav15|WT|vanilla": "nav15/dataDistances/26-07-27_Nav15_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
    "nav15|WT|masked": "nav15/dataDistances/26-07-25_Nav15_wt_masked_AF2_distances_extra_ifm_all_ok_rmsd_3A.csv",
    "nav15|WT|masked_v2": "nav15/dataDistances/26-07-27_Nav15_wt_maskedv2_AF2_distances_all_ok_rmsd_3A.csv",
    "nav15|WT|masked_v2_noIFM": "nav15/dataDistances/26-07-27_Nav15_wt_maskedv2_noIFM_AF2_distances_all_ok_rmsd_3A.csv",
    "nav15|QQQ|vanilla": "nav15/dataDistances/26-07-27_Nav15_qqq_vanilla_AF2_distances_all_ok_rmsd_3A.csv",
    "nav15|QQQ|masked": "nav15/dataDistances/26-07-27_Nav15_qqq_masked_AF2_distances_all_ok_rmsd_3A.csv",
    "nav15|QQQ|masked_v2": "nav15/dataDistances/26-07-27_Nav15_qqq_maskedv2_AF2_distances_all_ok_rmsd_3A.csv",
    "cav12|WT|vanilla": "cav12/dataDistances/26-02-10_Cav12_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
    "cav12|WT|masked": "cav12/dataDistances/26-02-10_Cav12_wt_maskedAF2_distances_all_ok_rmsd_3A.csv",
    "cav12|G402S|vanilla": "cav12/dataDistances/26-02-10_Cav12_g402s_vanillaAF2_distances_all_ok_rmsd_3A.csv",
    "cav12|G402S|masked": "cav12/dataDistances/26-02-10_Cav12_g402s_maskedAF2_distances_all_ok_rmsd_3A.csv",
    "cav12|G406R|vanilla": "cav12/dataDistances/26-07-25_Cav1.2_g406r_vanillaAF2_distances_all_ok_rmsd_3A.csv",
    "cav12|G406R|masked": "cav12/dataDistances/26-07-25_Cav1.2_g406r_maskedAF2_distances_all_ok_rmsd_3A.csv",
}


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    """Read a regular CSV or its locally available Git LFS object."""
    resolved = resolve_local_lfs_object(path, ROOT)
    return pd.read_csv(resolved, **kwargs)


def add_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    return parse_trajectory_metadata(frame)


def trajectory_to_seed(
    frame: pd.DataFrame, value: str, *, within_trajectory: str = "mean"
) -> pd.Series:
    grouped = frame.groupby(["seed", "model_number"])[value]
    if within_trajectory == "mean":
        trajectories = grouped.mean()
    elif within_trajectory == "median":
        trajectories = grouped.median()
    else:
        raise ValueError(within_trajectory)
    return trajectories.groupby("seed").mean().dropna().sort_index()


def bootstrap_contrast(
    a: pd.Series,
    b: pd.Series,
    *,
    replicates: int,
    seed: int,
    ratio: bool = False,
) -> dict[str, float | int]:
    """Independent whole-seed bootstrap of B-A or log(B/A)."""
    av, bv = a.to_numpy(float), b.to_numpy(float)
    rng = np.random.default_rng(seed)
    draw_a = av[rng.integers(0, len(av), size=(replicates, len(av)))].mean(axis=1)
    draw_b = bv[rng.integers(0, len(bv), size=(replicates, len(bv)))].mean(axis=1)
    if ratio:
        with np.errstate(divide="ignore", invalid="ignore"):
            samples = np.log(draw_b / draw_a)
            point = math.log(bv.mean() / av.mean())
        finite = samples[np.isfinite(samples)]
        return {
            "estimate_log_ratio": point,
            "estimate_ratio": math.exp(point),
            "CI_low_log_ratio": float(np.quantile(finite, .025)) if len(finite) else np.nan,
            "CI_high_log_ratio": float(np.quantile(finite, .975)) if len(finite) else np.nan,
            "CI_low_ratio": float(np.exp(np.quantile(finite, .025))) if len(finite) else np.nan,
            "CI_high_ratio": float(np.exp(np.quantile(finite, .975))) if len(finite) else np.nan,
            "nonfinite_bootstrap_fraction": float(1 - len(finite) / len(samples)),
            "bootstrap_replicates": replicates,
        }
    samples = draw_b - draw_a
    return {
        "estimate_B_minus_A": float(bv.mean() - av.mean()),
        "CI_low_B_minus_A": float(np.quantile(samples, .025)),
        "CI_high_B_minus_A": float(np.quantile(samples, .975)),
        "bootstrap_replicates": replicates,
    }


def experimental_ca_distances(path: Path) -> dict[str, float]:
    coordinates: dict[tuple[str, str, int], tuple[float, float, float]] = {}
    with path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            residue, chain = line[17:20].strip(), line[21].strip()
            try:
                number = int(line[22:26])
            except ValueError:
                continue
            if (residue, number) not in {("GLU", 423), ("ASN", 179)}:
                continue
            coordinates[(chain, residue, number)] = (
                float(line[30:38]), float(line[38:46]), float(line[46:54])
            )
    result = {}
    for chain in "ABCD":
        result[chain] = math.dist(
            coordinates[(chain, "GLU", 423)], coordinates[(chain, "ASN", 179)]
        )
    return result


def derive_l403a_threshold(output: Path) -> tuple[float, np.ndarray, np.ndarray]:
    wt = experimental_ca_distances(ROOT / "kv21/experimental/8SD3.pdb")
    mutant = experimental_ca_distances(ROOT / "kv21/experimental/8SDA.pdb")
    wt_max = max(wt.values())
    shifted = sorted(value for value in mutant.values() if value > wt_max)
    if len(shifted) != 2:
        raise ValueError(f"Expected two 8SDA distances above the WT maximum, observed {shifted}")
    threshold = (wt_max + shifted[0]) / 2
    rows = []
    for structure, values in (("8SD3", wt), ("8SDA", mutant)):
        for chain, value in values.items():
            rows.append({
                "structure": structure, "chain": chain, "E423_N179_CA_distance_A": value,
                "classification_at_derived_threshold": "shifted" if value >= threshold else "WT-like",
            })
    table = pd.DataFrame(rows)
    table["derived_threshold_A"] = threshold
    table["derivation"] = "midpoint(maximum 8SD3 value, minimum 8SDA value above the 8SD3 maximum)"
    save(table, output / "l403a_experimental_threshold_derivation.csv")
    return threshold, np.sort(np.fromiter(wt.values(), float)), np.sort(np.fromiter(mutant.values(), float))


def select_one(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    numbered = frame.dropna(subset=["recycle_number"]).sort_values(
        ["seed", "model_number", "recycle_number", "pdb_file"]
    )
    return numbered.groupby(["seed", "model_number"], as_index=False).nth(
        0 if rule == "earliest" else -1
    ).reset_index(drop=True)


def l403a_analysis(output: Path, bootstrap: int, permutations: int) -> dict[str, pd.Series]:
    threshold, wt_vector, mutant_vector = derive_l403a_threshold(output)
    source = ROOT / "kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv"
    frame = add_metadata(read_csv(source))
    frame[L403_COLUMNS] = frame[L403_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame["max_distance_A"] = frame[L403_COLUMNS].max(axis=1)
    frame["shifted_subunits"] = frame[L403_COLUMNS].ge(threshold).sum(axis=1)
    frame["any_shifted"] = frame["shifted_subunits"].ge(1).astype(float)
    ordered = np.sort(frame[L403_COLUMNS].to_numpy(float), axis=1)
    frame["RMSE_to_8SDA_vector_A"] = np.sqrt(np.mean(np.square(ordered - mutant_vector), axis=1))
    frame["RMSE_to_8SD3_vector_A"] = np.sqrt(np.mean(np.square(ordered - wt_vector), axis=1))
    frame["closer_to_8SDA_vector"] = (
        frame["RMSE_to_8SDA_vector_A"] < frame["RMSE_to_8SD3_vector_A"]
    ).astype(float)

    parts = {name: part.copy() for name, part in frame.groupby("condition")}
    seed_values: dict[str, dict[str, pd.Series]] = {"vanilla": {}, "masked": {}}
    definitions = {
        "continuous_max_distance_A": ("max_distance_A", "median"),
        "any_shifted_fraction": ("any_shifted", "mean"),
        "RMSE_to_8SDA_vector_A": ("RMSE_to_8SDA_vector_A", "median"),
        "closer_to_8SDA_vector_fraction": ("closer_to_8SDA_vector", "mean"),
    }
    rows = []
    for metric_index, (metric, (column, reduction)) in enumerate(definitions.items()):
        for protocol, part in parts.items():
            seed_values[protocol][metric] = trajectory_to_seed(
                part, column, within_trajectory=reduction
            )
        contrast = bootstrap_contrast(
            seed_values["vanilla"][metric], seed_values["masked"][metric],
            replicates=bootstrap, seed=BASE_SEED + metric_index,
        )
        rows.append({
            "metric": metric,
            "within_trajectory_reduction": reduction,
            "vanilla_seed_balanced_estimate": seed_values["vanilla"][metric].mean(),
            "masked_seed_balanced_estimate": seed_values["masked"][metric].mean(),
            **contrast,
            "resampling_unit": "seed; five AF2 models retained as within-seed strata",
        })
    ratio = bootstrap_contrast(
        seed_values["vanilla"]["any_shifted_fraction"],
        seed_values["masked"]["any_shifted_fraction"],
        replicates=bootstrap, seed=BASE_SEED + 20, ratio=True,
    )
    rows.append({
        "metric": "any_shifted_risk_ratio_secondary",
        "within_trajectory_reduction": "fraction",
        "vanilla_seed_balanced_estimate": seed_values["vanilla"]["any_shifted_fraction"].mean(),
        "masked_seed_balanced_estimate": seed_values["masked"]["any_shifted_fraction"].mean(),
        **ratio,
        "resampling_unit": "seed; log risk ratio",
    })
    for rule_index, rule in enumerate(("earliest", "latest")):
        reduced = {protocol: select_one(part, rule) for protocol, part in parts.items()}
        values = {
            protocol: trajectory_to_seed(part, "any_shifted", within_trajectory="mean")
            for protocol, part in reduced.items()
        }
        rows.append({
            "metric": f"any_shifted_fraction_{rule}",
            "within_trajectory_reduction": rule,
            "vanilla_seed_balanced_estimate": values["vanilla"].mean(),
            "masked_seed_balanced_estimate": values["masked"].mean(),
            **bootstrap_contrast(values["vanilla"], values["masked"], replicates=bootstrap,
                                 seed=BASE_SEED + 30 + rule_index),
            "resampling_unit": "seed",
        })
    save(pd.DataFrame(rows), output / "l403a_seed_block_contrasts.csv")

    occupancy = []
    for number in range(5):
        values = {}
        for protocol, part in parts.items():
            work = part.assign(_category=part["shifted_subunits"].eq(number).astype(float))
            values[protocol] = trajectory_to_seed(work, "_category", within_trajectory="mean")
        occupancy.append({
            "shifted_subunits": number,
            "vanilla_probability": values["vanilla"].mean(),
            "masked_probability": values["masked"].mean(),
            **bootstrap_contrast(values["vanilla"], values["masked"], replicates=bootstrap,
                                 seed=BASE_SEED + 40 + number),
        })
    save(pd.DataFrame(occupancy), output / "l403a_shifted_subunit_distribution_seed_block.csv")

    w1 = analyze_seed_distance(
        parts["vanilla"], parts["masked"], "max_distance_A",
        n_permutations=permutations, n_bootstrap=bootstrap, random_seed=BASE_SEED + 50,
        within_trajectory_reduction="median",
    )
    save(pd.DataFrame([w1]), output / "l403a_continuous_max_distance_seed_block_w1.csv")
    sensitivity = leave_one_model_out(
        parts["vanilla"], parts["masked"], "max_distance_A",
        within_trajectory_reduction="median",
    )
    sensitivity.insert(0, "metric", "continuous_max_distance_A")
    save(sensitivity, output / "l403a_leave_one_AF2_model_out.csv")
    return {protocol: values["any_shifted_fraction"] for protocol, values in seed_values.items()} | {
        f"{protocol}_continuous": values["continuous_max_distance_A"]
        for protocol, values in seed_values.items()
    }


def master_cohort(output: Path) -> pd.DataFrame:
    manifests = {
        "kv21": ROOT / "kv21/dataRMSF/qc/kv21_all_ok3_selection_manifest.csv",
        "nav15": ROOT / "nav15/dataRMSF/qc/nav15_all_ok3_selection_manifest.csv",
        "cav12": ROOT / "cav12/dataRMSF/qc/cav12_all_ok3_selection_manifest.csv",
    }
    pieces = []
    for channel, path in manifests.items():
        frame = read_csv(path)
        frame["channel"] = channel
        # Some Cav1.2 manifests retain condition="unknown" even though the
        # authoritative dataset key carries the sequence background.
        frame["sequence"] = frame["dataset"].str.split("_").str[0].str.upper()
        frame["protocol"] = frame["dataset"].str.split("_", n=1).str[1].replace({
            "masked_v2_noifm": "masked_v2_noIFM",
        })
        frame["ensemble_id"] = frame[["channel", "sequence", "protocol"]].agg("|".join, axis=1)
        pieces.append(frame)
    master = pd.concat(pieces, ignore_index=True)
    master["_dataset_key"] = master["dataset"].str.lower()

    # Mapping/alignment QC is encoded in the RMSF alignment metadata rather
    # than in the convergence manifests.  Join it explicitly so mapping is a
    # visible cohort stage instead of an unresolved proxy.
    mapping_pieces = []
    for channel in ("kv21", "nav15", "cav12"):
        path = ROOT / f"{channel}/dataRMSF/merged/{channel}_alignment_metadata.csv"
        mapping = read_csv(path)
        mapping["channel"] = channel
        mapping["_dataset_key"] = mapping["dataset"].str.lower()
        mapping["pdb_basename"] = mapping["pdb_file"].map(lambda value: Path(str(value)).name)
        status_columns = [column for column in mapping if column == "status" or (
            column.startswith("chain_") and column.endswith("_status")
        )]
        status_ok = pd.Series(True, index=mapping.index)
        for column in status_columns:
            status_ok &= mapping[column].astype(str).str.lower().eq("ok")
        mapping["mapping_metadata_membership"] = True
        mapping["mapping_QC_pass"] = mapping["alignment_success"].fillna(False) & status_ok
        mapping["mapping_status"] = np.where(
            mapping["mapping_QC_pass"], "pass",
            mapping.get("alignment_error", pd.Series("", index=mapping.index)).fillna("").replace("", "failed"),
        )
        mapping["mapping_QC_source"] = str(path.relative_to(ROOT))
        mapping_pieces.append(mapping[[
            "channel", "_dataset_key", "pdb_basename", "mapping_metadata_membership",
            "mapping_QC_pass", "mapping_status", "mapping_QC_source", "matched_core_ca",
            "core_coverage",
        ]].rename(columns={
            "matched_core_ca": "mapping_matched_core_ca",
            "core_coverage": "mapping_core_coverage",
        }))
    mapping = pd.concat(mapping_pieces, ignore_index=True)
    if mapping.duplicated(["channel", "_dataset_key", "pdb_basename"]).any():
        raise ValueError("Alignment metadata contains duplicate channel/dataset/PDB mapping keys")
    master = master.merge(
        mapping, on=["channel", "_dataset_key", "pdb_basename"],
        how="left", validate="one_to_one",
    )
    master["mapping_metadata_membership"] = master["mapping_metadata_membership"].fillna(False)
    master["mapping_QC_pass"] = master["mapping_QC_pass"].fillna(False)
    master["distance_source_membership"] = False
    master["distance_final_source"] = ""
    for ensemble, relative in FINAL_DISTANCE_PATHS.items():
        selected = set(read_csv(ROOT / relative, usecols=["pdb_file"])["pdb_file"].map(lambda x: Path(str(x)).name))
        mask = master["ensemble_id"].eq(ensemble)
        master.loc[mask, "distance_source_membership"] = master.loc[mask, "pdb_basename"].isin(selected)
        master.loc[mask, "distance_final_source"] = relative
    # A file name containing "all_ok" is not sufficient provenance for final
    # QC (the legacy WT-standard-mask NaV1.5 table also contains 500 r0 rows).
    master["distance_final_cohort"] = (
        master["distance_source_membership"] & master["all_ok_3"].fillna(False)
    )
    master["exclusion_reason"] = np.select(
        [
            ~master["parse_ok"].fillna(False),
            ~master["mapping_metadata_membership"],
            ~master["mapping_QC_pass"],
            ~master["all_ok"].fillna(False),
            ~master["all_ok_3"].fillna(False),
            ~master["distance_final_cohort"],
        ],
        [
            "filename_or_structure_parse_failure",
            "not_present_in_alignment_mapping_metadata",
            "did_not_pass_alignment_mapping_QC",
            "did_not_pass_recycle_convergence",
            "did_not_pass_structural_integrity_QC",
            "not_in_analysis_specific_final_distance_cohort",
        ],
        default="included_in_final_distance_cohort",
    )
    keep = [
        "channel", "sequence", "protocol", "ensemble_id", "pdb_basename", "seed",
        "model_number", "recycle_number", "parse_ok", "all_ok", "all_ok_3",
        "mapping_metadata_membership", "mapping_QC_pass", "mapping_status",
        "mapping_matched_core_ca", "mapping_core_coverage", "mapping_QC_source",
        "distance_source_membership", "distance_final_cohort", "exclusion_reason",
        "distance_final_source",
    ]
    save(master[keep], output / "master_structure_cohort.csv")
    flow = master.groupby(["channel", "sequence", "protocol", "ensemble_id"]).agg(
        nominal_structures=("pdb_basename", "size"), nominal_seeds=("seed", "nunique"),
        mapping_QC_snapshots=("mapping_QC_pass", "sum"),
        converged_snapshots=("all_ok", "sum"), final_QC_snapshots=("all_ok_3", "sum"),
        analysis_final_snapshots=("distance_final_cohort", "sum"),
    ).reset_index()
    trajectories = master.groupby(["ensemble_id", "seed", "model_number"]).agg(
        mapping_QC=("mapping_QC_pass", "max"), converged=("all_ok", "max"),
        final_QC=("all_ok_3", "max"),
        analysis_final=("distance_final_cohort", "max"),
    ).reset_index().groupby("ensemble_id").agg(
        nominal_trajectories=("model_number", "size"),
        mapping_QC_trajectories=("mapping_QC", "sum"),
        converged_trajectories=("converged", "sum"), final_QC_trajectories=("final_QC", "sum"),
        analysis_final_trajectories=("analysis_final", "sum"),
    ).reset_index()
    flow = flow.merge(trajectories, on="ensemble_id", how="left")
    flow["mapping_QC_status"] = "explicit per-structure alignment metadata joined by channel/dataset/PDB basename"
    save(flow, output / "master_cohort_flow_summary.csv")
    return master


def retention_analysis(master: pd.DataFrame, output: Path, bootstrap: int) -> None:
    trajectory = master.groupby(
        ["channel", "sequence", "protocol", "ensemble_id", "seed", "model_number"]
    ).agg(
        converged=("all_ok", "max"), final_QC=("all_ok_3", "max"),
        analysis_final=("distance_final_cohort", "max"),
    ).reset_index()
    rows, contrasts, loo = [], [], []
    seed_tables = {}
    for keys, part in trajectory.groupby(["channel", "sequence", "protocol", "ensemble_id"]):
        channel, sequence, protocol, ensemble = keys
        seed_table = part.groupby("seed")[["converged", "final_QC", "analysis_final"]].mean()
        seed_tables[ensemble] = seed_table
        for stage in seed_table:
            rng = np.random.default_rng(BASE_SEED + len(rows))
            values = seed_table[stage].to_numpy(float)
            samples = values[rng.integers(0, len(values), size=(bootstrap, len(values)))].mean(axis=1)
            rows.append({
                "channel": channel, "sequence": sequence, "protocol": protocol,
                "stage": stage, "estimate": values.mean(), "CI_low": np.quantile(samples, .025),
                "CI_high": np.quantile(samples, .975), "seeds": len(values),
                "nominal_model_trajectories": len(part),
            })
        for omitted in sorted(part.model_number.unique()):
            subset = part[part.model_number.ne(omitted)]
            estimates = subset.groupby("seed")[["converged", "final_QC", "analysis_final"]].mean().mean()
            for stage, estimate in estimates.items():
                loo.append({"ensemble_id": ensemble, "omitted_model_number": omitted,
                            "stage": stage, "estimate": estimate})
    for (channel, sequence), group in trajectory.groupby(["channel", "sequence"]):
        protocols = sorted(group.protocol.unique())
        if "vanilla" not in protocols:
            continue
        vanilla_id = f"{channel}|{sequence}|vanilla"
        if vanilla_id not in seed_tables:
            continue
        for protocol in [item for item in protocols if item != "vanilla"]:
            other_id = f"{channel}|{sequence}|{protocol}"
            if other_id not in seed_tables:
                continue
            for stage in ("converged", "final_QC", "analysis_final"):
                contrasts.append({
                    "channel": channel, "sequence": sequence,
                    "contrast": f"{protocol}-vanilla", "stage": stage,
                    **bootstrap_contrast(
                        seed_tables[vanilla_id][stage], seed_tables[other_id][stage],
                        replicates=bootstrap, seed=BASE_SEED + 200 + len(contrasts),
                    ),
                })
    save(pd.DataFrame(rows), output / "qc_retention_seed_block.csv")
    save(pd.DataFrame(contrasts), output / "qc_retention_contrasts_seed_block.csv")
    save(pd.DataFrame(loo), output / "qc_retention_leave_one_AF2_model_out.csv")


def f412l_analysis(output: Path, bootstrap: int) -> None:
    source = ROOT / "kv21/dataRMSD/analysis/comparison_v5/f412l_pocket_D_paper_nexus_shortest_contacts_long_v5.csv"
    long = read_csv(source)
    value = "Shortest heavy-atom distance (Å)"
    wide = long.pivot_table(index=["Protocol", "pdb_file"], columns="Contact", values=value, aggfunc="first").reset_index()
    rename = {}
    for column in wide:
        if "L412–L316" in str(column): rename[column] = "L412_L316_A"
        if "L412–L329" in str(column): rename[column] = "L412_L329_A"
        if "L412–L403" in str(column): rename[column] = "L412_L403_A"
    wide = add_metadata(wide.rename(columns=rename))
    rows = []
    for metric_index, column in enumerate(rename.values()):
        values = {}
        for protocol, part in wide.groupby(wide.Protocol.str.lower()):
            work = part.assign(_contact=pd.to_numeric(part[column], errors="coerce").le(4).astype(float))
            values[protocol] = trajectory_to_seed(work, "_contact", within_trajectory="mean")
        rows.append({
            "contact": column, "outcome": "fraction_distance_le_4A",
            "vanilla_seed_balanced": values["vanilla"].mean(),
            "masked_seed_balanced": values["masked"].mean(),
            **bootstrap_contrast(values["vanilla"], values["masked"], replicates=bootstrap,
                                 seed=BASE_SEED + 300 + metric_index),
        })
    save(pd.DataFrame(rows), output / "f412l_direct_seed_block_contrasts.csv")

    masked = wide[wide.Protocol.str.lower().eq("masked")].sort_values(
        ["seed", "model_number", "recycle_number", "pdb_file"]
    ).groupby(["seed", "model_number"], as_index=False).tail(1)
    distance_path = ROOT / FINAL_DISTANCE_PATHS["kv21|F412L|masked"]
    distances = read_csv(distance_path)
    local = [column for column in distances if column.startswith("shortest_") and "LEU414" in column]
    distances = distances[["pdb_file", *local]].copy()
    distances["minimum_L412_centered_local_distance_A"] = distances[local].apply(
        pd.to_numeric, errors="coerce"
    ).min(axis=1)
    candidates = masked.merge(
        distances[["pdb_file", "minimum_L412_centered_local_distance_A"]],
        on="pdb_file", how="left", validate="one_to_one",
    )
    contact_columns = list(rename.values())
    candidates["passes_local_2A_screen"] = candidates["minimum_L412_centered_local_distance_A"].ge(2)
    candidates["all_three_contacts_weakened"] = candidates[contact_columns].gt(4).all(axis=1)
    eligible = candidates[candidates.passes_local_2A_screen & candidates.all_three_contacts_weakened].copy()
    center = eligible[contact_columns].median()
    scale = (eligible[contact_columns].quantile(.75) - eligible[contact_columns].quantile(.25)).replace(0, 1)
    eligible["robust_scaled_distance_to_subpopulation_medoid"] = np.sqrt(
        np.square((eligible[contact_columns] - center) / scale).sum(axis=1)
    )
    selected_name = eligible.sort_values(
        ["robust_scaled_distance_to_subpopulation_medoid", "pdb_file"]
    ).iloc[0].pdb_file
    candidates = candidates.merge(
        eligible[["pdb_file", "robust_scaled_distance_to_subpopulation_medoid"]],
        on="pdb_file", how="left",
    )
    candidates["selection_status"] = np.where(
        candidates.pdb_file.eq(selected_name), "objective_medoid_representative", "not_selected"
    )
    candidates["selection_rule"] = (
        "latest final-QC contact-mapped snapshot per seed-model trajectory; all 54 "
        "L412-centered local distances >=2 A; all three prespecified contacts >4 A; "
        "minimum robust-scaled distance to the eligible population median"
    )
    candidates["structure_file_available_locally"] = False
    save(candidates.sort_values(["selection_status", "pdb_file"]), output / "f412l_objective_representative_selection.csv")


def seed_bootstrap_spearman(frame: pd.DataFrame, x: str, y: str, bootstrap: int, seed: int) -> tuple[float, float, float]:
    blocks = [part[[x, y]].dropna().to_numpy(float) for _, part in frame.groupby("seed")]
    point = float(spearmanr(frame[x], frame[y]).statistic)
    rng = np.random.default_rng(seed)
    samples = np.empty(bootstrap)
    for index in range(bootstrap):
        draw = np.concatenate([blocks[item] for item in rng.integers(0, len(blocks), len(blocks))])
        samples[index] = spearmanr(draw[:, 0], draw[:, 1]).statistic
    return point, float(np.quantile(samples, .025)), float(np.quantile(samples, .975))


def nav15_correlation(output: Path, bootstrap: int) -> None:
    path = ROOT / FINAL_DISTANCE_PATHS["nav15|QQQ|vanilla"]
    frame = add_metadata(read_csv(path))
    frame["motif_receptor_A"] = frame[[
        "CA_GLN1170_CA-ASN1343_CA", "CA_GLN1170_CA-ASN1449_CA"
    ]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    frame["gate_span_A"] = frame[GATE_COLUMNS].apply(pd.to_numeric, errors="coerce").max(axis=1)
    rows = []
    descriptive = float(spearmanr(frame.motif_receptor_A, frame.gate_span_A).statistic)
    for index, reduction in enumerate(("earliest", "latest", "trajectory_median")):
        if reduction in {"earliest", "latest"}:
            reduced = select_one(frame, reduction)
        else:
            reduced = frame.groupby(["seed", "model_number"], as_index=False)[
                ["motif_receptor_A", "gate_span_A"]
            ].median()
        point, low, high = seed_bootstrap_spearman(
            reduced, "motif_receptor_A", "gate_span_A", bootstrap, BASE_SEED + 400 + index
        )
        rows.append({
            "reduction": reduction, "spearman_rho": point, "seed_bootstrap_CI_low": low,
            "seed_bootstrap_CI_high": high, "snapshot_level_descriptive_rho": descriptive,
            "metric_definition": "per-structure mean of two C-alpha motif-receptor distances versus maximum of six gate spans",
            "inference": "association direction and magnitude are not stable across trajectory reductions",
        })
    save(pd.DataFrame(rows), output / "nav15_qqq_seed_block_correlation_sensitivity.csv")


def cav12_focal(output: Path, bootstrap: int) -> None:
    categorical_rows = []
    g402_parts = {}
    for protocol in ("vanilla", "masked"):
        frame = add_metadata(read_csv(ROOT / FINAL_DISTANCE_PATHS[f"cav12|G402S|{protocol}"]))
        columns = [column for column in frame if column.startswith("shortest_SER402-")]
        numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
        frame["nearest_partner"] = numeric.idxmin(axis=1).str.replace("shortest_SER402-", "", regex=False)
        g402_parts[protocol] = frame
    partners = sorted(set(g402_parts["vanilla"].nearest_partner) | set(g402_parts["masked"].nearest_partner))
    for index, partner in enumerate(partners):
        values = {}
        for protocol, frame in g402_parts.items():
            work = frame.assign(_nearest=frame.nearest_partner.eq(partner).astype(float))
            values[protocol] = trajectory_to_seed(work, "_nearest", within_trajectory="mean")
        categorical_rows.append({
            "partner": partner, "vanilla_probability": values["vanilla"].mean(),
            "masked_probability": values["masked"].mean(),
            **bootstrap_contrast(values["vanilla"], values["masked"], replicates=bootstrap,
                                 seed=BASE_SEED + 500 + index),
        })
    save(pd.DataFrame(categorical_rows).sort_values("vanilla_probability", ascending=False),
         output / "cav12_g402s_nearest_partner_seed_block.csv")

    validity_rows, contact_rows = [], []
    parts = {}
    for protocol in ("vanilla", "masked"):
        frame = add_metadata(read_csv(ROOT / FINAL_DISTANCE_PATHS[f"cav12|G406R|{protocol}"]))
        centered = [column for column in frame if column.startswith("shortest_ARG406-")]
        frame["locally_valid"] = (~frame[centered].apply(pd.to_numeric, errors="coerce").lt(2).any(axis=1)).astype(float)
        parts[protocol] = frame
    validity = {
        protocol: trajectory_to_seed(frame, "locally_valid", within_trajectory="mean")
        for protocol, frame in parts.items()
    }
    survival = {}
    for protocol, frame in parts.items():
        by_trajectory = frame.groupby(["seed", "model_number"])["locally_valid"].max()
        survival[protocol] = by_trajectory.groupby("seed").mean()
    for outcome, values in (("fraction_locally_valid_snapshots", validity),
                            ("fraction_trajectories_with_any_valid_snapshot", survival)):
        validity_rows.append({
            "outcome": outcome, "vanilla_seed_balanced": values["vanilla"].mean(),
            "masked_seed_balanced": values["masked"].mean(),
            **bootstrap_contrast(values["vanilla"], values["masked"], replicates=bootstrap,
                                 seed=BASE_SEED + 600 + len(validity_rows)),
        })
    for partner_index, partner in enumerate(("ASP1528", "ASP1533")):
        values = {}
        for protocol, frame in parts.items():
            clean = frame[frame.locally_valid.eq(1)].copy()
            clean["_contact"] = pd.to_numeric(
                clean[f"shortest_ARG406-{partner}"], errors="coerce"
            ).le(4).astype(float)
            values[protocol] = trajectory_to_seed(clean, "_contact", within_trajectory="mean")
        contact_rows.append({
            "partner": partner, "conditioning": "locally_valid_survivors_only",
            "vanilla_equal_seed_model_trajectory": values["vanilla"].mean(),
            "masked_equal_seed_model_trajectory": values["masked"].mean(),
            **bootstrap_contrast(values["vanilla"], values["masked"], replicates=bootstrap,
                                 seed=BASE_SEED + 620 + partner_index),
        })
    save(pd.DataFrame(validity_rows), output / "cav12_g406r_local_validity_seed_block.csv")
    save(pd.DataFrame(contact_rows), output / "cav12_g406r_conditional_contacts_seed_block.csv")


def kv21_interaction(output: Path, bootstrap: int) -> None:
    frames = {}
    for condition in ("WT", "L403A"):
        for protocol in ("vanilla", "masked"):
            if condition == "L403A":
                source = ROOT / "kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv"
                frame = read_csv(source)
                frame = frame[frame.condition.eq(protocol)].copy()
            else:
                relative = (
                    f"kv21/dataDistances/26-02-11_Kv2.1_wt_{protocol}AF2_distances_"
                    "all_ok_rmsd_3A_structural_interface_qc.csv"
                )
                frame = read_csv(ROOT / relative)
            frame = add_metadata(frame)
            frame[L403_COLUMNS] = frame[L403_COLUMNS].apply(pd.to_numeric, errors="coerce")
            frame["max_E423_N179_A"] = frame[L403_COLUMNS].max(axis=1)
            frames[(condition, protocol)] = trajectory_to_seed(
                frame, "max_E423_N179_A", within_trajectory="median"
            )
    point = (
        frames[("L403A", "masked")].mean() - frames[("L403A", "vanilla")].mean()
        - frames[("WT", "masked")].mean() + frames[("WT", "vanilla")].mean()
    )
    rng = np.random.default_rng(BASE_SEED + 700)
    draws = {}
    for key, values in frames.items():
        array = values.to_numpy(float)
        draws[key] = array[rng.integers(0, len(array), size=(bootstrap, len(array)))].mean(axis=1)
    samples = draws[("L403A", "masked")] - draws[("L403A", "vanilla")] - draws[("WT", "masked")] + draws[("WT", "vanilla")]
    row = {
        "coordinate": "maximum tetramer E423-N179 C-alpha distance",
        "estimand": "(L403A_masked-L403A_vanilla)-(WT_masked-WT_vanilla)",
        "interaction_estimate_A": point,
        "seed_bootstrap_CI_low_A": np.quantile(samples, .025),
        "seed_bootstrap_CI_high_A": np.quantile(samples, .975),
        "within_trajectory_reduction": "median",
        "within_seed_model_weighting": "equal available AF2 parameterizations",
    }
    save(pd.DataFrame([row]), output / "kv21_l403a_masking_interaction_seed_block.csv")


def repeated_first100(seed_values: dict[str, pd.Series], output: Path, bootstrap: int) -> None:
    records, summaries = [], []
    metrics = {
        "continuous_max_distance": (seed_values["vanilla_continuous"], seed_values["masked_continuous"]),
        "rare_shifted_fraction": (seed_values["vanilla"], seed_values["masked"]),
    }
    rng = np.random.default_rng(BASE_SEED + 800)
    draws = 1000
    for metric, (vanilla, masked) in metrics.items():
        common = np.array(sorted(set(vanilla.index) & set(masked.index)))
        if len(common) < 20:
            raise ValueError(f"{metric}: fewer than 20 common seed IDs")
        full_effect = masked.mean() - vanilla.mean()
        metric_rows = []
        for draw_index in range(draws):
            chosen = rng.choice(common, 20, replace=False)
            a, b = vanilla.loc[chosen].to_numpy(float), masked.loc[chosen].to_numpy(float)
            effect = b.mean() - a.mean()
            boot_a = a[rng.integers(0, len(a), size=(200, len(a)))].mean(axis=1)
            boot_b = b[rng.integers(0, len(b), size=(200, len(b)))].mean(axis=1)
            boot_effect = boot_b - boot_a
            row = {
                "metric": metric, "draw": draw_index + 1,
                "nominal_trajectories_per_protocol": 100,
                "common_seeds_sampled": 20, "effect": effect,
                "full_ensemble_effect": full_effect,
                "same_direction_as_full": np.sign(effect) == np.sign(full_effect),
                "relative_error": abs(effect - full_effect) / max(abs(full_effect), 1e-12),
                "subset_CI_covers_full_effect": np.quantile(boot_effect, .025) <= full_effect <= np.quantile(boot_effect, .975),
                "masked_rare_state_detected": bool((b > 0).any()) if metric == "rare_shifted_fraction" else np.nan,
            }
            records.append(row); metric_rows.append(row)
        table = pd.DataFrame(metric_rows)
        summaries.append({
            "metric": metric, "draws": draws,
            "fraction_same_direction": table.same_direction_as_full.mean(),
            "median_relative_error": table.relative_error.median(),
            "fraction_subset_CI_covers_full_effect": table.subset_CI_covers_full_effect.mean(),
            "probability_detect_masked_rare_state": table.masked_rare_state_detected.mean(),
        })
    save(pd.DataFrame(records), output / "first100_repeated_common_seed_draws.csv")
    save(pd.DataFrame(summaries), output / "first100_repeated_common_seed_summary.csv")


def write_run_metadata(output: Path, mode: str, bootstrap: int, permutations: int) -> None:
    nested_oid = "e93fda439af7fb35a7cb8464485e164247b5ff64b784054df476f1349f12cbb4"
    nested = ROOT / ".git/lfs/objects" / nested_oid[:2] / nested_oid[2:4] / nested_oid
    full_panel = output / "full_panel/all_distance_seed_block_panel_run_summary.json"
    regenerated_nav15 = output / "nav15_regional_rmsd/nav15_regional_rmsd_run_summary.json"
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        commit = "unavailable"
    report = {
        "analysis": "seed_block_structural_ensemble_inference",
        "mode": mode, "bootstrap_replicates": bootstrap, "permutations": permutations,
        "random_seed": BASE_SEED, "git_commit": commit,
        "primary_resampling_unit": "seed",
        "within_trajectory_reduction": "median for primary continuous geometry; explicitly named fractions or earliest/latest summaries for focal categorical analyses",
        "within_seed_weighting": "equal available AF2 model parameterizations after the specified trajectory reduction",
        "artifact_availability": {
            "nav15_historical_nested_lfs_object": nested.is_file(),
            "nav15_regional_rmsd_regeneration": regenerated_nav15.is_file(),
            "f412l_objective_representative_pdb": bool(list(ROOT.glob("**/*f412l*.pdb"))),
            "all_distance_seed_block_panel": full_panel.is_file(),
        },
        "design_scope": {
            "kv21": "shared mask supports sequence-by-masking contrasts",
            "nav15_cav12": "condition-specific masks make masked WT-mutant comparisons protocol-specific",
        },
    }
    (output / "seed_block_revision_run_summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/seed_block")
    parser.add_argument("--mode", choices=("exploratory", "publication"), default="publication")
    args = parser.parse_args()
    settings = {
        "exploratory": (500, 999),
        "publication": (2000, 9999),
    }
    bootstrap, permutations = settings[args.mode]
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    seed_values = l403a_analysis(output, bootstrap, permutations)
    master = master_cohort(output)
    retention_analysis(master, output, bootstrap)
    f412l_analysis(output, bootstrap)
    nav15_correlation(output, bootstrap)
    cav12_focal(output, bootstrap)
    kv21_interaction(output, bootstrap)
    repeated_first100(seed_values, output, bootstrap)
    write_run_metadata(output, args.mode, bootstrap, permutations)
    print(f"Seed-block analysis completed: {output}")


if __name__ == "__main__":
    main()
