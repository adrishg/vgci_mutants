#!/usr/bin/env python3
"""Run the versioned paired-seed statistical revision.

All outputs are additive and written below ``paired_seed_v2``.  The script
uses nominal seed labels as design keys but reports that actual RNG values are
not recoverable from the available run metadata.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_seed_block_revision import (
    FINAL_DISTANCE_PATHS,
    GATE_COLUMNS,
    L403_COLUMNS,
    add_metadata,
    derive_l403a_threshold,
    master_cohort,
    read_csv,
)
from shared.paired_seed_statistics import (
    analyze_paired_seed_distance,
    joint_seed_bootstrap_contrast,
    paired_common_seed_bootstrap,
    paired_factorial_interaction,
    paired_seed_categorical_vector_contrast,
    paired_seed_ratio_of_means,
    paired_seed_retention_contrast,
    select_paired_estimand,
)


BASE_SEED = 20260824
MASK_IDS = {
    ("kv21", "vanilla"): "unmasked",
    ("kv21", "masked"): "kv21_common",
    ("nav15", "vanilla"): "unmasked",
    ("nav15", "masked"): "nav15_standard_plus_IFM",
    ("nav15", "masked_v2"): "nav15_v2",
    ("nav15", "masked_v2_noIFM"): "nav15_v2_noIFM",
    ("cav12", "vanilla"): "unmasked",
    ("cav12", "masked"): "condition_specific_primary_mask",
}


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def trajectory_summary(frame: pd.DataFrame, value: str, reduction: str) -> pd.DataFrame:
    work = add_metadata(frame) if not {"seed", "model_number"}.issubset(frame) else frame.copy()
    work[value] = pd.to_numeric(work[value], errors="coerce")
    work = work.dropna(subset=[value])
    grouped = work.groupby(["seed", "model_number"])[value]
    if reduction == "median":
        result = grouped.median()
    elif reduction == "mean":
        result = grouped.mean()
    elif reduction in {"earliest", "latest"}:
        ordered = work.sort_values(["seed", "model_number", "recycle_number", "pdb_file"])
        selected = ordered.groupby(["seed", "model_number"], as_index=False).head(1) \
            if reduction == "earliest" else ordered.groupby(["seed", "model_number"], as_index=False).tail(1)
        return selected[["seed", "model_number", value]].reset_index(drop=True)
    else:
        raise ValueError(reduction)
    return result.rename(value).reset_index()


def paired_scalar_rows(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value: str,
    *,
    contrast: str,
    outcome: str,
    bootstrap: int,
    seed: int,
) -> list[dict[str, object]]:
    rows = []
    primary = joint_seed_bootstrap_contrast(
        frame_a, frame_b, value, n_bootstrap=bootstrap, random_seed=seed
    )
    rows.append({"contrast": contrast, "outcome": outcome, "estimand_id": "primary_joint_nominal_seed", **primary})
    common = paired_common_seed_bootstrap(
        frame_a, frame_b, value, n_bootstrap=bootstrap, random_seed=seed + 1
    )
    rows.append({"contrast": contrast, "outcome": outcome, "estimand_id": "common_seed", **common})
    model_a, model_b, audit = select_paired_estimand(
        frame_a, frame_b, value, estimand="common_model_seed"
    )
    model = paired_common_seed_bootstrap(
        model_a, model_b, value, n_bootstrap=bootstrap, random_seed=seed + 2
    )
    rows.append({
        "contrast": contrast, "outcome": outcome, "estimand_id": "common_model_seed",
        **audit, **model,
    })
    return rows


def nominalized_trajectory_table(
    frame: pd.DataFrame,
    value: str,
    *,
    nominal_seeds: list[int] | np.ndarray | pd.Series | None = None,
) -> pd.DataFrame:
    if nominal_seeds is None:
        nominal_seeds = sorted(pd.to_numeric(frame["seed"], errors="raise").astype(int).unique())
    summarized = trajectory_summary(frame, value, "mean").set_index(["seed", "model_number"])
    index = pd.MultiIndex.from_product(
        [list(nominal_seeds), [1, 2, 3, 4, 5]], names=["seed", "model_number"]
    )
    return summarized.reindex(index, fill_value=0).reset_index()


def g402_common_ca_columns(frame: pd.DataFrame, background: str) -> list[str]:
    """Return the atomically matched position-402 C-alpha partner panel.

    This deliberately rejects the legacy shortest-heavy columns so glycine can
    never be described as having a non-hydrogen side-chain distance.
    """
    residue = {"WT": "GLY402", "G402S": "SER402"}.get(background)
    if residue is None:
        raise ValueError("background must be WT or G402S")
    partners = (
        "PHE1519", "VAL1520", "ALA1521", "VAL1522", "ILE1523", "MET1524",
        "ASP1525", "ASN1526", "PHE1527", "ASP1528", "TYR1529", "LEU1530",
        "THR1531", "ARG1532", "ASP1533", "TRP1534", "SER1535",
    )
    columns = [f"CA_{residue}_CA-{partner}_CA" for partner in partners]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise KeyError(f"Missing atomically matched position-402 C-alpha columns: {missing}")
    if any(column.startswith("shortest_") for column in columns):
        raise AssertionError("G402 primary comparison cannot use undocumented side-chain fallback")
    return columns


def make_inventory(master: pd.DataFrame, output: Path) -> None:
    inventory = master.groupby(["channel", "sequence", "protocol", "ensemble_id"], as_index=False).agg(
        nominal_seed_labels=("seed", "nunique"),
        retained_snapshots=("distance_final_cohort", "sum"),
        mapping_qc_snapshots=("mapping_QC_pass", "sum"),
        convergence_qc_snapshots=("all_ok", "sum"),
        structural_qc_snapshots=("all_ok_3", "sum"),
    )
    trajectory_counts = master[["ensemble_id", "seed", "model_number"]].drop_duplicates().groupby(
        "ensemble_id", as_index=False
    ).size().rename(columns={"size": "nominal_model_seed_trajectories"})
    inventory = inventory.merge(trajectory_counts, on="ensemble_id", validate="one_to_one")
    inventory["mask_id"] = [MASK_IDS.get((c, p), "unresolved") for c, p in zip(inventory.channel, inventory.protocol)]
    seed_ranges = master.groupby("ensemble_id").seed.agg(seed_id_min="min", seed_id_max="max").reset_index()
    inventory = inventory.merge(seed_ranges, on="ensemble_id", validate="one_to_one")
    inventory["actual_random_seed_status"] = "unavailable in run metadata; numeric labels are nominal design keys"
    inventory["source_csv"] = inventory.ensemble_id.map(FINAL_DISTANCE_PATHS).fillna("")
    inventory.to_csv(output / "DATASET_INVENTORY.tsv", sep="\t", index=False)

    trajectory = master.groupby(
        ["channel", "sequence", "protocol", "ensemble_id", "seed", "model_number"], as_index=False
    ).agg(
        trajectory_basename=("pdb_basename", "first"),
        mapping_qc=("mapping_QC_pass", "max"),
        convergence_qc=("all_ok", "max"),
        final_qc=("all_ok_3", "max"),
        analysis_final=("distance_final_cohort", "max"),
    )
    trajectory["mask_id"] = [MASK_IDS.get((c, p), "unresolved") for c, p in zip(trajectory.channel, trajectory.protocol)]
    trajectory["actual_random_seed"] = ""
    trajectory["actual_random_seed_status"] = "not recorded/recovered"
    trajectory["nominal_in_design"] = True
    trajectory = trajectory.rename(columns={
        "sequence": "sequence_background", "protocol": "msa_protocol",
        "seed": "seed_id", "model_number": "af2_model",
    })
    columns = [
        "channel", "sequence_background", "msa_protocol", "mask_id", "seed_id",
        "actual_random_seed", "actual_random_seed_status", "af2_model",
        "trajectory_basename", "nominal_in_design", "mapping_qc", "convergence_qc",
        "final_qc", "analysis_final",
    ]
    trajectory[columns].to_csv(output / "SEED_REGISTRY.tsv", sep="\t", index=False)

    trajectory_flow = master.groupby(
        ["channel", "sequence", "protocol", "ensemble_id", "seed", "model_number"], as_index=False
    ).agg(
        mapping_qc=("mapping_QC_pass", "max"), convergence_qc=("all_ok", "max"),
        structural_qc=("all_ok_3", "max"), interface_qc=("distance_source_membership", "max"),
        analysis_final=("distance_final_cohort", "max"),
    )
    flow = trajectory_flow.groupby(
        ["channel", "sequence", "protocol", "ensemble_id"], as_index=False
    ).agg(
        nominal_seeds=("seed", "nunique"),
        nominal_model_seed_trajectories=("model_number", "size"),
        mapping_qc_trajectories=("mapping_qc", "sum"),
        converged_trajectories=("convergence_qc", "sum"),
        final_structural_qc_trajectories=("structural_qc", "sum"),
        channel_interface_qc_trajectories=("interface_qc", "sum"),
        analysis_final_trajectories=("analysis_final", "sum"),
    )
    snapshots = master.groupby("ensemble_id", as_index=False).agg(
        retained_snapshots=("distance_final_cohort", "sum")
    )
    exclusion = master[~master.exclusion_reason.eq("included_in_final_distance_cohort")].groupby(
        "ensemble_id"
    ).exclusion_reason.agg(lambda values: values.value_counts().index[0] if len(values) else "none")
    flow = flow.merge(snapshots, on="ensemble_id", validate="one_to_one")
    flow["primary_exclusion_reason"] = flow.ensemble_id.map(exclusion).fillna("none")
    save(flow, output / "master_cohort_flow_summary.csv")


def retention(master: pd.DataFrame, output: Path, bootstrap: int) -> None:
    trajectory = master.groupby(
        ["channel", "sequence", "protocol", "seed", "model_number"], as_index=False
    ).agg(
        mapping_qc=("mapping_QC_pass", "max"),
        convergence_qc=("all_ok", "max"),
        structural_qc=("all_ok_3", "max"),
        analysis_final=("distance_final_cohort", "max"),
    )
    rows = []
    for (channel, sequence), backgrounds in trajectory.groupby(["channel", "sequence"]):
        if "vanilla" not in set(backgrounds.protocol):
            continue
        a = backgrounds[backgrounds.protocol.eq("vanilla")]
        for protocol, b in backgrounds[~backgrounds.protocol.eq("vanilla")].groupby("protocol"):
            for stage_index, stage in enumerate(("mapping_qc", "convergence_qc", "structural_qc", "analysis_final")):
                result = paired_seed_retention_contrast(
                    a, b, pass_col=stage,
                    nominal_seeds_a=sorted(a.seed.unique()),
                    nominal_seeds_b=sorted(b.seed.unique()),
                    n_bootstrap=bootstrap,
                    random_seed=BASE_SEED + 1000 + len(rows) + stage_index,
                )
                rows.append({
                    "channel": channel, "sequence_background": sequence,
                    "contrast": f"{protocol}-vanilla", "stage": stage,
                    "mask_id_B": MASK_IDS.get((channel, protocol), "unresolved"), **result,
                })
    save(pd.DataFrame(rows), output / "paired_retention_contrasts.csv")


def l403a(output: Path, bootstrap: int, permutations: int) -> None:
    threshold, wt_vector, mutant_vector = derive_l403a_threshold(output)
    frame = add_metadata(read_csv(ROOT / "kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv"))
    frame[L403_COLUMNS] = frame[L403_COLUMNS].apply(pd.to_numeric, errors="coerce")
    frame["max_E423_N179_A"] = frame[L403_COLUMNS].max(axis=1)
    ordered = np.sort(frame[L403_COLUMNS].to_numpy(float), axis=1)
    frame["RMSE_to_8SD3_A"] = np.sqrt(np.mean((ordered - wt_vector) ** 2, axis=1))
    frame["RMSE_to_8SDA_A"] = np.sqrt(np.mean((ordered - mutant_vector) ** 2, axis=1))
    frame["closer_to_8SDA"] = (frame.RMSE_to_8SDA_A < frame.RMSE_to_8SD3_A).astype(float)
    frame["shifted_subunits"] = frame[L403_COLUMNS].ge(threshold).sum(axis=1)
    frame["any_shifted"] = frame.shifted_subunits.ge(1).astype(float)
    parts = {name: part.copy() for name, part in frame.groupby("condition")}

    rows = []
    definitions = [
        ("maximum_E423_N179_CA_distance_A", "max_E423_N179_A", "median"),
        ("ordered_vector_RMSE_to_8SD3_A", "RMSE_to_8SD3_A", "median"),
        ("ordered_vector_RMSE_to_8SDA_A", "RMSE_to_8SDA_A", "median"),
        ("fraction_closer_to_8SDA_ordered_vector", "closer_to_8SDA", "mean"),
        ("fraction_any_subunit_at_experiment_anchored_cutoff", "any_shifted", "mean"),
    ]
    for index, (outcome, column, reduction) in enumerate(definitions):
        a, b = (trajectory_summary(parts[p], column, reduction) for p in ("vanilla", "masked"))
        rows.extend(paired_scalar_rows(
            a, b, column, contrast="L403A masked - vanilla", outcome=outcome,
            bootstrap=bootstrap, seed=BASE_SEED + 2000 + index * 10,
        ))
        if column == "any_shifted":
            ratio = paired_seed_ratio_of_means(
                a, b, column, n_bootstrap=bootstrap, random_seed=BASE_SEED + 2060
            )
            rows.append({
                "contrast": "L403A masked / vanilla", "outcome": outcome,
                "estimand_id": "ratio_secondary", **ratio,
            })
    for reduction in ("earliest", "latest"):
        a, b = (trajectory_summary(parts[p], "any_shifted", reduction) for p in ("vanilla", "masked"))
        rows.extend(paired_scalar_rows(
            a, b, "any_shifted", contrast="L403A masked - vanilla",
            outcome=f"any_shifted_{reduction}_retained_recycle",
            bootstrap=bootstrap, seed=BASE_SEED + 2070 + (reduction == "latest"),
        ))
    save(pd.DataFrame(rows), output / "l403a_focal_paired_seed.csv")

    w1_rows = []
    a = trajectory_summary(parts["vanilla"], "max_E423_N179_A", "median")
    b = trajectory_summary(parts["masked"], "max_E423_N179_A", "median")
    for index, estimand in enumerate(("primary_joint_nominal_seed", "common_seed", "common_model_seed")):
        w1_rows.append(analyze_paired_seed_distance(
            a, b, "max_E423_N179_A", estimand=estimand,
            n_bootstrap=bootstrap, n_permutations=permutations,
            random_seed=BASE_SEED + 2100 + index * 10,
        ))
    save(pd.DataFrame(w1_rows), output / "l403a_w1_paired_seed.csv")

    sensitivity_rows = []
    for cutoff in np.arange(11.5, 14.5001, .25):
        work = frame.assign(_target=frame[L403_COLUMNS].ge(cutoff).any(axis=1).astype(float))
        a, b = (trajectory_summary(work[work.condition.eq(p)], "_target", "mean") for p in ("vanilla", "masked"))
        result = joint_seed_bootstrap_contrast(
            a, b, "_target", n_bootstrap=bootstrap,
            random_seed=BASE_SEED + 2200 + int(round(cutoff * 100)),
        )
        sensitivity_rows.append({
            "cutoff_A": cutoff,
            "cutoff_role": "prediction-independent experiment-anchored cutoff" if abs(cutoff - threshold) < .126 else "sensitivity grid",
            **result,
        })
    save(pd.DataFrame(sensitivity_rows), output / "l403a_threshold_sensitivity.csv")

    category_summary, category_rows = paired_seed_categorical_vector_contrast(
        parts["vanilla"], parts["masked"], category_col="shifted_subunits",
        categories=[0, 1, 2, 3, 4], n_bootstrap=bootstrap, random_seed=BASE_SEED + 2300,
    )
    for key, value in category_summary.items():
        category_rows[key] = value
    category_rows["chain_mapping_note"] = "sorted/count summary; predicted chain labels are not mapped to experimental chain identities"
    save(category_rows, output / "l403a_zero_to_four_subunits.csv")

    yields = []
    for protocol in ("vanilla", "masked"):
        nominal = nominalized_trajectory_table(parts[protocol], "any_shifted")
        nominal["protocol"] = protocol
        yields.append(nominal)
    a, b = yields
    yield_result = paired_scalar_rows(
        a, b, "any_shifted", contrast="L403A masked - vanilla",
        outcome="QC_adjusted_fraction_retained_snapshots_with_any_shifted_subunit_per_nominal_trajectory",
        bootstrap=bootstrap, seed=BASE_SEED + 2400,
    )
    save(pd.DataFrame(yield_result), output / "usable_target_geometry_yields_l403a.csv")

    four = {}
    for background in ("WT", "L403A"):
        for protocol in ("vanilla", "masked"):
            if background == "L403A":
                source = parts[protocol]
            else:
                source = add_metadata(read_csv(ROOT / FINAL_DISTANCE_PATHS[f"kv21|WT|{protocol}"]))
                source[L403_COLUMNS] = source[L403_COLUMNS].apply(pd.to_numeric, errors="coerce")
                source["max_E423_N179_A"] = source[L403_COLUMNS].max(axis=1)
            four[f"{'wt' if background == 'WT' else 'mutant'}_{protocol}"] = trajectory_summary(
                source, "max_E423_N179_A", "median"
            )
    interaction = paired_factorial_interaction(
        four, "max_E423_N179_A", statistic="mean", n_bootstrap=bootstrap,
        random_seed=BASE_SEED + 2500,
    )
    interaction.update({
        "mutant": "L403A", "outcome": "maximum E423-N179 C-alpha distance",
        "estimand": "mean complete-seed [(L403A masked-L403A vanilla)-(WT masked-WT vanilla)]",
    })
    save(pd.DataFrame([interaction]), output / "kv21_interactions_paired_seed.csv")


def f412l(output: Path, bootstrap: int) -> None:
    path = ROOT / "kv21/dataRMSD/analysis/comparison_v5/f412l_pocket_D_paper_nexus_shortest_contacts_long_v5.csv"
    long = read_csv(path)
    value = "Shortest heavy-atom distance (Å)"
    wide = long.pivot_table(index=["Protocol", "pdb_file"], columns="Contact", values=value, aggfunc="first").reset_index()
    rename = {}
    for column in wide:
        if "L412–L316" in str(column): rename[column] = "L412_L316_A"
        elif "L412–L329" in str(column): rename[column] = "L412_L329_A"
        elif "L412–L403" in str(column): rename[column] = "L412_L403_A"
    wide = add_metadata(wide.rename(columns=rename))
    parts = {name.lower(): part.copy() for name, part in wide.groupby("Protocol")}
    rows = []
    for metric_index, column in enumerate(rename.values()):
        a, b = (trajectory_summary(parts[p], column, "median") for p in ("vanilla", "masked"))
        rows.extend(paired_scalar_rows(
            a, b, column, contrast="F412L masked - vanilla", outcome=f"continuous_{column}",
            bootstrap=bootstrap, seed=BASE_SEED + 3000 + metric_index * 100,
        ))
        for threshold in (3.5, 4.0, 4.5):
            threshold_parts = {}
            for protocol in ("vanilla", "masked"):
                work = parts[protocol].assign(_prox=pd.to_numeric(parts[protocol][column], errors="coerce").le(threshold).astype(float))
                threshold_parts[protocol] = trajectory_summary(work, "_prox", "mean")
            result = paired_scalar_rows(
                threshold_parts["vanilla"], threshold_parts["masked"], "_prox",
                contrast="F412L masked - vanilla",
                outcome=f"fraction_{column}_within_{threshold:.1f}A",
                bootstrap=bootstrap, seed=BASE_SEED + 3100 + metric_index * 20 + int(threshold * 2),
            )
            for row in result:
                row["threshold_A"] = threshold
            rows.extend(result)
        for overlap in (1.8, 2.0, 2.2):
            overlap_parts = {}
            for protocol in ("vanilla", "masked"):
                work = parts[protocol].assign(_overlap=pd.to_numeric(parts[protocol][column], errors="coerce").lt(overlap).astype(float))
                overlap_parts[protocol] = trajectory_summary(work, "_overlap", "mean")
            rows.extend(paired_scalar_rows(
                overlap_parts["vanilla"], overlap_parts["masked"], "_overlap",
                contrast="F412L masked - vanilla", outcome=f"severe_overlap_{column}_below_{overlap:.1f}A",
                bootstrap=bootstrap, seed=BASE_SEED + 3200 + metric_index * 20 + int(overlap * 10),
            ))
    save(pd.DataFrame(rows), output / "f412l_contacts_paired_seed.csv")

    yields = []
    for column in rename.values():
        condition = {}
        for protocol in ("vanilla", "masked"):
            work = parts[protocol].assign(_target=pd.to_numeric(parts[protocol][column], errors="coerce").le(4).astype(float))
            condition[protocol] = nominalized_trajectory_table(work, "_target")
        yields.extend(paired_scalar_rows(
            condition["vanilla"], condition["masked"], "_target",
            contrast="F412L masked - vanilla", outcome=f"QC_adjusted_within_4A_yield_{column}",
            bootstrap=bootstrap, seed=BASE_SEED + 3300 + len(yields),
        ))
    save(pd.DataFrame(yields), output / "usable_target_geometry_yields_f412l.csv")


def nav15(output: Path, bootstrap: int, permutations: int) -> None:
    tables = {}
    for ensemble, relative in FINAL_DISTANCE_PATHS.items():
        if not ensemble.startswith("nav15|"):
            continue
        frame = add_metadata(read_csv(ROOT / relative))
        residue = "PHE" if "|WT|" in ensemble else "GLN"
        frame["motif_receptor_A"] = frame[[
            f"CA_{residue}1170_CA-ASN1343_CA", f"CA_{residue}1170_CA-ASN1449_CA"
        ]].apply(pd.to_numeric, errors="coerce").mean(axis=1)
        if set(GATE_COLUMNS).issubset(frame.columns):
            frame["gate_span_A"] = frame[GATE_COLUMNS].apply(pd.to_numeric, errors="coerce").max(axis=1)
        tables[ensemble] = frame
    specs = [
        ("nav15|WT|vanilla", "nav15|QQQ|vanilla", "vanilla QQQ - vanilla WT", "unmasked"),
        ("nav15|WT|vanilla", "nav15|WT|masked", "WT nav15_standard - vanilla", "nav15_standard"),
        ("nav15|WT|vanilla", "nav15|WT|masked_v2", "WT nav15_v2 - vanilla", "nav15_v2"),
        ("nav15|WT|vanilla", "nav15|WT|masked_v2_noIFM", "WT nav15_v2_noIFM - vanilla", "nav15_v2_noIFM"),
        ("nav15|QQQ|vanilla", "nav15|QQQ|masked", "QQQ nav15_standard_plus_IFM - vanilla", "nav15_standard_plus_IFM"),
        ("nav15|QQQ|vanilla", "nav15|QQQ|masked_v2", "QQQ nav15_v2 - vanilla", "nav15_v2"),
    ]
    rows = []
    for index, (key_a, key_b, contrast, mask_id) in enumerate(specs):
        for outcome in ("motif_receptor_A", "gate_span_A"):
            if outcome not in tables[key_a] or outcome not in tables[key_b]:
                rows.append({
                    "contrast": contrast, "outcome": outcome, "estimand_id": "not_estimable",
                    "mask_id": mask_id, "status": "required gate coordinates absent from one registered source table",
                })
                continue
            a, b = (trajectory_summary(tables[key], outcome, "median") for key in (key_a, key_b))
            scalar = paired_scalar_rows(
                a, b, outcome, contrast=contrast, outcome=outcome,
                bootstrap=bootstrap, seed=BASE_SEED + 4000 + index * 100,
            )
            for row in scalar:
                row["mask_id"] = mask_id
            rows.extend(scalar)
            if outcome == "motif_receptor_A":
                w1 = analyze_paired_seed_distance(
                    a, b, outcome, estimand="primary_joint_nominal_seed",
                    n_bootstrap=bootstrap, n_permutations=permutations,
                    random_seed=BASE_SEED + 4050 + index * 100,
                )
                rows.append({"contrast": contrast, "outcome": "motif_receptor_W1", "estimand_id": "primary_joint_nominal_seed", "mask_id": mask_id, **w1})
    save(pd.DataFrame(rows), output / "nav15_focal_paired_seed.csv")


def g402s(output: Path, bootstrap: int, permutations: int) -> None:
    frames = {}
    for background in ("WT", "G402S"):
        for protocol in ("vanilla", "masked"):
            frame = add_metadata(read_csv(ROOT / FINAL_DISTANCE_PATHS[f"cav12|{background}|{protocol}"]))
            residue = "GLY402" if background == "WT" else "SER402"
            ca_columns = g402_common_ca_columns(frame, background)
            numeric = frame[ca_columns].apply(pd.to_numeric, errors="coerce")
            normalized = [column.replace(residue, "POSITION402") for column in ca_columns]
            numeric.columns = normalized
            for column in normalized:
                frame[column] = numeric[column]
            frame["nearest_partner_CA"] = numeric.idxmin(axis=1).str.extract(r"-(.+)_CA$")[0]
            sorted_values = np.sort(numeric.to_numpy(float), axis=1)
            frame["nearest_CA_distance_A"] = sorted_values[:, 0]
            frame["nearest_minus_second_margin_A"] = sorted_values[:, 1] - sorted_values[:, 0]
            frames[(background, protocol)] = frame

    rows = []
    for protocol in ("vanilla", "masked"):
        for outcome in ("nearest_CA_distance_A", "nearest_minus_second_margin_A"):
            a, b = (trajectory_summary(frames[(background, protocol)], outcome, "median") for background in ("WT", "G402S"))
            rows.extend(paired_scalar_rows(
                a, b, outcome, contrast=f"G402S - WT under {protocol}", outcome=outcome,
                bootstrap=bootstrap, seed=BASE_SEED + 5000 + len(rows),
            ))
            if outcome == "nearest_CA_distance_A" and protocol == "vanilla":
                rows.append({
                    "contrast": "G402S - WT under vanilla", "outcome": "nearest_CA_distance_W1",
                    "estimand_id": "primary_joint_nominal_seed", **analyze_paired_seed_distance(
                        a, b, outcome, estimand="primary_joint_nominal_seed",
                        n_bootstrap=bootstrap, n_permutations=permutations,
                        random_seed=BASE_SEED + 5100,
                    ),
                })
    save(pd.DataFrame(rows), output / "g402s_focal_paired_seed.csv")

    summary, categories = paired_seed_categorical_vector_contrast(
        frames[("G402S", "vanilla")], frames[("G402S", "masked")],
        category_col="nearest_partner_CA",
        categories=sorted(set(frames[("G402S", "vanilla")].nearest_partner_CA.dropna()) |
                          set(frames[("G402S", "masked")].nearest_partner_CA.dropna())),
        n_bootstrap=bootstrap, random_seed=BASE_SEED + 5200,
    )
    for key, value in summary.items(): categories[key] = value
    categories["metric_definition"] = "atomically matched position-402 C-alpha to partner C-alpha distances"
    categories["legacy_glycine_audit"] = "shortest_GLY402-* equals glycine backbone/all-heavy-atom minimum, not a side-chain distance; replaced for WT-mutant primary comparison"
    save(categories, output / "g402s_partner_distribution.csv")


def g406r(output: Path, bootstrap: int) -> None:
    parts = {
        protocol: add_metadata(read_csv(ROOT / FINAL_DISTANCE_PATHS[f"cav12|G406R|{protocol}"]))
        for protocol in ("vanilla", "masked")
    }
    rows, sensitivity, yields = [], [], []
    centered = [column for column in parts["vanilla"] if column.startswith("shortest_ARG406-")]
    for overlap_cutoff in (1.8, 2.0, 2.2):
        derived = {}
        for protocol, frame in parts.items():
            work = frame.copy()
            distances = work[centered].apply(pd.to_numeric, errors="coerce")
            work["overlap_pass"] = (~distances.lt(overlap_cutoff).any(axis=1)).astype(float)
            for partner in ("ASP1528", "ASP1533"):
                proximity = pd.to_numeric(work[f"shortest_ARG406-{partner}"], errors="coerce").le(4)
                work[f"conditional_{partner}"] = np.where(work.overlap_pass.eq(1), proximity.astype(float), np.nan)
                work[f"unconditional_{partner}"] = (work.overlap_pass.eq(1) & proximity).astype(float)
            derived[protocol] = work
        for outcome in ("overlap_pass", "conditional_ASP1528", "conditional_ASP1533", "unconditional_ASP1528", "unconditional_ASP1533"):
            a, b = (trajectory_summary(derived[p], outcome, "mean") for p in ("vanilla", "masked"))
            result = paired_scalar_rows(
                a, b, outcome, contrast="G406R masked - vanilla",
                outcome=outcome, bootstrap=bootstrap,
                seed=BASE_SEED + 6000 + int(overlap_cutoff * 100) + len(rows),
            )
            for row in result:
                row["overlap_threshold_A"] = overlap_cutoff
                row["terminology"] = "R406-centered local-overlap pass status"
            (rows if overlap_cutoff == 2.0 else sensitivity).extend(result)
        # Any overlap-pass snapshot per trajectory, kept distinct from snapshot fraction.
        for protocol in ("vanilla", "masked"):
            trajectory_any = derived[protocol].groupby(["seed", "model_number"], as_index=False).overlap_pass.max()
            derived[protocol + "_any"] = trajectory_any
        any_rows = paired_scalar_rows(
            derived["vanilla_any"], derived["masked_any"], "overlap_pass",
            contrast="G406R masked - vanilla", outcome="fraction_nominal_trajectories_with_any_overlap_pass_snapshot",
            bootstrap=bootstrap, seed=BASE_SEED + 6100 + int(overlap_cutoff * 100),
        )
        for row in any_rows: row["overlap_threshold_A"] = overlap_cutoff
        (rows if overlap_cutoff == 2.0 else sensitivity).extend(any_rows)
        if overlap_cutoff == 2.0:
            for partner in ("ASP1528", "ASP1533"):
                nominal = {}
                for protocol in ("vanilla", "masked"):
                    nominal[protocol] = nominalized_trajectory_table(derived[protocol], f"unconditional_{partner}")
                yields.extend(paired_scalar_rows(
                    nominal["vanilla"], nominal["masked"], f"unconditional_{partner}",
                    contrast="G406R masked - vanilla",
                    outcome=f"QC_adjusted_overlap_pass_and_within_4A_{partner}_yield",
                    bootstrap=bootstrap, seed=BASE_SEED + 6200 + len(yields),
                ))
    save(pd.DataFrame(rows), output / "g406r_overlap_and_contacts.csv")
    save(pd.DataFrame(sensitivity), output / "g406r_overlap_threshold_sensitivity.csv")
    save(pd.DataFrame(yields), output / "usable_target_geometry_yields_g406r.csv")


def run_summary(output: Path, mode: str, bootstrap: int, permutations: int) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    report = {
        "analysis": "paired_seed_v2", "mode": mode, "date": "2026-08-24",
        "git_starting_commit": commit, "python": platform.python_version(),
        "base_random_seed": BASE_SEED, "bootstrap_replicates": bootstrap,
        "permutation_replicates": permutations,
        "primary_estimand": "marginal QC-qualified contrast with joint nominal-seed resampling",
        "pairing_provenance": "actual RNG seed values unavailable; pairing keys are nominal seed labels",
        "source_hashes": "audit/SOURCE_INPUT_HASHES.tsv",
        "source_mutation_policy": "read-only inputs; all results written under paired_seed_v2",
    }
    (output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    parser.add_argument("--mode", choices=("exploratory", "publication"), default="publication")
    args = parser.parse_args()
    bootstrap, permutations = ((250, 499) if args.mode == "exploratory" else (2000, 9999))
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    master = master_cohort(output)
    make_inventory(master, output)
    retention(master, output, bootstrap)
    l403a(output, bootstrap, permutations)
    f412l(output, bootstrap)
    nav15(output, bootstrap, permutations)
    g402s(output, bootstrap, permutations)
    g406r(output, bootstrap)
    run_summary(output, args.mode, bootstrap, permutations)
    print(f"paired-seed revision completed: {output}")


if __name__ == "__main__":
    main()
