#!/usr/bin/env python3
"""Run the trajectory-aware manuscript statistics revision.

All outputs are additive and are written only beneath the requested output
directory. Existing notebooks, source CSVs, figures, and manuscript files are
never modified.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import platform
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wasserstein_distance, spearmanr
import seaborn as sns

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from trajectory_statistics import (
    add_trajectory_columns,
    cluster_bootstrap,
    equal_trajectory_estimate,
    parse_model_name,
    select_one_snapshot,
    test_filename_parser,
)


BOOTSTRAP_REPLICATES = 1000
DISTANCE_COLUMNS = {
    "A": "CA_CA_A_GLU425_CA-A_ASN181_CA",
    "B": "CA_CA_B_GLU425_CA-B_ASN181_CA",
    "C": "CA_CA_C_GLU425_CA-C_ASN181_CA",
    "D": "CA_CA_D_GLU425_CA-D_ASN181_CA",
}
L403A_THRESHOLD = 12.84


def check_not_lfs_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        prefix = handle.read(200)
    if prefix.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"Unresolved Git LFS pointer: {path}")


def resolve_lfs(path: Path, repo_root: Path) -> Path:
    with path.open("rb") as handle:
        prefix = handle.read(256).decode("utf-8", errors="ignore")
    if not prefix.startswith("version https://git-lfs.github.com/spec/v1"):
        return path
    oid_line = next((line for line in prefix.splitlines() if line.startswith("oid sha256:")), None)
    if oid_line is None:
        raise RuntimeError(f"Malformed Git LFS pointer: {path}")
    oid = oid_line.split(":", 1)[1]
    target = repo_root / ".git" / "lfs" / "objects" / oid[:2] / oid[2:4] / oid
    if not target.is_file():
        raise FileNotFoundError(f"Local Git LFS object unavailable for {path}")
    return target


def read_csv(path: Path, repo_root: Path, **kwargs) -> pd.DataFrame:
    resolved = resolve_lfs(path, repo_root)
    logging.info("input=%s resolved=%s", path, resolved)
    compression = "gzip" if path.suffix == ".gz" else "infer"
    return pd.read_csv(resolved, compression=compression, **kwargs)


def require_columns(frame: pd.DataFrame, columns: list[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise KeyError(f"{source} missing exact required columns: {missing}")
    logging.info("columns[%s]=%s", source, columns)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    logging.info("saved=%s rows=%d", path, len(frame))


def bootstrap_row(frame, statistic, seed, label, **metadata):
    point, low, high = cluster_bootstrap(
        frame, statistic, replicates=BOOTSTRAP_REPLICATES, seed=seed
    )
    return {
        **metadata,
        "metric": label,
        "estimate": point,
        "cluster_bootstrap_95CI_low": low,
        "cluster_bootstrap_95CI_high": high,
        "independent_trajectories": frame["trajectory_id"].nunique(),
        "retained_snapshots": len(frame),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": seed,
    }


def build_counts(repo_root: Path, output: Path) -> pd.DataFrame:
    manifests = [
        repo_root / "kv21/dataRMSF/qc/kv21_all_ok3_selection_manifest.csv",
        repo_root / "nav15/dataRMSF/qc/nav15_all_ok3_selection_manifest.csv",
        repo_root / "cav12/dataRMSF/qc/cav12_all_ok3_selection_manifest.csv",
    ]
    rows = []
    composition = []
    for path in manifests:
        frame = read_csv(path, repo_root)
        required = ["dataset", "model_number", "seed", "recycle_number", "parse_ok", "all_ok", "all_ok_3"]
        require_columns(frame, required, path)
        for dataset, part in frame.groupby("dataset"):
            trajectory = part["model_number"].astype(str) + "|" + part["seed"].astype(str)
            generated = trajectory.nunique()
            all_ok = part[part["all_ok"].fillna(False)]
            final = part[part["all_ok_3"].fillna(False)]
            final_tid = final["model_number"].astype(str) + "|" + final["seed"].astype(str)
            per_trajectory = final.assign(_tid=final_tid).groupby("_tid").size()
            rows.append({
                "condition": dataset,
                "generated_model_seed_trajectories": generated,
                "trajectories_passing_recycle_convergence": (
                    all_ok["model_number"].astype(str) + "|" + all_ok["seed"].astype(str)
                ).nunique(),
                "trajectories_passing_residue_chain_mapping": np.nan,
                "trajectories_passing_structural_integrity_qc": final_tid.nunique(),
                "trajectories_passing_analysis_specific_qc": final_tid.nunique(),
                "retained_recycle_snapshots": len(final),
                "snapshots_per_trajectory_min": per_trajectory.min() if len(per_trajectory) else np.nan,
                "snapshots_per_trajectory_median": per_trajectory.median() if len(per_trajectory) else np.nan,
                "snapshots_per_trajectory_max": per_trajectory.max() if len(per_trajectory) else np.nan,
                "mapping_count_status": "unresolved in convergence manifest; analysis-specific tables audited separately",
            })
            for model, model_part in final.groupby("model_number"):
                composition.append({
                    "condition": dataset, "dimension": "af2_model", "level": model,
                    "snapshots": len(model_part),
                    "trajectories": (model_part["model_number"].astype(str) + "|" + model_part["seed"].astype(str)).nunique(),
                })
    result = pd.DataFrame(rows).sort_values("condition")
    save_table(result, output / "tables/Table_S2_trajectory_and_snapshot_counts.csv")
    save_table(pd.DataFrame(composition), output / "tables/filtering_composition_audit.csv")
    return result


def l403a_analysis(repo_root: Path, output: Path, seed: int) -> pd.DataFrame:
    source = repo_root / "kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv"
    frame = read_csv(source, repo_root)
    require_columns(frame, ["condition", "pdb_file", *DISTANCE_COLUMNS.values()], source)
    chain_cols = list(DISTANCE_COLUMNS.values())
    for column in chain_cols:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["max_distance_A"] = frame[chain_cols].max(axis=1)
    frame["shifted_subunits"] = frame[chain_cols].ge(L403A_THRESHOLD).sum(axis=1)
    frame["any_shifted"] = frame["shifted_subunits"].ge(1).astype(float)
    frame["four_shifted"] = frame["shifted_subunits"].eq(4).astype(float)
    frame = pd.concat([
        add_trajectory_columns(part, dataset=f"l403a_{condition}")
        for condition, part in frame.groupby("condition")
    ], ignore_index=True)
    rows = []
    for index, (condition, part) in enumerate(frame.groupby("condition")):
        funcs = {
            "q95_max_E423_N179_A": lambda d: float(d["max_distance_A"].quantile(.95)),
            "q99_max_E423_N179_A": lambda d: float(d["max_distance_A"].quantile(.99)),
            "q999_max_E423_N179_A": lambda d: float(d["max_distance_A"].quantile(.999)),
            "maximum_E423_N179_A": lambda d: float(d["max_distance_A"].max()),
            "fraction_any_shifted_ge_12.84A": lambda d: float(d["any_shifted"].mean()),
            "fraction_four_shifted_ge_12.84A": lambda d: float(d["four_shifted"].mean()),
            "fraction_any_ge_14.17A": lambda d: float(d["max_distance_A"].ge(14.17).mean()),
            "fraction_any_ge_16.24A": lambda d: float(d["max_distance_A"].ge(16.24).mean()),
        }
        for metric_index, (label, func) in enumerate(funcs.items()):
            row = bootstrap_row(part, func, seed + index * 100 + metric_index, label, condition=condition)
            if "fraction" in label:
                value_col = "four_shifted" if "four_shifted" in label else "any_shifted"
                if "14.17" in label:
                    part = part.assign(_landmark=part["max_distance_A"].ge(14.17).astype(float)); value_col = "_landmark"
                if "16.24" in label:
                    part = part.assign(_landmark=part["max_distance_A"].ge(16.24).astype(float)); value_col = "_landmark"
                row["equal_trajectory_estimate"] = equal_trajectory_estimate(part, value_col, np.mean)
            else:
                row["equal_trajectory_estimate"] = equal_trajectory_estimate(part, "max_distance_A", np.median, np.median)
            for rule in ("earliest", "latest"):
                selected = select_one_snapshot(part, rule)
                row[f"{rule}_retained_one_per_trajectory"] = func(selected)
            rows.append(row)
        occupancy = part.groupby("trajectory_id")["shifted_subunits"].value_counts(normalize=True).unstack(fill_value=0)
        for number in range(5):
            rows.append({
                "condition": condition,
                "metric": f"snapshot_fraction_{number}_shifted_subunits",
                "estimate": float(part["shifted_subunits"].eq(number).mean()),
                "equal_trajectory_estimate": float(occupancy.get(number, pd.Series(0, index=occupancy.index)).mean()),
                "independent_trajectories": part["trajectory_id"].nunique(),
                "retained_snapshots": len(part),
            })
    summary = pd.DataFrame(rows)
    save_table(summary, output / "tables/L403A_trajectory_aware_summary.csv")

    thresholds = np.round(np.arange(11.5, 15.01, 0.1), 2)
    sensitivity = []
    for condition, part in frame.groupby("condition"):
        for threshold in thresholds:
            work = part.assign(_event=part["max_distance_A"].ge(threshold).astype(float))
            point, low, high = cluster_bootstrap(
                work, lambda d: float(d["_event"].mean()),
                replicates=500, seed=seed + int(threshold * 10),
            )
            sensitivity.append({"condition": condition, "threshold_A": threshold, "fraction": point, "ci_low": low, "ci_high": high})
    sensitivity = pd.DataFrame(sensitivity)
    save_table(sensitivity, output / "tables/L403A_threshold_sensitivity.csv")

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3))
    for condition, color in [("vanilla", "#A8D5B2"), ("masked", "#168A45")]:
        part = frame[frame["condition"].eq(condition)]
        sns.ecdfplot(data=part, x="max_distance_A", ax=axes[0], label=condition.title(), color=color)
        occ = part["shifted_subunits"].value_counts(normalize=True).reindex(range(5), fill_value=0)
        axes[1].plot(range(5), occ, marker="o", label=condition.title(), color=color)
        draw = sensitivity[sensitivity["condition"].eq(condition)]
        axes[2].plot(draw["threshold_A"], draw["fraction"], color=color, label=condition.title())
        axes[2].fill_between(draw["threshold_A"], draw["ci_low"], draw["ci_high"], color=color, alpha=.2)
    axes[0].set(xlabel="Maximum E423-N179 Cα distance (Å)", ylabel="ECDF", title="A  Maximum interface distance")
    axes[1].set(xlabel="Shifted subunits per structure", ylabel="Descriptive snapshot fraction", title="B  Tetramer occupancy", xticks=range(5))
    axes[2].set(xlabel="Shift threshold (Å)", ylabel="Fraction with ≥1 shifted subunit", title="C  Threshold sensitivity")
    axes[2].axvline(L403A_THRESHOLD, color="#777777", ls="--", lw=1)
    for ax in axes: ax.legend(frameon=False)
    fig.suptitle("Kv2.1 L403A trajectory-aware E423-N179 sensitivity", fontweight="bold")
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"figures/Figure_S3_L403A_threshold_and_occupancy.{suffix}", dpi=400, bbox_inches="tight")
    plt.close(fig)
    return summary


def f412l_analysis(repo_root: Path, output: Path, seed: int) -> pd.DataFrame:
    source = repo_root / "kv21/dataRMSD/analysis/comparison_v5/f412l_pocket_D_paper_nexus_shortest_contacts_long_v5.csv"
    frame = read_csv(source, repo_root)
    value = "Shortest heavy-atom distance (Å)"
    require_columns(frame, ["Protocol", "pdb_file", "Contact", value], source)
    frame[value] = pd.to_numeric(frame[value], errors="coerce")
    frame = pd.concat([
        add_trajectory_columns(part, dataset=f"f412l_{protocol.lower()}")
        for protocol, part in frame.groupby("Protocol")
    ], ignore_index=True)
    rows = []
    for group_index, ((protocol, contact), part) in enumerate(frame.groupby(["Protocol", "Contact"])):
        values = part[value].dropna()
        base = {
            "protocol": protocol, "contact": contact,
            "snapshot_median_A": values.median(), "snapshot_IQR_A": values.quantile(.75)-values.quantile(.25),
            "snapshot_q05_A": values.quantile(.05), "snapshot_q95_A": values.quantile(.95),
            "snapshot_fraction_le_4A": values.le(4).mean(), "snapshot_fraction_lt_2A": values.lt(2).mean(),
            "independent_trajectories": part["trajectory_id"].nunique(), "retained_snapshots": len(part),
            "equal_trajectory_median_A": equal_trajectory_estimate(part, value, np.median, np.median),
            "equal_trajectory_fraction_le_4A": equal_trajectory_estimate(part.assign(_x=part[value].le(4)), "_x", np.mean),
            "equal_trajectory_fraction_lt_2A": equal_trajectory_estimate(part.assign(_x=part[value].lt(2)), "_x", np.mean),
        }
        for metric_index, (name, func) in enumerate({
            "median": lambda d: float(d[value].median()),
            "fraction_le_4A": lambda d: float(d[value].le(4).mean()),
            "fraction_lt_2A": lambda d: float(d[value].lt(2).mean()),
        }.items()):
            _, low, high = cluster_bootstrap(part, func, replicates=BOOTSTRAP_REPLICATES, seed=seed+group_index*10+metric_index)
            base[f"{name}_cluster_CI_low"] = low; base[f"{name}_cluster_CI_high"] = high
        rows.append(base)
    result = pd.DataFrame(rows)
    effects = []
    for contact, part in frame.groupby("Contact"):
        van = part[part["Protocol"].str.lower().eq("vanilla")][value].dropna()
        mask = part[part["Protocol"].str.lower().eq("masked")][value].dropna()
        effects.append({"contact": contact, "masked_minus_vanilla_median_A": mask.median()-van.median(), "masked_minus_vanilla_fraction_le_4A": mask.le(4).mean()-van.le(4).mean(), "masked_minus_vanilla_fraction_lt_2A": mask.lt(2).mean()-van.lt(2).mean(), "wasserstein_distance_A": wasserstein_distance(van, mask)})
    result = result.merge(pd.DataFrame(effects), on="contact", how="left")
    save_table(result, output / "tables/F412L_contact_statistics.csv")
    return result


def contact_summary(frame, value_col, dataset, seed):
    frame = add_trajectory_columns(frame, dataset=dataset)
    values = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.assign(_value=values).dropna(subset=["_value"])
    base = {"snapshot_median_A": frame["_value"].median(), "snapshot_IQR_A": frame["_value"].quantile(.75)-frame["_value"].quantile(.25), "snapshot_contact_le_4A": frame["_value"].le(4).mean(), "snapshot_overlap_lt_2A": frame["_value"].lt(2).mean(), "equal_trajectory_contact_le_4A": equal_trajectory_estimate(frame.assign(_x=frame["_value"].le(4)), "_x", np.mean), "independent_trajectories": frame["trajectory_id"].nunique(), "retained_snapshots": len(frame)}
    for index, (name, func) in enumerate({"median":lambda d:float(d["_value"].median()), "contact_le_4A":lambda d:float(d["_value"].le(4).mean()), "overlap_lt_2A":lambda d:float(d["_value"].lt(2).mean())}.items()):
        _, low, high = cluster_bootstrap(frame, func, replicates=BOOTSTRAP_REPLICATES, seed=seed+index)
        base[f"{name}_CI_low"] = low; base[f"{name}_CI_high"] = high
    for rule in ("earliest", "latest"):
        selected = select_one_snapshot(frame, rule)
        base[f"{rule}_contact_le_4A"] = selected["_value"].le(4).mean()
    return base, frame


def cav12_analysis(repo_root: Path, output: Path, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    g402_files = {
        "WT_vanilla": "cav12/dataDistances/26-02-10_Cav12_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
        "WT_masked": "cav12/dataDistances/26-02-10_Cav12_wt_maskedAF2_distances_all_ok_rmsd_3A.csv",
        "G402S_vanilla": "cav12/dataDistances/26-02-10_Cav12_g402s_vanillaAF2_distances_all_ok_rmsd_3A.csv",
        "G402S_masked": "cav12/dataDistances/26-02-10_Cav12_g402s_maskedAF2_distances_all_ok_rmsd_3A.csv",
    }
    contacts = {
        "WT_vanilla": [("G402-M1524", "shortest_GLY402-MET1524")],
        "WT_masked": [("G402-M1524", "shortest_GLY402-MET1524")],
        "G402S_vanilla": [("S402-M1524", "shortest_SER402-MET1524"), ("S402-I1523", "shortest_SER402-ILE1523")],
        "G402S_masked": [("S402-M1524", "shortest_SER402-MET1524"), ("S402-I1523", "shortest_SER402-ILE1523")],
    }
    rows=[]
    for file_index, (dataset, relative) in enumerate(g402_files.items()):
        source=repo_root/relative; frame=read_csv(source,repo_root)
        for contact_index,(contact,column) in enumerate(contacts[dataset]):
            require_columns(frame,["pdb_file",column],source)
            summary,_=contact_summary(frame[["pdb_file",column]].copy(),column,dataset,seed+file_index*20+contact_index*5)
            rows.append({"dataset":dataset,"contact":contact,"source_file":relative,"source_column":column,**summary})
    g402=pd.DataFrame(rows); save_table(g402,output/"tables/Cav12_G402S_contact_statistics.csv")

    g406_files={
        "vanilla":"cav12/dataDistances/26-07-25_Cav1.2_g406r_vanillaAF2_distances_all_ok_rmsd_3A.csv",
        "masked":"cav12/dataDistances/26-07-25_Cav1.2_g406r_maskedAF2_distances_all_ok_rmsd_3A.csv",
    }
    rows=[]; plot_frames=[]
    for file_index,(protocol,relative) in enumerate(g406_files.items()):
        source=repo_root/relative; frame=read_csv(source,repo_root)
        centered=[c for c in frame if c.startswith("shortest_ARG406-")]
        require_columns(frame,["pdb_file","shortest_ARG406-ASP1528","shortest_ARG406-ASP1533","shortest_ARG406-THR1531"],source)
        if not centered: raise KeyError(f"No R406-centered columns in {source}")
        frame=add_trajectory_columns(frame,dataset=f"g406r_{protocol}")
        numeric=frame[centered].apply(pd.to_numeric,errors="coerce")
        frame["severe_overlap"] = numeric.lt(2).any(axis=1)
        frame["ensemble"]="raw"
        filtered=frame[~frame["severe_overlap"]].copy(); filtered["ensemble"]="clash_filtered"
        no_clean = frame.groupby("trajectory_id")["severe_overlap"].all().sum()
        for ensemble_name,part in [("raw",frame),("clash_filtered",filtered)]:
            for partner in ("ASP1528","ASP1533","THR1531"):
                column=f"shortest_ARG406-{partner}"
                work=part.assign(_value=pd.to_numeric(part[column],errors="coerce")).dropna(subset=["_value"])
                summary={"snapshot_median_A":work["_value"].median(),"snapshot_IQR_A":work["_value"].quantile(.75)-work["_value"].quantile(.25),"snapshot_contact_le_4A":work["_value"].le(4).mean(),"equal_trajectory_contact_le_4A":equal_trajectory_estimate(work.assign(_x=work["_value"].le(4)),"_x",np.mean),"independent_trajectories":work["trajectory_id"].nunique(),"retained_snapshots":len(work)}
                _,low,high=cluster_bootstrap(work,lambda d:float(d["_value"].le(4).mean()),replicates=BOOTSTRAP_REPLICATES,seed=seed+file_index*100+(0 if ensemble_name=="raw" else 40)+len(rows))
                rows.append({"protocol":protocol,"ensemble":ensemble_name,"partner":partner,"source_file":relative,"source_column":column,"fraction_snapshots_removed":frame["severe_overlap"].mean() if ensemble_name=="clash_filtered" else 0,"trajectories_without_clash_free_snapshot":int(no_clean),"contact_fraction_CI_low":low,"contact_fraction_CI_high":high,**summary})
            plot_frames.append(part[["trajectory_id","ensemble","shortest_ARG406-ASP1528","shortest_ARG406-ASP1533"]].assign(protocol=protocol))
    g406=pd.DataFrame(rows); save_table(g406,output/"tables/Cav12_G406R_raw_vs_clash_filtered.csv")
    plot=pd.concat(plot_frames,ignore_index=True).melt(id_vars=["trajectory_id","protocol","ensemble"],var_name="contact",value_name="distance_A")
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
    for ax,ensemble in zip(axes,["raw","clash_filtered"]):
        sns.violinplot(data=plot[plot["ensemble"].eq(ensemble)],x="contact",y="distance_A",hue="protocol",split=True,inner="quart",cut=0,ax=ax,palette={"vanilla":"#A8D5B2","masked":"#168A45"})
        ax.axhline(4,color="#777",ls="--",lw=1); ax.set_title(ensemble.replace("_"," ").title()); ax.tick_params(axis="x",rotation=15)
    fig.suptitle("CaV1.2 G406R acidic-partner proximity: clash-filter sensitivity",fontweight="bold"); fig.tight_layout(); fig.savefig(output/"figures/Cav12_G406R_clash_filter_sensitivity.pdf",bbox_inches="tight"); fig.savefig(output/"figures/Cav12_G406R_clash_filter_sensitivity.png",dpi=400,bbox_inches="tight"); plt.close(fig)
    return g402,g406


def nav15_analysis(repo_root: Path, output: Path, seed: int) -> pd.DataFrame:
    files={
        "WT | vanilla":"nav15/dataDistances/26-07-27_Nav15_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
        "WT | masked":"nav15/dataDistances/26-07-25_Nav15_wt_masked_AF2_distances_extra_ifm_all_ok_rmsd_3A.csv",
        "WT | masked v2":"nav15/dataDistances/26-07-27_Nav15_wt_maskedv2_AF2_distances_all_ok_rmsd_3A.csv",
        "WT | masked v2 no-IFM":"nav15/dataDistances/26-07-27_Nav15_wt_maskedv2_noIFM_AF2_distances_all_ok_rmsd_3A.csv",
        "QQQ | vanilla":"nav15/dataDistances/26-07-27_Nav15_qqq_vanilla_AF2_distances_all_ok_rmsd_3A.csv",
        "QQQ | masked":"nav15/dataDistances/26-07-27_Nav15_qqq_masked_AF2_distances_all_ok_rmsd_3A.csv",
        "QQQ | masked v2":"nav15/dataDistances/26-07-27_Nav15_qqq_maskedv2_AF2_distances_all_ok_rmsd_3A.csv",
    }
    gate_cols=["CA_MET415_CA-ALA742_CA","CA_MET415_CA-ILE1154_CA","CA_MET415_CA-ILE1455_CA","CA_ALA742_CA-ILE1154_CA","CA_ALA742_CA-ILE1455_CA","CA_ILE1154_CA-ILE1455_CA"]
    rows=[]; combined=[]; map_rows=[]
    for file_index,(dataset,relative) in enumerate(files.items()):
        source=repo_root/relative; frame=read_csv(source,repo_root); condition=dataset.split(" | ")[0]
        central="PHE1170" if condition=="WT" else "GLN1170"; motif=["ILE1169","PHE1170","MET1171"] if condition=="WT" else ["GLN1169","GLN1170","GLN1171"]
        ca1=f"CA_{central}_CA-ASN1343_CA"; ca2=f"CA_{central}_CA-ASN1449_CA"; term1=f"shortest_{central}-ASN1343"; term2=f"shortest_{central}-ASN1449"; motif_cols=[f"shortest_{res}-ASN1343" for res in motif]
        required=["pdb_file",ca1,ca2,term1,term2]; require_columns(frame,required,source)
        gate_available=all(column in frame for column in gate_cols)
        motif_available=all(column in frame for column in motif_cols)
        if not gate_available:
            logging.warning("analysis_unavailable dataset=%s missing_gate_columns=%s",dataset,[c for c in gate_cols if c not in frame])
        if not motif_available:
            logging.warning("analysis_unavailable dataset=%s missing_complete_motif_columns=%s",dataset,[c for c in motif_cols if c not in frame])
        frame=add_trajectory_columns(frame,dataset=dataset.replace(" | ","_")); frame["dataset"]=dataset
        frame["motif_separation_A"]=frame[[ca1,ca2]].apply(pd.to_numeric,errors="coerce").mean(axis=1); frame["gate_max_A"]=(frame[gate_cols].apply(pd.to_numeric,errors="coerce").max(axis=1) if gate_available else np.nan); frame["terminal_N1659_A"]=pd.to_numeric(frame[term1],errors="coerce"); frame["terminal_N1765_A"]=pd.to_numeric(frame[term2],errors="coerce"); frame["whole_motif_N1659_A"]=(frame[motif_cols].apply(pd.to_numeric,errors="coerce").min(axis=1) if motif_available else np.nan)
        available_metrics=["motif_separation_A","terminal_N1659_A","terminal_N1765_A"]
        if gate_available: available_metrics.append("gate_max_A")
        if motif_available: available_metrics.append("whole_motif_N1659_A")
        for metric_index,column in enumerate(available_metrics):
            work=frame.dropna(subset=[column]); row=bootstrap_row(work,lambda d,c=column:float(d[c].median()),seed+file_index*20+metric_index,f"median_{column}",dataset=dataset); row["equal_trajectory_estimate"]=equal_trajectory_estimate(work,column,np.median,np.median); row["earliest_one_per_trajectory"]=select_one_snapshot(work,"earliest")[column].median(); row["latest_one_per_trajectory"]=select_one_snapshot(work,"latest")[column].median(); rows.append(row)
        if gate_available:
            pair=frame.dropna(subset=["motif_separation_A","gate_max_A"]); rho=float(spearmanr(pair["motif_separation_A"],pair["gate_max_A"]).statistic); early=select_one_snapshot(pair,"earliest"); late=select_one_snapshot(pair,"latest"); traj=pair.groupby("trajectory_id")[["motif_separation_A","gate_max_A"]].median()
            _,low,high=cluster_bootstrap(pair,lambda d:float(spearmanr(d["motif_separation_A"],d["gate_max_A"]).statistic),replicates=BOOTSTRAP_REPLICATES,seed=seed+file_index*20+10)
            rows.append({"dataset":dataset,"metric":"spearman_motif_separation_vs_gate","estimate":rho,"cluster_bootstrap_95CI_low":low,"cluster_bootstrap_95CI_high":high,"one_earliest_snapshot_per_trajectory":spearmanr(early["motif_separation_A"],early["gate_max_A"]).statistic,"one_latest_snapshot_per_trajectory":spearmanr(late["motif_separation_A"],late["gate_max_A"]).statistic,"trajectory_median_spearman":spearmanr(traj["motif_separation_A"],traj["gate_max_A"]).statistic,"independent_trajectories":pair["trajectory_id"].nunique(),"retained_snapshots":len(pair)})
        combined.append(frame[["dataset","trajectory_id","recycle_number","terminal_N1659_A","terminal_N1765_A","whole_motif_N1659_A","motif_separation_A","gate_max_A"]])
        map_rows.append({"figure_panel":"Figure 2E baseline" if dataset=="WT | masked v2" else "Figure 5 / Figure S4 control","sequence":condition,"protocol":dataset.split(" | ",1)[1],"mask_name":{"WT | vanilla":"unmodified MSA","WT | masked":"nav15_standard","WT | masked v2":"mask v2","WT | masked v2 no-IFM":"mask v2 noIFM","QQQ | vanilla":"unmodified MSA","QQQ | masked":"nav15_standard_plus_IFM","QQQ | masked v2":"mask v2"}[dataset],"source_csv":relative,"source_notebook":"nav15/Nav15_IFM_latching_analysis.ipynb","trajectory_count":frame["trajectory_id"].nunique(),"snapshot_count":len(frame),"notes":f"Exact label preserved; designs are not interchangeable. gate_available={gate_available}; complete_motif_available={motif_available}."})
    result=pd.DataFrame(rows); save_table(result,output/"tables/Nav15_trajectory_aware_statistics.csv"); save_table(pd.DataFrame(map_rows),output/"tables/Nav15_figure_condition_map.csv")
    plot=pd.concat(combined,ignore_index=True); selected=plot[plot["dataset"].isin(["WT | vanilla","WT | masked","WT | masked v2","WT | masked v2 no-IFM","QQQ | vanilla","QQQ | masked","QQQ | masked v2"])]
    fig,axes=plt.subplots(1,2,figsize=(12,4.6)); sns.violinplot(data=selected,x="dataset",y="whole_motif_N1659_A",inner="quart",cut=0,ax=axes[0],color="#9D7BB0"); axes[0].tick_params(axis="x",rotation=35); axes[0].set(xlabel="",ylabel="Minimum motif-N1659 terminal distance (Å)",title="A  Mask-design control")
    sns.scatterplot(data=selected.sample(min(5000,len(selected)),random_state=seed),x="terminal_N1659_A",y="motif_separation_A",hue="dataset",s=10,alpha=.25,ax=axes[1]); axes[1].set(title="B  Cα versus terminal-atom consistency",xlabel="Central motif-N1659 terminal distance (Å)",ylabel="Mean Cα motif-receptor separation (Å)"); axes[1].legend(fontsize=6)
    fig.suptitle("NaV1.5 IFM/QQQ mask controls",fontweight="bold"); fig.tight_layout(); fig.savefig(output/"figures/Figure_S4_Nav15_mask_controls.pdf",bbox_inches="tight"); fig.savefig(output/"figures/Figure_S4_Nav15_mask_controls.png",dpi=400,bbox_inches="tight"); plt.close(fig)
    return result


def manuscript_audit(output: Path, counts, l403a, f412l, nav15, g402, g406) -> pd.DataFrame:
    claims=[
        ("Results - Kv2.1 L403A","Fig. 3B","Vanilla q95/q99/max max E423-N179 = 11.36/12.42/14.80 Å","11.36; 12.42; 14.80","l403a_vanilla","kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv","maximum_distance_A","kv21/check_L403A_E423_N179_extremes.py"),
        ("Results - Kv2.1 L403A","Fig. 3B","Masked q95/q99/max max E423-N179 = 13.65/14.50/15.55 Å","13.65; 14.50; 15.55","l403a_masked","kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv","maximum_distance_A","kv21/check_L403A_E423_N179_extremes.py"),
        ("Results - Kv2.1 L403A","Fig. 3B","9.94% masked vs 0.02% vanilla have any subunit >=12.8 Å","9.94%; 0.02%","L403A masked/vanilla","kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv","four mapped GLU425-ASN181 Cα columns","kv21/check_L403A_E423_N179_extremes.py"),
        ("Results - NaV1.5","Fig. 5E","QQQ vanilla Spearman rho = 0.13","0.13","QQQ vanilla","nav15/dataDistances/26-07-27_Nav15_qqq_vanilla_AF2_distances_all_ok_rmsd_3A.csv","mean CA_GLN1170-ASN1343/1449 versus max six gate spans","nav15/Nav15_IFM_latching_analysis.ipynb"),
        ("Results - CaV1.2 G402S","Fig. 6D","S402-I1523 occupancy 19.2% vanilla and 12.1% masked","19.2%; 12.1%","G402S","cav12/dataDistances/26-02-10_Cav12_g402s_*_all_ok_rmsd_3A.csv","shortest_SER402-ILE1523 <=4 Å","cav12/Cav12_G402S_mutationSite_analysis.ipynb"),
        ("Results - CaV1.2 G406R","Fig. 6C","R406-D1533 occupancy 29.0% vanilla and 8.8% masked; R406-D1528 20.6% and 25.5%","29.0%; 8.8%; 20.6%; 25.5%","G406R raw","cav12/dataDistances/26-07-25_Cav1.2_g406r_*_all_ok_rmsd_3A.csv","shortest_ARG406-ASP1533/ASP1528 <=4 Å","cav12/Cav12_G406R_mutationSite_analysis.ipynb"),
        ("Results - CaV1.2 G406R","Fig. 6C","R406-T1531 overlap <2 Å = 25.2% vanilla and 78.1% masked","25.2%; 78.1%","G406R raw","cav12/dataDistances/26-07-25_Cav1.2_g406r_*_all_ok_rmsd_3A.csv","shortest_ARG406-THR1531 <2 Å","cav12/Cav12_G406R_mutationSite_analysis.ipynb"),
    ]
    rows=[]
    for section,figure,claim,reported,condition,source,columns,script in claims:
        rows.append({"section":section,"figure":figure,"manuscript_claim":claim,"reported_value":reported,"condition":condition,"source_file":source,"source_columns":columns,"analysis_notebook_or_script":script,"current_denominator":"retained structural snapshots","independent_trajectory_count":"see linked trajectory-aware table","recomputed_value":"see linked trajectory-aware table","match_status":"traced; trajectory-aware revision generated","notes":"Snapshot-level value is descriptive, not an independent-replicate estimate."})
    rows.extend([
        {"section":"Results - Kv2.1 RMSF","figure":"Fig. S2","manuscript_claim":"Masked-minus-vanilla RMSF medians and SD ratios","reported_value":"3.50/2.08/1.92; +1.47/+1.31/+1.31 Å","condition":"Kv2.1 WT/L403A/F412L","source_file":"kv21/dataRMSF analysis products","source_columns":"multiple profile columns","analysis_notebook_or_script":"kv21/Kv21_ensemble_RMSF.ipynb","current_denominator":"retained structures contributing to ensemble profile","independent_trajectory_count":"unresolved in this revision","recomputed_value":"not recomputed","match_status":"unresolved exact claim-to-column provenance","notes":"Do not silently substitute a different RMSF definition."},
        {"section":"Results - Kv2.1 L403A","figure":"Fig. 3B","manuscript_claim":"12.84 Å shifted-interface threshold","reported_value":"12.84 Å","condition":"L403A","source_file":"kv21/check_L403A_E423_N179_extremes.py","source_columns":"hard-coded THRESHOLDS dictionary","analysis_notebook_or_script":"kv21/check_L403A_E423_N179_extremes.py","current_denominator":"not applicable","independent_trajectory_count":"not applicable","recomputed_value":"threshold sensitivity 11.5-15.0 Å generated","match_status":"threshold value traced; derivation unresolved","notes":"No independent derivation was documented in the traced source."},
        {"section":"Results - NaV1.5 regional RMSD","figure":"Fig. S5","manuscript_claim":"Regional RMSD medians to 8VYJ","reported_value":"pore 2.190/2.169; IFM 2.87/19.20; pocket 3.05/7.46; linker 4.56/12.17 Å","condition":"WT vanilla/masked","source_file":"nav15/dataRMSD products","source_columns":"regional RMSD columns","analysis_notebook_or_script":"Nav15 RMSD analysis","current_denominator":"retained snapshot-reference rows","independent_trajectory_count":"unresolved pending exact mask/source row audit","recomputed_value":"not recomputed","match_status":"unresolved exact Figure S5 source mapping","notes":"Figure S5 not fabricated from summary tables."},
    ])
    audit=pd.DataFrame(rows); save_table(audit,output/"manuscript_number_audit.csv"); return audit


def write_documents(output: Path, audit: pd.DataFrame, seed: int) -> None:
    methods=f"""# Statistical methods\n\nThe primary independent sampling unit was the AlphaFold model-parameterization/random-seed trajectory. Recycle snapshots within a trajectory and multiple subunits within one structure were treated as correlated observations. Snapshot-level distributions are reported descriptively as summaries among retained structural snapshots and are not interpreted as thermodynamic populations.\n\nTrajectory-aware uncertainty was estimated by a cluster bootstrap that sampled whole model-seed trajectories with replacement within each condition and retained every qualifying snapshot from each sampled trajectory. Unless otherwise stated, {BOOTSTRAP_REPLICATES:,} bootstrap replicates and random seed {seed} were used, with percentile 95% confidence intervals. Equal-trajectory estimates first summarized each trajectory and then weighted trajectories equally. Sensitivity analyses retained either the earliest or latest numbered qualifying recycle from each trajectory. The L403A shifted-interface analysis additionally varied the threshold from 11.5 to 15.0 Å in 0.1 Å increments.\n\nSevere G406R mutation-site overlap was defined before analysis as any R406-centered shortest heavy-atom distance <2 Å. Raw and clash-filtered ensembles were analyzed in parallel. Contact frequencies describe protocol sampling frequency, not equilibrium occupancy.\n"""
    (output/"manuscript_updates").mkdir(parents=True,exist_ok=True)
    (output/"manuscript_updates/statistical_methods.md").write_text(methods)
    results="# Statistics results: minimal replacements\n\n"
    for _,row in audit.iterrows():
        if row["match_status"].startswith("unresolved"): continue
        results+=f"## {row['section']} - {row['figure']}\n\n**CURRENT CLAIM**\n\n{row['manuscript_claim']}\n\n**RECOMPUTED RESULT**\n\nSee the exact linked output table; the original snapshot statistic remains descriptive.\n\n**TRAJECTORY-AWARE RESULT**\n\nReport the cluster-bootstrap interval and independent trajectory count from the linked table.\n\n**RECOMMENDED MINIMAL REPLACEMENT**\n\nRetain the descriptive estimate but identify it as the fraction or summary among retained structural snapshots and append the trajectory-bootstrap 95% confidence interval and number of independent model-seed trajectories.\n\n"
    (output/"manuscript_updates/statistics_results.md").write_text(results)
    (output/"manuscript_updates/figure_caption_updates.md").write_text("# Figure caption updates\n\nFor Figures 3-6 and new Figures S3-S5, state that violin/distribution summaries include retained structural snapshots, while confidence intervals were obtained by resampling independent model-seed trajectories. Contact and shifted-state frequencies are protocol-sampling descriptors rather than thermodynamic occupancies.\n")


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--repo-root",type=Path,default=Path.cwd()); parser.add_argument("--output-dir",type=Path,default=Path("analysis/statistics_revision")); parser.add_argument("--seed",type=int,default=20260803); args=parser.parse_args()
    repo=args.repo_root.resolve(); output=(repo/args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir; [ (output/name).mkdir(parents=True,exist_ok=True) for name in ("scripts","tables","figures","manuscript_updates","logs") ]
    logging.basicConfig(filename=output/"logs/run_all_statistics.log",level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s",filemode="w"); logging.getLogger().addHandler(logging.StreamHandler())
    test_filename_parser(); counts=build_counts(repo,output); l403a=l403a_analysis(repo,output,args.seed); f412l=f412l_analysis(repo,output,args.seed+1000); nav15=nav15_analysis(repo,output,args.seed+2000); g402,g406=cav12_analysis(repo,output,args.seed+3000); audit=manuscript_audit(output,counts,l403a,f412l,nav15,g402,g406); write_documents(output,audit,args.seed)
    versions={"python":sys.version,"platform":platform.platform(),"numpy":np.__version__,"pandas":pd.__version__,"scipy":scipy.__version__,"matplotlib":plt.matplotlib.__version__,"seaborn":sns.__version__,"bootstrap_seed":args.seed,"bootstrap_replicates":BOOTSTRAP_REPLICATES}; (output/"environment_versions.txt").write_text("\n".join(f"{k}: {v}" for k,v in versions.items())+"\n")
    logging.info("completed output=%s",output)


if __name__=="__main__": main()
