#!/usr/bin/env python3
"""Create representative-model, sampling-depth, and final audit documentation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
from trajectory_statistics import add_trajectory_columns, parse_model_name
from run_all_statistics import resolve_lfs


def save(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def representative_audit(repo: Path, output: Path) -> None:
    contacts_path = repo / "kv21/dataRMSD/analysis/comparison_v5/f412l_pocket_D_paper_nexus_shortest_contacts_long_v5.csv"
    rmsd_path = repo / "kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD_v5.csv"
    contacts = pd.read_csv(contacts_path)
    value = "Shortest heavy-atom distance (Å)"
    wide = contacts.pivot_table(index=["Protocol", "pdb_file", "model_path"], columns="Contact", values=value, aggfunc="first").reset_index()
    rename = {}
    for column in wide:
        text = str(column)
        if "L412–L316" in text: rename[column] = "L412_L316"
        elif "L412–L329" in text: rename[column] = "L412_L329"
        elif "L412–L403" in text: rename[column] = "L412_L403"
    wide = wide.rename(columns=rename)
    wide = wide[wide["Protocol"].str.lower().eq("masked")].copy()
    parsed = wide["pdb_file"].map(parse_model_name)
    wide["trajectory_id"] = parsed.map(lambda x: f"f412l_masked|{x.trajectory_key}")
    wide["rank"] = parsed.map(lambda x: x.rank); wide["model"] = parsed.map(lambda x: x.model); wide["seed"] = parsed.map(lambda x: x.seed); wide["recycle"] = parsed.map(lambda x: x.recycle)
    use = ["pdb_file", "reference_id", "pocket_D_all_atom_rmsd_A", "analysis_status"]
    rmsd = pd.read_csv(rmsd_path, usecols=use)
    rmsd = rmsd[rmsd["analysis_status"].eq("ok") & rmsd["reference_id"].isin(["8SD3", "8SDA"])]
    pivot = rmsd.pivot_table(index="pdb_file", columns="reference_id", values="pocket_D_all_atom_rmsd_A", aggfunc="first").rename(columns={"8SD3":"pocket_RMSD_to_8SD3", "8SDA":"pocket_RMSD_to_8SDA"}).reset_index()
    wide = wide.merge(pivot, on="pdb_file", how="left")
    wide["minimum_mutation_site_distance"] = wide[["L412_L316", "L412_L329", "L412_L403"]].min(axis=1)
    selected = "kv21_f412l_masked_unrelaxed_rank_037_alphafold2_multimer_v3_model_4_seed_103.r1.pdb"
    if not wide["pdb_file"].eq(selected).any():
        parsed_selected = parse_model_name(selected)
        selected_rmsd = pivot[pivot["pdb_file"].eq(selected)]
        wide = pd.concat([wide, pd.DataFrame([{
            "Protocol": "Masked", "pdb_file": selected,
            "model_path": "/quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/f412l/masked/models/" + selected,
            "trajectory_id": f"f412l_masked|{parsed_selected.trajectory_key}",
            "rank": parsed_selected.rank, "model": parsed_selected.model,
            "seed": parsed_selected.seed, "recycle": parsed_selected.recycle,
            "L412_L316": np.nan, "L412_L329": np.nan, "L412_L403": np.nan,
            "minimum_mutation_site_distance": np.nan,
            "pocket_RMSD_to_8SD3": selected_rmsd["pocket_RMSD_to_8SD3"].iloc[0] if len(selected_rmsd) else np.nan,
            "pocket_RMSD_to_8SDA": selected_rmsd["pocket_RMSD_to_8SDA"].iloc[0] if len(selected_rmsd) else np.nan,
        }])], ignore_index=True)
    wide["selection_status"] = np.where(
        wide["pdb_file"].eq(selected),
        "manuscript representative; absent from corrected v5 final-QC contact set",
        "not selected",
    )
    wide["exclusion_reason"] = np.where(
        wide["pdb_file"].eq(selected),
        "all_ok=True and earliest_converged_selected=True, but all_ok_3=False; corrected v5 contact table begins at r2/r10 for this trajectory.",
        "No repository-stored, pre-registered equivalent-selection rule was found; retained for transparent candidate audit.",
    )
    columns = ["trajectory_id","pdb_file","model_path","rank","model","seed","recycle","L412_L316","L412_L329","L412_L403","minimum_mutation_site_distance","pocket_RMSD_to_8SD3","pocket_RMSD_to_8SDA","selection_status","exclusion_reason"]
    save(wide[columns].sort_values(["selection_status","rank","recycle"]), output / "tables/F412L_representative_candidate_audit.csv")


def trajectory_metric(frame: pd.DataFrame, value: str, statistic: str) -> float:
    if statistic == "median": return float(frame[value].median())
    if statistic == "q99": return float(frame[value].quantile(.99))
    if statistic == "mean": return float(frame[value].mean())
    raise ValueError(statistic)


def repeated_subsampling(frame, value, statistic, ns, repeats, seed, label, condition):
    groups = {
        key: pd.to_numeric(part[value], errors="coerce").dropna().to_numpy()
        for key,part in frame.groupby("trajectory_id")
    }
    keys = np.asarray(list(groups),dtype=object); rng=np.random.default_rng(seed); rows=[]
    for n in ns:
        actual=min(n,len(keys)); estimates=[]
        iterations=1 if actual==len(keys) else repeats
        for _ in range(iterations):
            chosen=keys if actual==len(keys) else rng.choice(keys,size=actual,replace=False)
            if statistic == "mean":
                total = sum(float(groups[key].sum()) for key in chosen)
                count = sum(int(groups[key].size) for key in chosen)
                estimates.append(total / count)
            else:
                draw = np.concatenate([groups[key] for key in chosen])
                estimates.append(float(np.quantile(draw, .5 if statistic == "median" else .99)))
        rows.append({"metric":label,"condition":condition,"requested_N":n,"actual_N":actual,"available_trajectories":len(keys),"repeats":iterations,"median_estimate":np.median(estimates),"interval_2.5":np.quantile(estimates,.025),"interval_97.5":np.quantile(estimates,.975),"random_seed":seed})
    return rows


def sampling_depth(repo: Path, output: Path, seed: int) -> None:
    rows=[]; ns=[25,50,100,200,300,500]
    l403=pd.read_csv(repo/"kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv")
    for condition,part in l403.groupby("condition"):
        part=add_trajectory_columns(part,dataset=f"l403a_{condition}"); part["max_A"]=part[["chain_A","chain_B","chain_C","chain_D"]].max(axis=1); part["any_shifted"]=part["max_A"].ge(12.84).astype(float)
        rows+=repeated_subsampling(part,"max_A","q99",ns,500,seed,f"L403A q99 maximum E423-N179",condition)
        rows+=repeated_subsampling(part,"any_shifted","mean",ns,500,seed+1,f"L403A any-shifted fraction",condition)
    f412=pd.read_csv(repo/"kv21/dataRMSD/analysis/comparison_v5/f412l_pocket_D_paper_nexus_shortest_contacts_long_v5.csv"); val="Shortest heavy-atom distance (Å)"
    for (protocol,contact),part in f412.groupby(["Protocol","Contact"]):
        part=add_trajectory_columns(part,dataset=f"f412l_{protocol.lower()}"); part["contact_le4"]=part[val].le(4).astype(float)
        rows+=repeated_subsampling(part,"contact_le4","mean",ns,500,seed+2,f"F412L {contact} contact <=4 A",protocol)
    nav_path=repo/"nav15/dataDistances/26-07-27_Nav15_qqq_vanilla_AF2_distances_all_ok_rmsd_3A.csv"
    nav=pd.read_csv(nav_path); nav=add_trajectory_columns(nav,dataset="qqq_vanilla")
    nav["motif_receptor_median_A"]=nav[["CA_GLN1170_CA-ASN1343_CA","CA_GLN1170_CA-ASN1449_CA"]].apply(pd.to_numeric,errors="coerce").mean(axis=1)
    rows+=repeated_subsampling(nav,"motif_receptor_median_A","median",ns,500,seed+3,"NaV1.5 QQQ motif-receptor median","vanilla")
    for offset,protocol in enumerate(("vanilla","masked")):
        g402_path=repo/f"cav12/dataDistances/26-02-10_Cav12_g402s_{protocol}AF2_distances_all_ok_rmsd_3A.csv"
        g402=pd.read_csv(g402_path); g402=add_trajectory_columns(g402,dataset=f"g402s_{protocol}"); g402["contact_le4"]=pd.to_numeric(g402["shortest_SER402-ILE1523"],errors="coerce").le(4).astype(float)
        rows+=repeated_subsampling(g402,"contact_le4","mean",ns,500,seed+4+offset,"CaV1.2 G402S S402-I1523 contact <=4 A",protocol)
    for offset,protocol in enumerate(("vanilla","masked")):
        g406_path=repo/f"cav12/dataDistances/26-07-25_Cav1.2_g406r_{protocol}AF2_distances_all_ok_rmsd_3A.csv"
        g406=pd.read_csv(resolve_lfs(g406_path,repo)); g406=add_trajectory_columns(g406,dataset=f"g406r_{protocol}")
        centered=[c for c in g406 if c.startswith("shortest_ARG406-")]; numeric=g406[centered].apply(pd.to_numeric,errors="coerce"); clean=g406[~numeric.lt(2).any(axis=1)].copy()
        for partner_index,partner in enumerate(("ASP1528","ASP1533")):
            clean["contact_le4"]=pd.to_numeric(clean[f"shortest_ARG406-{partner}"],errors="coerce").le(4).astype(float)
            rows+=repeated_subsampling(clean,"contact_le4","mean",ns,500,seed+6+offset*2+partner_index,f"CaV1.2 G406R clash-filtered R406-{partner} contact <=4 A",protocol)
    table=pd.DataFrame(rows); save(table,output/"tables/sampling_depth_stability.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    panels=[("L403A q99 maximum E423-N179",axes[0]),("L403A any-shifted fraction",axes[1])]
    for metric,ax in panels:
        for condition,part in table[table["metric"].eq(metric)].groupby("condition"):
            ax.plot(part["actual_N"],part["median_estimate"],marker="o",label=condition)
            ax.fill_between(part["actual_N"],part["interval_2.5"],part["interval_97.5"],alpha=.18)
        ax.set(xlabel="Independent trajectories sampled",title=metric); ax.legend(frameon=False); ax.grid(alpha=.25)
    fig.suptitle("Repeated random trajectory-subsampling stability",fontweight="bold"); fig.tight_layout(); fig.savefig(output/"figures/Figure_sampling_depth_stability.pdf",bbox_inches="tight"); fig.savefig(output/"figures/Figure_sampling_depth_stability.png",dpi=400,bbox_inches="tight"); plt.close(fig)


def write_readme(output: Path) -> None:
    text="""# Trajectory-aware statistics revision

## Scope completed

- Read and visually inspected the complete 26-page manuscript draft.
- Tested a canonical parser for rank, AlphaFold model, seed, and recycle naming patterns.
- Treated model-seed trajectories as independent units.
- Generated trajectory-bootstrap confidence intervals, equal-trajectory estimates, and earliest/latest retained-snapshot sensitivities for the primary L403A, F412L, NaV1.5, G402S, and G406R coordinates available with exact source mappings.
- Generated raw and severe-overlap-filtered G406R analyses.
- Added repeated trajectory subsampling for L403A and all three F412L contacts.

## Important discrepancies and unresolved items

1. The 12.84 A L403A threshold is hard-coded in `kv21/check_L403A_E423_N179_extremes.py`; no independent derivation was found. Figure S3 therefore includes threshold sensitivity from 11.5 to 15.0 A.
2. The original WT masked NaV1.5 IFM CSV lacks the six gate-span and complete-motif terminal-distance columns. Those metrics were marked unavailable rather than borrowed from a different mask design.
3. The exact trajectory-resolved input behind the proposed NaV1.5 regional-RMSD Figure S5 could not be resolved locally: the nominal compressed OK3 table contains a nested Git LFS pointer. Figure S5 was not fabricated from snapshot-level summary tables.
4. Exact mapping-QC counts are not present in the convergence manifests; Table S2 marks this stage unresolved instead of equating convergence with mapping.
5. The manuscript F412L representative (rank 037, model 4, seed 103, r1) has `all_ok=True` and is the `earliest_converged_selected` row, but `all_ok_3=False`; it is absent from the corrected v5 final-QC contact table, which retains later snapshots from that trajectory. The draft must not state that r1 passed the corrected final-QC set. No pre-registered repository rule defining all equivalent candidates was found.
6. The traced L403A analysis table contains 3,910 masked and 4,403 vanilla snapshots, versus 4,073 and 4,521 in Table S1. On the traced table, the >=12.84 A any-shifted fractions are 12.69% masked and 0.50% vanilla, not the draft's 9.94% and 0.02%. This denominator/source mismatch must be resolved before submission.
7. Repeated trajectory-subsampling was completed for L403A, all three F412L contacts, NaV1.5 QQQ motif-receptor separation, G402S S402-I1523 proximity, and clash-filtered G406R acidic-partner proximity.

## Interpretation

Snapshot-level percentages remain useful descriptions of geometries sampled by each prediction protocol. They are not independent-replicate estimates or thermodynamic populations. Main-text claims should pair those descriptive values with model-seed trajectory counts and trajectory-bootstrap confidence intervals.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/vgci-matplotlib conda run -n bioadri python analysis/statistics_revision/scripts/run_all_statistics.py --repo-root <REPO_ROOT> --output-dir analysis/statistics_revision --seed 20260803
MPLCONFIGDIR=/tmp/vgci-matplotlib conda run -n bioadri python analysis/statistics_revision/scripts/complete_remaining_outputs.py --repo-root <REPO_ROOT> --output-dir analysis/statistics_revision --seed 20260803
```

The full execution record is in `logs/run_all_statistics.log`. No original notebook, source CSV, figure, or manuscript file was overwritten.
"""
    (output/"README.md").write_text(text)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--repo-root",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--seed",type=int,default=20260803); args=parser.parse_args(); repo=args.repo_root.resolve(); output=(repo/args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    representative_audit(repo,output); sampling_depth(repo,output,args.seed); write_readme(output)


if __name__=="__main__": main()
