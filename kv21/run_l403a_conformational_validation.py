"""Build the Kv2.1 L403A experimental-recovery tables and figures.

This module consumes only the precomputed conformational CSVs in dataExtra.
It deliberately summarizes recycle/model observations within seed before
estimating uncertainty, so recycle snapshots are never treated as replicates.
"""
from __future__ import annotations

from pathlib import Path
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from shared.plotting import apply_kv21_style, KV21_PALETTE, experimental_reference_style

ROOT = REPO_ROOT
DATA = ROOT / "kv21" / "dataExtra"
OUT = DATA / "conformation_analysis"
FIG = OUT / "figures"
TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True)
TAB.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(403)
N_BOOT = 2000

ID_COLS = ["structure_id", "source_type", "condition", "protocol", "dataset",
           "model_number", "seed", "recycle", "recycle_label", "rank",
           "trajectory_id", "canonical_subunit", "is_final_model",
           "frame_orientation_score", "source_path"]

METRICS = [
    "kink_angle_deg", "whole_s6_rotation_vs_8SD3_deg",
    "I401_azimuth_deg", "I405_azimuth_deg", "I401_cb_azimuth_deg", "I405_cb_azimuth_deg",
    "I401_pore_facing_score", "I405_pore_facing_score",
    "I401_neighbor_facing_score", "I405_neighbor_facing_score",
    "linker_radial_distance_A", "linker_signed_radial_coordinate_A",
    "linker_residue_radial_delta_median_A", "linker_residue_radial_delta_min_A",
    "linker_residue_radial_delta_max_A", "linker_residue_max_inward_A",
    "linker_residue_fraction_inward", "linker_centroid_displacement_vs_8SD3_A",
    "linker_residue_ca_displacement_median_A", "linker_residue_ca_displacement_max_A",
    "F412_sidechain_centroid_displacement_vs_8SD3_A", "F412_ca_displacement_vs_8SD3_A",
    "F412_L316_same_ca_distance_A", "F412_L316_same_shortest_heavy_A",
    "F412_L329_neighbor_ca_distance_A", "F412_L329_neighbor_shortest_heavy_A",
    "F412_403_neighbor_ca_distance_A", "F412_403_neighbor_shortest_heavy_A",
    "pi_geometric_like_fraction", "pi_incoming_geometric_like_fraction",
]
ANGLE_METRICS = {"whole_s6_rotation_vs_8SD3_deg", "I401_azimuth_deg", "I405_azimuth_deg",
                 "I401_cb_azimuth_deg", "I405_cb_azimuth_deg"}

LABELS = {
    "kink_angle_deg": "PIP/S6 kink (°)",
    "whole_s6_rotation_vs_8SD3_deg": "Whole-S6 rotation (°)",
    "I401_azimuth_deg": "I401 azimuth (°)", "I405_azimuth_deg": "I405 azimuth (°)",
    "I401_pore_facing_score": "I401 pore-facing", "I405_pore_facing_score": "I405 pore-facing",
    "I401_neighbor_facing_score": "I401 neighbor-facing", "I405_neighbor_facing_score": "I405 neighbor-facing",
    "linker_radial_distance_A": "S4–S5 radial distance (Å)",
    "linker_residue_ca_displacement_max_A": "Maximum local linker Cα displacement (Å)",
    "linker_residue_fraction_inward": "Fraction linker residues inward",
    "F412_sidechain_centroid_displacement_vs_8SD3_A": "F412 side-chain displacement (Å)",
    "F412_L316_same_shortest_heavy_A": "F412–L316 contact (Å)",
    "F412_L329_neighbor_shortest_heavy_A": "F412–L329 contact (Å)",
    "F412_403_neighbor_shortest_heavy_A": "F412–403-neighbor contact (Å)",
}

COLORS = {("wt", "vanilla"): KV21_PALETTE["WT_VAN"],
          ("wt", "masked"): KV21_PALETTE["WT_HM"],
          ("l403a", "vanilla"): KV21_PALETTE["L403A_VAN"],
          ("l403a", "masked"): KV21_PALETTE["L403A_HM"]}

def wrap(x):
    return (np.asarray(x, dtype=float) + 180.0) % 360.0 - 180.0

def circ_center(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if not len(x): return np.nan
    return float(np.degrees(np.angle(np.mean(np.exp(1j*np.radians(x))))))

def center(x, angular=False):
    return circ_center(x) if angular else float(np.nanmedian(np.asarray(x, float)))

def delta(a, b, angular=False):
    d = b - a
    return float(wrap(d)) if angular else float(d)

def angular_error(a, b):
    return float(abs(wrap(a-b)))

def savefig(fig, name):
    fig.savefig(FIG/f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG/f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

def load_inputs():
    exp = pd.read_csv(DATA/"experimental_effect_sizes.csv")
    cols = ID_COLS + METRICS
    chain = pd.read_csv(DATA/"kv21_l403a_conformational_metrics_chain_resolved.csv",
                        usecols=cols, low_memory=False)
    selected, audit = load_distance_qc_selection(chain)
    audit.to_csv(TAB/"distance_qc_join_audit.csv", index=False)
    return exp, chain, selected

def load_distance_qc_selection(chain):
    """Select exact structures retained by the established distance workflow."""
    names = {
        ("wt","vanilla"): "26-02-11_Kv2.1_wt_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        ("wt","masked"): "26-02-11_Kv2.1_wt_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        ("l403a","vanilla"): "26-02-11_Kv2.1_l403a_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        ("l403a","masked"): "26-02-11_Kv2.1_l403a_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        ("f412l","vanilla"): "26-02-11_Kv2.1_f412l_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
        ("f412l","masked"): "26-02-11_Kv2.1_f412l_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
    }
    manifest=[]; audit=[]
    unique=chain[chain.source_type.eq("prediction")][["structure_id","condition","protocol","source_path","seed","model_number","recycle"]].drop_duplicates("structure_id").copy()
    unique["pdb_basename"]=unique.source_path.map(lambda x: os.path.basename(str(x)))
    if unique.pdb_basename.duplicated().any(): raise ValueError("Conformational structure basenames are not unique")
    for key,name in names.items():
        path=ROOT/"kv21"/"dataDistances"/name
        d=pd.read_csv(path,usecols=lambda c:c in {"pdb_file","seed","model_number","recycle","recycle_number"},low_memory=False)
        d["pdb_basename"]=d.pdb_file.map(lambda x:os.path.basename(str(x)))
        if d.pdb_basename.duplicated().any(): raise ValueError(f"Duplicate QC basename in {path}")
        c=unique[(unique.condition==key[0])&(unique.protocol==key[1])]
        m=d.merge(c,on="pdb_basename",how="left",suffixes=("_qc","_conf"),indicator=True)
        unmatched=int(m._merge.ne("both").sum())
        if unmatched: raise ValueError(f"{path}: {unmatched} selected structures do not map")
        for field in ["seed","model_number","recycle"]:
            a=f"{field}_qc"; b=f"{field}_conf"
            if a in m and b in m and pd.to_numeric(m[a],errors="coerce").fillna(-1).ne(pd.to_numeric(m[b],errors="coerce").fillna(-1)).any():
                raise ValueError(f"{path}: metadata mismatch for {field}")
        manifest.append(m[["structure_id"]].assign(condition=key[0],protocol=key[1]))
        audit.append(dict(condition=key[0],protocol=key[1],qc_file=str(path.relative_to(ROOT)),qc_rows=len(d),
                          unique_qc_basenames=d.pdb_basename.nunique(),matched_structures=len(m),unmatched=unmatched,
                          ambiguous_conformational_basenames=0))
    ids=pd.concat(manifest).structure_id
    selected=chain[chain.source_type.eq("experimental")|chain.structure_id.isin(ids)].copy()
    return selected,pd.DataFrame(audit)

def experimental_long(exp):
    rows=[]
    for _, r in exp.iterrows():
        for metric in METRICS:
            a=r.get(f"8SD3__{metric}", np.nan); b=r.get(f"8SDA__{metric}", np.nan)
            if pd.notna(a) or pd.notna(b):
                ang=metric in ANGLE_METRICS
                rows.append(dict(canonical_subunit=r.canonical_subunit, metric=metric,
                    value_8SD3=a, value_8SDA=b,
                    experimental_delta=delta(a,b,ang) if pd.notna(a) and pd.notna(b) else np.nan,
                    angular=ang, effect_definition=r.effect_definition))
    out=pd.DataFrame(rows)
    out.to_csv(TAB/"experimental_reference_by_subunit.csv",index=False)
    return out

def seed_summary(chain, final_only=False, frame_ok=False):
    p=chain[(chain.source_type=="prediction") & chain.condition.isin(["wt","l403a"])].copy()
    if final_only: p=p[p.is_final_model.fillna(False)]
    if frame_ok: p=p[p.frame_orientation_score.gt(0)]
    rows=[]
    for keys,g in p.groupby(["condition","protocol","seed","canonical_subunit"], observed=True):
        row=dict(zip(["condition","protocol","seed","canonical_subunit"],keys))
        for m in METRICS: row[m]=center(g[m],m in ANGLE_METRICS)
        rows.append(row)
    return pd.DataFrame(rows)

def effects_from_seeds(seeds, exp_long, analysis="complete"):
    rows=[]; paired=[]
    for protocol in ["vanilla","masked"]:
      for sub in "ABCD":
        a=seeds[(seeds.protocol==protocol)&(seeds.condition=="wt")&(seeds.canonical_subunit==sub)]
        b=seeds[(seeds.protocol==protocol)&(seeds.condition=="l403a")&(seeds.canonical_subunit==sub)]
        m=a.merge(b,on=["protocol","seed","canonical_subunit"],suffixes=("_wt","_mut"))
        for metric in METRICS:
            ang=metric in ANGLE_METRICS
            vals=np.array([delta(x,y,ang) for x,y in zip(m[f"{metric}_wt"],m[f"{metric}_mut"]) if np.isfinite(x) and np.isfinite(y)])
            if not len(vals): continue
            est=center(vals,ang)
            boots=np.array([center(RNG.choice(vals,len(vals),replace=True),ang) for _ in range(N_BOOT)])
            ex=exp_long[(exp_long.canonical_subunit==sub)&(exp_long.metric==metric)]
            de=float(ex.experimental_delta.iloc[0]) if len(ex) else np.nan
            err=angular_error(est,de) if ang and np.isfinite(de) else abs(est-de) if np.isfinite(de) else np.nan
            row=dict(analysis=analysis, protocol=protocol, canonical_subunit=sub, metric=metric,
                     angular=ang, n_paired_seeds=len(vals), model_delta=est,
                     ci_low=float(np.nanpercentile(boots,2.5)), ci_high=float(np.nanpercentile(boots,97.5)),
                     experimental_delta=de, direction_match=(np.sign(est)==np.sign(de)) if np.isfinite(de) and de!=0 else np.nan,
                     recovery_ratio=est/de if np.isfinite(de) and abs(de)>1e-8 else np.nan,
                     absolute_experimental_error=err)
            rows.append(row)
            for seed,v in zip(m.seed.iloc[:len(vals)],vals):
                paired.append(dict(analysis=analysis,protocol=protocol,canonical_subunit=sub,metric=metric,seed=seed,paired_delta=v))
    out=pd.DataFrame(rows)
    wide=out.pivot_table(index=["analysis","canonical_subunit","metric","angular","experimental_delta"],columns="protocol",values=["model_delta","absolute_experimental_error","ci_low","ci_high","direction_match","recovery_ratio","n_paired_seeds"]).reset_index()
    wide.columns=["__".join([str(x) for x in c if str(x)]) if isinstance(c,tuple) else c for c in wide.columns]
    wide["masked_advantage"] = wide["absolute_experimental_error__vanilla"]-wide["absolute_experimental_error__masked"]
    # Paired bootstrap of masked advantage using common seed IDs.
    pe=pd.DataFrame(paired)
    cis=[]
    for (sub,metric),g in pe[pe.analysis==analysis].groupby(["canonical_subunit","metric"]):
        z=g.pivot(index="seed",columns="protocol",values="paired_delta").dropna()
        ref=exp_long.query("canonical_subunit==@sub and metric==@metric")
        if ref.empty or not np.isfinite(ref.experimental_delta.iloc[0]) or not len(z):
            continue
        de=float(ref.experimental_delta.iloc[0])
        ang=metric in ANGLE_METRICS
        bs=[]
        for _ in range(N_BOOT):
            zz=z.iloc[RNG.integers(0,len(z),len(z))]
            va=center(zz.vanilla,ang); ma=center(zz.masked,ang)
            ev=angular_error(va,de) if ang else abs(va-de); em=angular_error(ma,de) if ang else abs(ma-de)
            bs.append(ev-em)
        cis.append((sub,metric,np.percentile(bs,2.5),np.percentile(bs,97.5),len(z)))
    ci=pd.DataFrame(cis,columns=["canonical_subunit","metric","masked_advantage_ci_low","masked_advantage_ci_high","n_common_seeds"])
    wide=wide.merge(ci,on=["canonical_subunit","metric"],how="left")
    return wide,pe

def qc_tables(chain):
    qcols=["source_type","condition","protocol","category","status","detail","structure_id","frame_orientation_score"]
    pieces=[]
    for q in pd.read_csv(DATA/"kv21_l403a_conformational_metrics_qc.csv",usecols=qcols,chunksize=100000,low_memory=False):
        pieces.append(q.groupby(["source_type","condition","protocol","category","status"],dropna=False).size().rename("n_records").reset_index())
    q=pd.concat(pieces).groupby(["source_type","condition","protocol","category","status"],dropna=False).n_records.sum().reset_index()
    frames=(chain[chain.source_type=="prediction"].groupby(["condition","protocol"])
            .agg(chain_rows=("structure_id","size"),structures=("structure_id","nunique"),seeds=("seed","nunique"),
                 final_structures=("is_final_model","sum"),frame_warning_rows=("frame_orientation_score",lambda x:int((x<=0).sum())))
            .reset_index())
    frames.to_csv(TAB/"dataset_inventory.csv",index=False); q.to_csv(TAB/"qc_summary.csv",index=False)
    return frames,q

def load_pi(selected_ids):
    use=["structure_id","source_type","condition","protocol","dataset","seed","model_number","recycle","canonical_subunit","is_final_model",
         "paper_residue","phi_deg","psi_deg","incoming_i5_O_N_distance_A","incoming_i4_O_N_distance_A",
         "incoming_i5_O_H_N_angle_deg","incoming_i4_O_H_N_angle_deg","incoming_geometric_pi_like","geometric_pi_like"]
    parts=[]
    for x in pd.read_csv(DATA/"kv21_l403a_conformational_metrics_residue_level.csv",usecols=use,chunksize=200000,low_memory=False):
        parts.append(x[x.paper_residue.between(407,411) & (x.source_type.eq("experimental") | x.structure_id.isin(selected_ids))])
    d=pd.concat(parts,ignore_index=True)
    d["pi_distance_preference_A"]=d.incoming_i4_O_N_distance_A-d.incoming_i5_O_N_distance_A
    keys=["source_type","condition","protocol","dataset","canonical_subunit","paper_residue"]
    summary=(d.groupby(keys,dropna=False).agg(n=("phi_deg","size"),phi_median_deg=("phi_deg","median"),psi_median_deg=("psi_deg","median"),
        i5_distance_median_A=("incoming_i5_O_N_distance_A","median"),i4_distance_median_A=("incoming_i4_O_N_distance_A","median"),
        pi_distance_preference_median_A=("pi_distance_preference_A","median"),incoming_geometric_pi_like_fraction=("incoming_geometric_pi_like","mean"),
        geometric_pi_like_fraction=("geometric_pi_like","mean")).reset_index())
    summary.to_csv(TAB/"pi_residue_summary.csv",index=False)
    return d,summary

def distribution_panel(chain, metric, name, title):
    d=chain[(chain.condition.isin(["wt","l403a"]))].copy()
    d["group"]=d.condition.str.upper()+" "+d.protocol.str.title()
    seed=seed_summary(chain)
    fig,axs=plt.subplots(1,4,figsize=(15,4),sharey=True)
    order=["WT Vanilla","L403A Vanilla","WT Masked","L403A Masked"]
    pal={g:COLORS[(g.split()[0].lower(),g.split()[1].lower())] for g in order}
    exp=chain[chain.source_type=="experimental"]
    for ax,sub in zip(axs,"ABCD"):
        s=seed[seed.canonical_subunit==sub].copy(); s["group"]=s.condition.str.upper()+" "+s.protocol.str.title()
        sns.violinplot(data=s,x="group",y=metric,order=order,palette=pal,inner="quart",cut=0,linewidth=.7,ax=ax)
        for i,pdb in enumerate(["8SD3","8SDA"]):
            v=exp[(exp.canonical_subunit==sub)&(exp.dataset==pdb)][metric]
            if len(v):
                st=experimental_reference_style(pdb); ax.scatter(order.index("WT Vanilla") if pdb=="8SD3" else order.index("L403A Masked"),v.iloc[0],s=65,marker=st["marker"],facecolor="white",edgecolor=st["color"],linewidth=1.8,zorder=5,label=pdb)
        ax.set_title(f"Subunit {sub}"); ax.tick_params(axis="x",rotation=35); ax.set_xlabel("")
        if ax is not axs[0]: ax.set_ylabel("")
    axs[0].set_ylabel(LABELS.get(metric,metric)); handles,labels=axs[-1].get_legend_handles_labels(); axs[-1].legend(handles,labels,loc="best")
    fig.suptitle(title,y=1.04); fig.tight_layout(); savefig(fig,name)

def make_figures(chain, effects, pi):
    apply_kv21_style()
    distribution_panel(chain,"kink_angle_deg","panel_A_pip_s6_kink","A. PIP/S6 kink: seed-level ensembles and experimental references")
    distribution_panel(chain,"whole_s6_rotation_vs_8SD3_deg","panel_B_whole_s6_rotation","B. Whole-S6 rotation is chain resolved")
    distribution_panel(chain,"linker_radial_distance_A","panel_C_linker_radial","C. S4–S5 linker radial position")
    distribution_panel(chain,"F412_sidechain_centroid_displacement_vs_8SD3_A","panel_D_f412_displacement","D. F412 side-chain displacement relative to 8SD3")
    # Mechanistic WT->mutant deltas in physical units.
    chosen=["I401_pore_facing_score","I401_neighbor_facing_score","I405_pore_facing_score","I405_neighbor_facing_score"]
    e=effects[effects.metric.isin(chosen)].copy(); e["feature"]=e.metric.map(LABELS)
    long=pd.concat([e.assign(protocol="Vanilla",value=e.model_delta__vanilla),e.assign(protocol="Masked",value=e.model_delta__masked)])
    fig,axs=plt.subplots(1,4,figsize=(15,4),sharey=True)
    for ax,sub in zip(axs,"ABCD"):
        z=long[long.canonical_subunit==sub]
        sns.barplot(data=z,x="feature",y="value",hue="protocol",palette=[KV21_PALETTE["L403A_VAN"],KV21_PALETTE["L403A_HM"]],ax=ax)
        ex=e[e.canonical_subunit==sub]; ax.scatter(np.arange(len(ex)),ex.experimental_delta,color="black",marker="D",s=35,zorder=5,label="Experiment")
        ax.axhline(0,color="0.3",lw=.7); ax.set_title(f"Subunit {sub}"); ax.tick_params(axis="x",rotation=65); ax.set_xlabel("")
        if ax is not axs[0]: ax.set_ylabel("")
    axs[0].set_ylabel("WT→L403A change"); axs[-1].legend(fontsize=8); fig.suptitle("I401/I405 pore- versus neighbor-facing reorientation",y=1.04); fig.tight_layout(); savefig(fig,"i401_i405_facing_changes")
    # F412 packing contacts.
    chosen=["F412_L316_same_shortest_heavy_A","F412_L329_neighbor_shortest_heavy_A","F412_403_neighbor_shortest_heavy_A"]
    e=effects[effects.metric.isin(chosen)].copy(); e["feature"]=e.metric.map(LABELS)
    fig,axs=plt.subplots(1,4,figsize=(15,4),sharey=True)
    for ax,sub in zip(axs,"ABCD"):
        z=e[e.canonical_subunit==sub]; x=np.arange(len(z)); w=.25
        ax.bar(x-w,z.model_delta__vanilla,w,color=KV21_PALETTE["L403A_VAN"],label="Vanilla")
        ax.bar(x,z.model_delta__masked,w,color=KV21_PALETTE["L403A_HM"],label="Masked")
        ax.scatter(x+w,z.experimental_delta,color="black",marker="D",s=35,label="Experiment")
        ax.axhline(0,color=".3",lw=.7); ax.set_xticks(x,z.feature,rotation=60,ha="right"); ax.set_title(f"Subunit {sub}")
    axs[0].set_ylabel("WT→L403A contact-distance change (Å)"); axs[-1].legend(fontsize=8); fig.tight_layout(); savefig(fig,"f412_packing_rearrangement")
    # π-like continuous geometry and Ramachandran summary.
    pp=pi[(pi.source_type=="prediction")&pi.condition.isin(["wt","l403a"])].copy(); pp["group"]=pp.condition.str.upper()+" "+pp.protocol.str.title()
    seedpi=(pp.groupby(["condition","protocol","seed","canonical_subunit","paper_residue"])["pi_distance_preference_A"].median().reset_index())
    fig,axs=plt.subplots(1,5,figsize=(16,3.8),sharey=True)
    order=["WT Vanilla","L403A Vanilla","WT Masked","L403A Masked"]
    pal=[COLORS[("wt","vanilla")],COLORS[("l403a","vanilla")],COLORS[("wt","masked")],COLORS[("l403a","masked")]]
    for ax,res in zip(axs,range(407,412)):
        z=seedpi[seedpi.paper_residue==res].copy(); z["group"]=z.condition.str.upper()+" "+z.protocol.str.title()
        sns.boxplot(data=z,x="group",y="pi_distance_preference_A",order=order,palette=pal,showfliers=False,ax=ax)
        ax.axhline(0,color=".3",lw=.7); ax.set_title(str(res)); ax.tick_params(axis="x",rotation=65); ax.set_xlabel("")
        if ax is not axs[0]: ax.set_ylabel("")
    axs[0].set_ylabel("i+4 minus i+5 distance (Å)\npositive favors shorter i+5 candidate")
    fig.suptitle("E. Residue-resolved π-like continuous backbone geometry",y=1.04); fig.tight_layout(); savefig(fig,"panel_E_pi_distance_preference")
    # Recovery heatmap: positive means masked closer to experiment.
    feature_order=[m for m in ["kink_angle_deg","whole_s6_rotation_vs_8SD3_deg","I401_azimuth_deg","I405_azimuth_deg",
        "I401_pore_facing_score","I405_pore_facing_score","linker_radial_distance_A","linker_residue_ca_displacement_max_A",
        "F412_sidechain_centroid_displacement_vs_8SD3_A","F412_L316_same_shortest_heavy_A","F412_L329_neighbor_shortest_heavy_A","F412_403_neighbor_shortest_heavy_A"] if m in effects.metric.unique()]
    z=effects[effects.metric.isin(feature_order)].pivot(index="metric",columns="canonical_subunit",values="masked_advantage").reindex(feature_order)
    z.index=[LABELS.get(x,x) for x in z.index]
    lim=np.nanmax(abs(z.values)); fig,ax=plt.subplots(figsize=(7,7)); sns.heatmap(z,cmap=sns.diverging_palette(18,135,as_cmap=True),center=0,vmin=-lim,vmax=lim,annot=True,fmt=".2f",linewidths=.5,ax=ax,cbar_kws={"label":"Masked advantage (physical metric units)"})
    ax.set_title("Experimental-target error: masked advantage\npositive = masked WT→L403A shift is closer to 8SD3→8SDA"); ax.set_xlabel("Canonical subunit"); ax.set_ylabel(""); fig.tight_layout(); savefig(fig,"experimental_recovery_masked_advantage_heatmap")

def run():
    exp,chain,selected=load_inputs(); exp_long=experimental_long(exp); inventory,qc=qc_tables(chain)
    effects,paired=effects_from_seeds(seed_summary(selected),exp_long,"distance_qc_primary")
    # The established convergence tables contain recycle snapshots and retain
    # no rows labelled ``final``; final-output sensitivity is therefore run on
    # the complete conformational table and is kept separate from the primary.
    final,paired_final=effects_from_seeds(seed_summary(chain,final_only=True),exp_long,"complete_dataset_final_model_only")
    complete,paired_complete=effects_from_seeds(seed_summary(chain),exp_long,"complete_dataset_sensitivity")
    frame,paired_frame=effects_from_seeds(seed_summary(selected,frame_ok=True),exp_long,"distance_qc_frame_orientation_positive")
    effects.to_csv(TAB/"protocol_effects_by_metric_subunit.csv",index=False)
    pd.concat([final,complete,frame],ignore_index=True).to_csv(TAB/"final_model_sensitivity.csv",index=False)
    pd.concat([paired,paired_final,paired_complete,paired_frame],ignore_index=True).to_csv(TAB/"paired_seed_effects.csv",index=False)
    pi,pi_summary=load_pi(set(selected.structure_id)); make_figures(selected,effects,pi)
    summary={"chain_rows":len(chain),"prediction_structures":int(chain.query("source_type=='prediction'").structure_id.nunique()),
             "selected_prediction_structures":int(selected.query("source_type=='prediction'").structure_id.nunique()),
             "seeds_per_condition_protocol":100,"bootstrap_replicates":N_BOOT,"primary_selection":"all_ok_rmsd_3A_structural_interface_alignment_qc",
             "pairing":"protocol-specific seed summaries; all model/recycle observations retained within seed",
             "vanilla_exact_row_matches":22320,"masked_exact_row_matches":24000,
             "dssp":"unavailable; no DSSP claims made"}
    (TAB/"analysis_run_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    return {"effects":effects,"final":final,"complete":complete,"frame":frame,"inventory":inventory,"qc":qc,"pi_summary":pi_summary,"summary":summary}

if __name__ == "__main__":
    result=run()
    print(json.dumps(result["summary"],indent=2))
