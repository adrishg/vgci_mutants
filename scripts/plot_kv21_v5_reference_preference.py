#!/usr/bin/env python3
"""Plot the v5 paired 8SDA-minus-8SD3 RMSD density for one Kv2.1 sequence."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.plotting import ensemble_protocol_palette  # noqa: E402
from shared.rmsd_analysis import apply_kv21_rmsd_qc  # noqa: E402


SOURCE = ROOT / "kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD_v5.csv"
MANIFEST = ROOT / "kv21/dataRMSF/qc/kv21_all_ok3_selection_manifest.csv"
OUTPUT = ROOT / "kv21/dataRMSD/analysis/comparison_v5"
PROTOCOL_ORDER = ["Vanilla", "Masked"]
METRIC_LABELS = {
    "full_system_ca_rmsd_A": "full-system Cα RMSD",
    "general_tm_rmsd_A": "general-TM Cα RMSD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make a standalone density panel like the right-hand reference-preference panel."
    )
    parser.add_argument(
        "--sequence", choices=("wt", "l403a", "f412l"), default="l403a"
    )
    parser.add_argument(
        "--metric", choices=tuple(METRIC_LABELS), default="full_system_ca_rmsd_A"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = [
        "dataset", "sequence_condition", "protocol", "pdb_file", "model_path",
        "reference_id", "analysis_status", "analysis_error",
        "selected_core_postfit_rmsd_A", "selected_alignment_postfit_rmsd_A",
        "full_system_ca_rmsd_A", "general_tm_rmsd_A",
    ]
    raw = pd.read_csv(SOURCE, usecols=columns, low_memory=False)

    print(f"Total rows: {len(raw):,}")
    print(f"Unique models (model_path): {raw.model_path.nunique():,}")
    print("\nRows by sequence/protocol/reference:")
    print(raw.groupby(["sequence_condition", "protocol", "reference_id"]).size())
    print("\nAnalysis status:")
    print(raw.analysis_status.value_counts(dropna=False))
    print(
        "\nDuplicate model_path + reference_id rows:",
        int(raw.duplicated(["model_path", "reference_id"]).sum()),
    )
    print("\nMissing plotted/QC values:")
    print(raw[[args.metric, "selected_alignment_postfit_rmsd_A"]].isna().sum())

    manifest = pd.read_csv(MANIFEST, usecols=["pdb_basename", "all_ok_3"])
    if manifest.pdb_basename.duplicated().any():
        raise ValueError("QC manifest has duplicate pdb_basename values")
    selected = set(
        manifest.loc[manifest.all_ok_3.fillna(False), "pdb_basename"].astype(str)
    )
    in_manifest = raw.pdb_file.astype(str).map(lambda value: Path(value).name).isin(selected)
    print(f"\nFilter allOK3: {len(raw):,} -> {int(in_manifest.sum()):,} rows")
    work = raw.loc[in_manifest].copy()

    status_ok = work.analysis_status.eq("ok")
    print(f"Filter analysis_status == 'ok': {len(work):,} -> {int(status_ok.sum()):,} rows")
    work = work.loc[status_ok].copy()

    before_qc = len(work)
    work = apply_kv21_rmsd_qc(work, ROOT)
    print(f"Existing structural/alignment QC: {before_qc:,} -> {len(work):,} rows")

    sequence_rows = work.sequence_condition.str.lower().eq(args.sequence)
    print(f"Filter sequence == {args.sequence}: {len(work):,} -> {int(sequence_rows.sum()):,} rows")
    work = work.loc[sequence_rows].copy()
    work["Protocol"] = work.protocol.map({"vanilla": "Vanilla", "masked": "Masked"})

    identity = ["dataset", "sequence_condition", "protocol", "Protocol", "pdb_file"]
    duplicate_pairs = work.duplicated(identity + ["reference_id"]).sum()
    if duplicate_pairs:
        raise ValueError(f"Found {duplicate_pairs} duplicate model/reference rows after QC")
    wide = work.pivot(index=identity, columns="reference_id", values=args.metric)
    missing_reference = wide[["8SD3", "8SDA"]].isna().any(axis=1)
    print(f"Models lacking either reference after QC: {int(missing_reference.sum()):,}")
    if missing_reference.any():
        raise ValueError("Every plotted model must have both 8SD3 and 8SDA measurements")
    paired = wide.reset_index()
    paired["delta_8SDA_minus_8SD3_A"] = paired["8SDA"] - paired["8SD3"]

    OUTPUT.mkdir(parents=True, exist_ok=True)
    stem = f"kv21_{args.sequence}_{args.metric}_reference_preference_v5"
    paired.to_csv(OUTPUT / f"{stem}_paired_long.csv", index=False)
    counts = (
        paired.groupby("Protocol", observed=False)
        .agg(
            models=("pdb_file", "nunique"),
            median_delta_A=("delta_8SDA_minus_8SD3_A", "median"),
            fraction_closer_to_8SDA=("delta_8SDA_minus_8SD3_A", lambda x: (x < 0).mean()),
        )
        .reindex(PROTOCOL_ORDER)
        .reset_index()
    )
    counts.to_csv(OUTPUT / f"{stem}_summary.csv", index=False)
    print("\nPlotted model counts and summaries:")
    print(counts.to_string(index=False))

    palette = ensemble_protocol_palette("kv21", args.sequence)
    fig, ax = plt.subplots(figsize=(5.35, 4.2))
    sns.histplot(
        data=paired,
        x="delta_8SDA_minus_8SD3_A",
        hue="Protocol",
        hue_order=PROTOCOL_ORDER,
        palette=palette,
        element="step",
        fill=False,
        stat="density",
        common_norm=False,
        linewidth=1.8,
        ax=ax,
    )
    ax.axvline(0, color="#625D68", lw=0.9, ls="--")
    ax.set(
        xlabel="RMSD(8SDA | L403A) − RMSD(8SD3 | WT) (Å)",
        ylabel="Density",
        title=f"Kv2.1 {args.sequence.upper()} | {METRIC_LABELS[args.metric]} resemblance",
    )
    ax.text(0.02, 0.97, "← closer to 8SDA | L403A", transform=ax.transAxes, va="top", fontsize=9)
    ax.text(0.98, 0.97, "closer to 8SD3 | WT →", transform=ax.transAxes, va="top", ha="right", fontsize=9)
    if ax.legend_:
        ax.legend_.set_title("Protocol")
    sns.despine(ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT / f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(OUTPUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
