"""Cross-project final-QC retention summaries by AlphaFold recycle."""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from shared.plotting import CAV12_PALETTE, KV21_PALETTE, NAV15_PALETTE


SEED_RE = re.compile(r"_seed_(\d+)", re.I)
RECYCLE_RE = re.compile(r"\.r(\d+)\.pdb$", re.I)
CHANNEL_ORDER = ("Cav1.2", "Kv2.1", "Nav1.5")
CHANNEL_COLORS = {
    "Cav1.2": CAV12_PALETTE["G406R_HM"],
    "Kv2.1": KV21_PALETTE["L403A_HM"],
    "Nav1.5": NAV15_PALETTE["QQQ_HM"],
}


def recycle_retention_table(datasets: dict) -> pd.DataFrame:
    """Count final-QC snapshots at each recycle for full and first-100 cohorts."""
    rows = []
    for dataset, config in datasets.items():
        for protocol in ("vanilla", "masked"):
            frame = pd.read_csv(config[protocol], usecols=["pdb_file"])
            names = frame.pdb_file.astype(str)
            parsed = pd.DataFrame({
                "seed": pd.to_numeric(names.str.extract(SEED_RE, expand=False), errors="coerce"),
                "recycle": pd.to_numeric(names.str.extract(RECYCLE_RE, expand=False), errors="coerce"),
                "pdb_file": names,
            }).dropna(subset=["seed", "recycle"])
            parsed[["seed", "recycle"]] = parsed[["seed", "recycle"]].astype(int)
            first_seeds = sorted(parsed.seed.unique())[:20]
            for cohort, selected, denominator in (
                ("Nominal first 100", parsed[parsed.seed.isin(first_seeds)], 100),
                ("Full 500", parsed, 500),
            ):
                counts = selected.groupby("recycle").pdb_file.nunique()
                for recycle in range(1, 11):
                    retained = int(counts.get(recycle, 0))
                    rows.append({
                        "channel": config["channel"], "dataset": dataset,
                        "condition": config["condition"], "protocol": protocol,
                        "cohort": cohort, "recycle": recycle,
                        "nominal_trajectories": denominator,
                        "retained": retained, "excluded": denominator-retained,
                        "retained_fraction": retained/denominator,
                    })
    return pd.DataFrame(rows)


def _channel_matrix(table, channel, cohort, value):
    part = table[(table.channel == channel) & (table.cohort == cohort)].copy()
    part["row"] = part.dataset.str.replace(r"^(CaV1\.2|Kv2\.1|NaV1\.5)\s+", "", regex=True) + " | " + part.protocol.str.capitalize()
    return part.pivot(index="row", columns="recycle", values=value)


def plot_recycle_retention(table: pd.DataFrame, output_dir: Path):
    """Create retained-percent and excluded-count heatmaps in project palettes."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {}
    for value, filename, title, cbar_label in (
        ("retained_fraction", "cross_project_retained_by_recycle_heatmap",
         "Final-QC retention by recycle", "Retained fraction"),
        ("excluded", "cross_project_excluded_by_recycle_heatmap",
         "Final-QC attrition by recycle", "Excluded trajectories"),
    ):
        fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
        for row, cohort in enumerate(("Nominal first 100", "Full 500")):
            denominator = 100 if cohort.startswith("Nominal") else 500
            for col, channel in enumerate(CHANNEL_ORDER):
                ax = axes[row, col]
                matrix = _channel_matrix(table, channel, cohort, value)
                retained = _channel_matrix(table, channel, cohort, "retained")
                annotations = retained.map(lambda x: f"{int(x)}")
                if value == "retained_fraction":
                    vmin, vmax = 0, 1
                else:
                    vmin, vmax = 0, denominator
                sns.heatmap(
                    matrix, cmap=sns.light_palette(CHANNEL_COLORS[channel], as_cmap=True),
                    vmin=vmin, vmax=vmax, annot=annotations, fmt="", linewidths=.45,
                    linecolor="white", cbar=col == 2,
                    cbar_kws={"label": cbar_label}, ax=ax,
                    annot_kws={"fontsize":7},
                )
                ax.set_title(channel if row == 0 else "", color=CHANNEL_COLORS[channel], fontweight="bold")
                ax.set_xlabel("Recycle" if row == 1 else "")
                ax.set_ylabel(cohort if col == 0 else "")
                ax.tick_params(axis="x", rotation=0)
                ax.tick_params(axis="y", rotation=0, labelsize=8)
        note = "Cell labels are retained trajectories; color encodes retained fraction" if value == "retained_fraction" else "Cell labels are retained trajectories; darker color means more excluded"
        fig.suptitle(f"{title}\n{note}", fontsize=17, fontweight="semibold")
        for suffix in ("png", "pdf"):
            fig.savefig(output_dir/f"{filename}.{suffix}", dpi=300 if suffix == "png" else None,
                        bbox_inches="tight", facecolor="white")
        figures[value] = fig
    return figures


def run_recycle_retention_analysis(datasets: dict, output_dir: Path, table_dir: Path):
    table = recycle_retention_table(datasets)
    table_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_dir/"cross_project_recycle_retention.csv", index=False)
    figures = plot_recycle_retention(table, output_dir)
    return table, figures
