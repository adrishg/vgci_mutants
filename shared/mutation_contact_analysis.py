"""Shortest-distance analysis for newly introduced mutation side chains."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def mutation_contact_table(
    wt: pd.DataFrame,
    mutant: pd.DataFrame,
    wt_residue: str,
    mutant_residue: str,
    protocol: str,
) -> pd.DataFrame:
    """Compare partner-specific shortest distances for WT and mutant residues."""
    rows = []
    prefix = f"shortest_{mutant_residue}-"
    for mutant_column in [column for column in mutant if column.startswith(prefix)]:
        partner = mutant_column.split("-", 1)[1]
        wt_column = f"shortest_{wt_residue}-{partner}"
        mutant_values = pd.to_numeric(mutant[mutant_column], errors="coerce").dropna()
        wt_values = (
            pd.to_numeric(wt[wt_column], errors="coerce").dropna()
            if wt_column in wt
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "Protocol": protocol,
                "Partner": partner,
                "WT median (Å)": wt_values.median() if len(wt_values) else np.nan,
                "Mutant median (Å)": mutant_values.median(),
                "Mutant − WT (Å)": (
                    mutant_values.median() - wt_values.median()
                    if len(wt_values)
                    else np.nan
                ),
                "WT contact ≤4 Å": (wt_values <= 4).mean() if len(wt_values) else np.nan,
                "Mutant contact ≤4 Å": (mutant_values <= 4).mean(),
                "Mutant overlap <2 Å": (mutant_values < 2).mean(),
                "n": len(mutant_values),
            }
        )
    return pd.DataFrame(rows)


def plot_mutation_contacts(
    comparisons: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    wt_residue: str,
    mutant_residue: str,
    mutant_label: str,
    colors: Mapping[str, str],
    top_n: int = 10,
    channel: str = "Cav1.2",
):
    tables = [
        mutation_contact_table(wt, mutant, wt_residue, mutant_residue, protocol)
        for protocol, (wt, mutant) in comparisons.items()
    ]
    table = pd.concat(tables, ignore_index=True)
    ranking = (
        table.groupby("Partner")["Mutant contact ≤4 Å"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = table[table["Partner"].isin(ranking)].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(
        data=plot_df,
        x="Partner",
        y="Mutant contact ≤4 Å",
        hue="Protocol",
        order=list(ranking),
        palette=colors,
        ax=ax,
    )
    ax.set_ylabel("Fraction of models with shortest distance ≤4 Å")
    ax.set_xlabel(f"Partner of {mutant_residue}")
    ax.set_title(f"{channel} | {mutant_label} | mutation-side-chain contacts")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.45, linestyle="--")
    sns.despine(ax=ax)
    fig.tight_layout()
    return table.sort_values(
        ["Mutant contact ≤4 Å", "Mutant median (Å)"], ascending=[False, True]
    ), fig


def plot_selected_contact_distributions(
    comparisons: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    wt_residue: str,
    mutant_residue: str,
    mutant_label: str,
    partners: list[str],
    colors: Mapping[str, tuple[str, str]],
    channel: str = "Cav1.2",
):
    """Show WT and mutant shortest-distance distributions for selected partners."""
    records = []
    for protocol, (wt, mutant) in comparisons.items():
        for partner in partners:
            columns = {
                "WT": f"shortest_{wt_residue}-{partner}",
                mutant_label: f"shortest_{mutant_residue}-{partner}",
            }
            for ensemble, column in columns.items():
                frame = wt if ensemble == "WT" else mutant
                if column not in frame:
                    continue
                records.extend(
                    {
                        "Protocol": protocol,
                        "Partner": partner,
                        "Ensemble": ensemble,
                        "Shortest distance (Å)": value,
                    }
                    for value in pd.to_numeric(frame[column], errors="coerce").dropna()
                )
    plot_df = pd.DataFrame(records)
    protocols = list(comparisons)
    fig, axes = plt.subplots(
        1, len(protocols), figsize=(5.4 * len(protocols), 5.5),
        sharey=True, squeeze=False,
    )
    for index, protocol in enumerate(protocols):
        ax = axes[0, index]
        part = plot_df[plot_df["Protocol"].eq(protocol)]
        sns.violinplot(
            data=part,
            x="Partner",
            y="Shortest distance (Å)",
            hue="Ensemble",
            order=partners,
            hue_order=["WT", mutant_label],
            split=True,
            inner="quartile",
            cut=0,
            linewidth=0.6,
            palette={"WT": colors[protocol][0], mutant_label: colors[protocol][1]},
            ax=ax,
        )
        ax.axhline(4, color="#7B6D86", linewidth=0.8, linestyle="--")
        ax.axhline(2, color="#C44E52", linewidth=0.9, linestyle=":")
        ax.axhspan(0, 2, color="#F4D6D7", alpha=0.35, zorder=0)
        ax.set_title(protocol)
        ax.set_xlabel("Interaction partner")
        ax.tick_params(axis="x", rotation=35)
        if index:
            ax.get_legend().remove()
        else:
            ax.legend(title="Sequence")
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        f"{channel} | {mutant_label} | mutation-site shortest distances",
        fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Dashed line: candidate contact (4 Å). Red region: probable atomic overlap (<2 Å).",
        ha="center", fontsize=9, color="#665A70",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    return plot_df, fig
