"""Main-text and supplemental Nav1.5 mask-presentation figures."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from shared.nav15_pore_shape import ensemble_gate_shape
from shared.plotting import NAV15_EXPERIMENTAL_STYLES


POCKET_RECEPTORS = {
    "F1473": "PHE1157",
    "Q1476": "GLN1160",
    "M1320": "MET1004",
    "M1652": "MET1336",
    "N1659": "ASN1343",
    "I1660": "ILE1344",
}


def pocket_metrics(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    motif = (
        ("ILE1169", "PHE1170", "MET1171")
        if condition == "WT"
        else ("GLN1169", "GLN1170", "GLN1171")
    )
    contacts = {}
    for label, receptor in POCKET_RECEPTORS.items():
        columns = [
            f"shortest_{residue}-{receptor}"
            for residue in motif
            if f"shortest_{residue}-{receptor}" in frame
        ]
        if not columns:
            raise KeyError(f"No motif contact columns available for {label}")
        contacts[label] = frame[columns].apply(
            pd.to_numeric, errors="coerce"
        ).min(axis=1)
    result = pd.DataFrame(contacts)
    result["contacts_le6"] = (result <= 6.0).sum(axis=1)
    result["engaged_fingerprint"] = result["contacts_le6"] >= 2
    result["pocket_mean_A"] = result[list(POCKET_RECEPTORS)].mean(axis=1)
    return result


def presentation_table(
    datasets: Mapping[str, tuple[pd.DataFrame, str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries, model_rows = [], []
    for label, (frame, condition) in datasets.items():
        pocket = pocket_metrics(frame, condition)
        shape = ensemble_gate_shape(frame, label)
        summaries.append(
            {
                "Dataset": label,
                "Condition": condition,
                "n": len(frame),
                "Pocket-engaged fraction": pocket["engaged_fingerprint"].mean(),
                "Pocket mean median (Å)": pocket["pocket_mean_A"].median(),
                "Gate diagonal median (Å)": shape["mean_diagonal_A"].median(),
                "Gate aspect median": shape["side_aspect_ratio"].median(),
            }
        )
        model_rows.append(
            pd.DataFrame(
                {
                    "Dataset": label,
                    "Condition": condition,
                    "Pocket engaged": pocket["engaged_fingerprint"].values,
                    "Pocket mean (Å)": pocket["pocket_mean_A"].values,
                    "Gate diagonal (Å)": shape["mean_diagonal_A"].values,
                    "Gate aspect ratio": shape["side_aspect_ratio"].values,
                }
            )
        )
    return pd.DataFrame(summaries), pd.concat(model_rows, ignore_index=True)


def plot_main_mask_tradeoff(
    summaries: pd.DataFrame,
    models: pd.DataFrame,
    palette: Mapping[str, str],
    experimental_shape: pd.DataFrame,
):
    """Main-text distance figure for the selected QQQ masking protocol."""
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8))
    order = ["WT | vanilla", "QQQ | vanilla", "QQQ | masked"]
    selected = models[models["Dataset"].isin(order)].copy()
    for ax, column, title in [
        (axes[0], "Pocket mean (Å)", "IFM/QQQ pocket proximity"),
        (axes[1], "Gate diagonal (Å)", "Intracellular pore opening"),
        (axes[2], "Gate aspect ratio", "Square–rectangle pore geometry"),
    ]:
        sns.violinplot(
            data=selected, x="Dataset", y=column, order=order,
            palette=[palette[label] for label in order],
            inner="quartile", cut=0, linewidth=0.7, ax=ax,
        )
        ax.set_xlabel("")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=18)

    for _, row in experimental_shape.iterrows():
        style = NAV15_EXPERIMENTAL_STYLES[row["dataset"]]
        for ax, column in [
            (axes[1], "mean_diagonal_A"),
            (axes[2], "side_aspect_ratio"),
        ]:
            ax.scatter(
                0.5, row[column], marker=style["marker"], s=30,
                facecolor="white", edgecolor=style["color"], linewidth=1.0,
                zorder=6,
            )
    axes[2].axhline(1.0, color="#8C7A96", linestyle=":", linewidth=0.9)
    for ax in axes:
        ax.grid(axis="y", color="#EEE9F2", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        r"$\mathrm{Na}_{\mathrm{V}}1.5$ | IFM→QQQ mutation and selected targeted-mask response",
        fontsize=15, fontweight="semibold",
    )
    fig.text(
        0.5, -0.01,
        "WT vanilla is the native reference. QQQ vanilla isolates the sequence change under the "
        "intact MSA; QQQ masked then tests whether relaxing the evolutionary prior reveals an "
        "additional mutant-associated pore geometry. Other mask designs are shown in the supplement.",
        ha="center", va="top", fontsize=8.5, color="#5F5666",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    return fig


def plot_selected_qqq_rmsd(
    rmsd: pd.DataFrame,
    palette: Mapping[str, str],
):
    """QQQ vanilla-versus-selected-mask RMSD to the 7FBS pore reference."""
    labels = {"vanilla": "QQQ | vanilla", "masked": "QQQ | masked"}
    work = rmsd[
        rmsd["sequence_condition"].astype(str).str.lower().eq("qqq")
        & rmsd["reference_id"].astype(str).eq("7FBS")
        & rmsd["protocol"].astype(str).str.lower().isin(labels)
    ].copy()
    work["Dataset"] = work["protocol"].astype(str).str.lower().map(labels)
    order = list(labels.values())
    measurements = [
        ("pore_domain__ca__core_aligned_rmsd_A", "Pore domain"),
        ("DII_s6__ca__core_aligned_rmsd_A", "DII S6"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.7))
    for ax, (column, title) in zip(axes, measurements):
        sns.violinplot(
            data=work, x="Dataset", y=column, order=order,
            palette=[palette[x] for x in order], inner="quartile",
            cut=0, linewidth=0.75, ax=ax,
        )
        ax.set(title=title, xlabel="", ylabel="Core-aligned Cα RMSD to 7FBS (Å)")
        ax.tick_params(axis="x", rotation=15)
        ax.grid(axis="y", color="#EEE9F2", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        r"$\mathrm{Na}_{\mathrm{V}}1.5$ IFM→QQQ | selected mask improves resemblance to the 7FBS pore",
        fontsize=14.5, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def plot_selected_qqq_reference_preference(
    paired: pd.DataFrame,
    palette: Mapping[str, str],
):
    """Histogram showing whether QQQ models are more 7FBS- or 8VYJ-like."""
    protocol_column = "Protocol" if "Protocol" in paired else "protocol"
    normalized = paired[protocol_column].astype(str).str.lower()
    labels = {"vanilla": "QQQ | vanilla", "masked": "QQQ | masked"}
    work = paired[normalized.isin(labels)].copy()
    work["Dataset"] = work[protocol_column].astype(str).str.lower().map(labels)
    order = list(labels.values())
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    sns.histplot(
        data=work, x="delta_b_minus_a_A", hue="Dataset", hue_order=order,
        palette={x: palette[x] for x in order}, element="step", fill=False,
        stat="density", common_norm=False, linewidth=2.0, ax=ax,
    )
    ax.axvline(0, color="#5F5666", linestyle="--", linewidth=1)
    ax.text(.02, .97, "← closer to 7FBS QQQ/open pore",
            transform=ax.transAxes, va="top", fontsize=9)
    ax.text(.98, .97, "closer to 8VYJ native open →",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)
    ax.legend(
        title="Ensemble", loc="lower right", frameon=True,
        facecolor="white", edgecolor="#D8D1DC",
    )
    ax.set(
        xlabel="Pore RMSD(7FBS) − RMSD(8VYJ) (Å)",
        ylabel="Density",
        title=r"$\mathrm{Na}_{\mathrm{V}}1.5$ IFM→QQQ | experimental-reference preference",
    )
    ax.grid(axis="y", color="#EEE9F2", linewidth=0.45, linestyle="--")
    sns.despine(ax=ax)
    fig.tight_layout()
    return fig


def plot_supplemental_mask_audit(
    summaries: pd.DataFrame,
    palette: Mapping[str, str],
):
    order = list(summaries["Dataset"])
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 5.2))
    metrics = [
        ("Pocket-engaged fraction", "Engaged pocket fraction"),
        ("Gate diagonal median (Å)", "Median gate diagonal (Å)"),
        ("Gate aspect median", "Median side aspect ratio"),
    ]
    for ax, (column, title) in zip(axes, metrics):
        ax.bar(
            range(len(order)), summaries.set_index("Dataset").loc[order, column],
            color=[palette[label] for label in order],
            edgecolor="#3D3942", linewidth=0.55,
        )
        ax.set_xticks(range(len(order)), order, rotation=34, ha="right")
        ax.set_title(title)
        ax.grid(axis="y", color="#EEE9F2", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    axes[2].axhline(1.0, color="#8C7A96", linestyle=":", linewidth=0.9)
    fig.suptitle(
        r"$\mathrm{Na}_{\mathrm{V}}1.5$ supplemental mask-design audit | all sequence and masking conditions",
        fontsize=15, fontweight="semibold",
    )
    fig.text(
        0.5, -0.015,
        "Pocket engagement is a descriptive six-residue fingerprint (≥2 shortest contacts ≤6 Å), "
        "not a functional state label. Masked v2 is matched across WT and QQQ; original masked is "
        "currently available as the strongest QQQ exploratory condition; noIFM retains IFM-site "
        "MSA information but not the complete receptor-coupling context.",
        ha="center", va="top", fontsize=8.4, color="#5F5666",
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    return fig


def plot_supplemental_pocket_profile(
    datasets: Mapping[str, tuple[pd.DataFrame, str]],
):
    rows = []
    for label, (frame, condition) in datasets.items():
        pocket = pocket_metrics(frame, condition)
        for receptor in POCKET_RECEPTORS:
            rows.append(
                {
                    "Dataset": label,
                    "Pocket residue": receptor,
                    "Median shortest distance (Å)": pocket[receptor].median(),
                }
            )
    table = pd.DataFrame(rows)
    matrix = table.pivot(
        index="Dataset", columns="Pocket residue",
        values="Median shortest distance (Å)",
    )
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    sns.heatmap(
        matrix, annot=True, fmt=".1f", cmap="Purples", linewidths=0.5,
        cbar_kws={"label": "Median whole-motif shortest distance (Å)"}, ax=ax,
    )
    ax.set_title(r"$\mathrm{Na}_{\mathrm{V}}1.5$ supplemental | complete IFM/QQQ pocket-distance profile")
    ax.set_xlabel("Canonical receptor-pocket residue")
    ax.set_ylabel("")
    fig.text(
        0.5, -0.01,
        "Distances use the minimum over the three IFM or QQQ residues. "
        "A short value indicates proximity to one motif residue and does not alone establish docking.",
        ha="center", fontsize=8.5, color="#5F5666",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return table, fig
