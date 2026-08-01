"""Publication-focused NaV1.5 Figure 5 panels.

The functions in this module intentionally restrict the main-text comparison
to the native WT reference, QQQ vanilla, and the selected original QQQ mask.
Alternative mask designs remain useful controls but are better suited to the
supplement because their simultaneous display obscures the pore-state result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns

from shared.plotting import (
    NAV15_EXPERIMENTAL_STYLES,
    NAV15_PALETTE,
    format_channel_title,
)


REFERENCE_LABELS = {
    "6UZ3": "6UZ3 | WT inactivated",
    "7FBS": "7FBS | engineered QQQ open",
    "7DTC": "7DTC | intermediate-inactivated",
    "8VYJ": "8VYJ | native open I",
    "8VYK": "8VYK | native open II",
    "8T6L": "8T6L | toxin-bound",
}

MAIN_REFERENCE_ORDER = ("6UZ3", "7DTC", "7FBS", "8VYJ", "8VYK", "8T6L")
OPEN_REFERENCE_ORDER = ("7FBS", "8VYJ", "8VYK")

ORTHOGONAL_GATE_COLUMNS = {
    "DI–DIII": "CA_MET415_CA-ILE1154_CA",
    "DII–DIV": "CA_ALA742_CA-ILE1455_CA",
}

GATE_SPAN_COLUMNS = (
    "CA_MET415_CA-ALA742_CA",
    "CA_MET415_CA-ILE1154_CA",
    "CA_MET415_CA-ILE1455_CA",
    "CA_ALA742_CA-ILE1154_CA",
    "CA_ALA742_CA-ILE1455_CA",
    "CA_ILE1154_CA-ILE1455_CA",
)


def _reference_handles(
    references: Sequence[str],
    *,
    markersize: float = 6.5,
) -> list[Line2D]:
    handles: list[Line2D] = []
    for pdb_id in references:
        style = NAV15_EXPERIMENTAL_STYLES[pdb_id]
        handles.append(
            Line2D(
                [0],
                [0],
                marker=style["marker"],
                linestyle="none",
                markersize=markersize,
                markerfacecolor="white",
                markeredgecolor=style["color"],
                markeredgewidth=1.2,
                label=REFERENCE_LABELS[pdb_id],
            )
        )
    return handles


def _protocol_handles() -> list[Patch]:
    return [
        Patch(
            facecolor=NAV15_PALETTE["QQQ_VAN"],
            edgecolor="#4A3C50",
            linewidth=0.7,
            label="QQQ | vanilla",
        ),
        Patch(
            facecolor=NAV15_PALETTE["QQQ_HM"],
            edgecolor="#35233D",
            linewidth=0.7,
            label="QQQ | masked",
        ),
    ]


def _style_axis(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#EEE9F2", linewidth=0.55, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)


def _draw_reference_column(
    ax: plt.Axes,
    experimental_shape: pd.DataFrame,
    column: str,
    *,
    x: float = 2.0,
    references: Sequence[str] = MAIN_REFERENCE_ORDER,
) -> None:
    offsets = np.linspace(-0.16, 0.16, len(references))
    for offset, pdb_id in zip(offsets, references):
        row = experimental_shape.loc[
            experimental_shape["dataset"].astype(str).eq(pdb_id)
        ]
        if row.empty:
            continue
        style = NAV15_EXPERIMENTAL_STYLES[pdb_id]
        ax.scatter(
            x + offset,
            float(row.iloc[0][column]),
            marker=style["marker"],
            s=42,
            facecolor="white",
            edgecolor=style["color"],
            linewidth=1.15,
            zorder=7,
        )


def plot_mask_tradeoff(
    summaries: pd.DataFrame,
    models: pd.DataFrame,
    experimental_shape: pd.DataFrame,
) -> plt.Figure:
    """Show the native-latch cost and the selected QQQ pore response."""
    fig, axes = plt.subplots(1, 3, figsize=(14.8, 6.1))

    summary_index = summaries.set_index("Dataset")
    wt_order = ["WT | vanilla", "WT | masked v2"]
    wt_values = summary_index.loc[wt_order, "Pocket-engaged fraction"]
    wt_colors = [NAV15_PALETTE["WT_VAN"], NAV15_PALETTE["WT_MASKED_V2"]]
    bars = axes[0].bar(
        range(2),
        wt_values,
        color=wt_colors,
        edgecolor="#4A3C50",
        linewidth=0.75,
        width=0.66,
    )
    axes[0].set_xticks(range(2), wt_order, rotation=18, ha="right")
    axes[0].set_ylabel("Models with ≥2 motif-pocket contacts ≤6 Å")
    axes[0].set_ylim(0, max(0.65, float(wt_values.max()) * 1.14))
    axes[0].set_title("Native IFM engagement", pad=17)
    for bar, value in zip(bars, wt_values):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.012,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#403646",
        )

    selected = models[
        models["Dataset"].isin(["QQQ | vanilla", "QQQ | masked"])
    ].copy()
    order = ["QQQ | vanilla", "QQQ | masked"]
    qqq_colors = [NAV15_PALETTE["QQQ_VAN"], NAV15_PALETTE["QQQ_HM"]]
    for ax, column, title, ylabel in [
        (
            axes[1],
            "Gate diagonal (Å)",
            "Intracellular-gate size",
            "Mean cross-pore diagonal (Å)",
        ),
        (
            axes[2],
            "Gate aspect ratio",
            "Four-domain gate shape",
            "Side aspect ratio (1 = square)",
        ),
    ]:
        sns.violinplot(
            data=selected,
            x="Dataset",
            y=column,
            order=order,
            palette=qqq_colors,
            inner="quartile",
            cut=0,
            linewidth=0.8,
            density_norm="width",
            ax=ax,
        )
        _draw_reference_column(
            ax,
            experimental_shape,
            "mean_diagonal_A" if column == "Gate diagonal (Å)" else "side_aspect_ratio",
            x=0.0,
        )
        _draw_reference_column(
            ax,
            experimental_shape,
            "mean_diagonal_A" if column == "Gate diagonal (Å)" else "side_aspect_ratio",
            x=1.0,
        )
        ax.set_xticks([0, 1], order)
        ax.tick_params(axis="x", rotation=18)
        for tick in ax.get_xticklabels():
            tick.set_ha("right")
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=17)
    axes[2].axhline(1.0, color="#8C7A96", linestyle=":", linewidth=1.0)

    for ax in axes:
        _style_axis(ax)

    reference_legend = fig.legend(
        handles=_reference_handles(MAIN_REFERENCE_ORDER),
        title="Experimental structures",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.add_artist(reference_legend)
    fig.suptitle(
        format_channel_title(
            "Nav1.5 | native IFM constraint and QQQ pore exploration"
        ),
        fontsize=17,
        fontweight="semibold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.18, 1, 0.91), w_pad=2.2)
    return fig


def plot_focused_pore_shape(
    ensemble_shape: pd.DataFrame,
    experimental_shape: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    sample_per_dataset: int = 1800,
    show_legend: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Joint pore-size/shape landscape for QQQ vanilla and selected masking."""
    own_figure = ax is None
    if own_figure:
        fig, ax = plt.subplots(figsize=(8.8, 7.4))
    else:
        fig = ax.figure
    assert ax is not None

    colors = {
        "QQQ | vanilla": NAV15_PALETTE["QQQ_VAN"],
        "QQQ | masked": NAV15_PALETTE["QQQ_HM"],
    }
    for label in ("QQQ | vanilla", "QQQ | masked"):
        part = ensemble_shape.loc[ensemble_shape["dataset"].eq(label)]
        draw = part.sample(
            min(len(part), sample_per_dataset), random_state=52
        )
        ax.scatter(
            draw["mean_diagonal_A"],
            draw["side_aspect_ratio"],
            s=13,
            alpha=0.20,
            color=colors[label],
            edgecolor="none",
            rasterized=True,
        )
        ax.scatter(
            part["mean_diagonal_A"].median(),
            part["side_aspect_ratio"].median(),
            s=90,
            color=colors[label],
            edgecolor="white",
            linewidth=1.35,
            zorder=6,
        )

    for pdb_id in MAIN_REFERENCE_ORDER:
        row = experimental_shape.loc[
            experimental_shape["dataset"].astype(str).eq(pdb_id)
        ]
        if row.empty:
            continue
        style = NAV15_EXPERIMENTAL_STYLES[pdb_id]
        ax.scatter(
            float(row.iloc[0]["mean_diagonal_A"]),
            float(row.iloc[0]["side_aspect_ratio"]),
            marker=style["marker"],
            s=62,
            facecolor="white",
            edgecolor=style["color"],
            linewidth=1.35,
            zorder=8,
        )

    ax.axhline(1.0, color="#8C7A96", linestyle=":", linewidth=1.0)
    ax.set(
        xlabel="Mean cross-pore diagonal (Å)",
        ylabel="Side aspect ratio (1 = square)",
        title="QQQ intracellular-gate geometry",
    )
    ax.grid(color="#EEE9F2", linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    sns.despine(ax=ax)

    if show_legend:
        handles = _protocol_handles() + _reference_handles(MAIN_REFERENCE_ORDER)
        legend_kwargs = {
            "handles": handles,
            "title": "Ensembles and experimental structures",
            "loc": "lower center" if own_figure else "upper center",
            "bbox_to_anchor": (0.5, 0.015) if own_figure else (0.5, -0.27),
            "ncol": 3,
            "frameon": True,
            "facecolor": "white",
            "edgecolor": "#D8D1DC",
        }
        if own_figure:
            fig.legend(**legend_kwargs)
        else:
            ax.legend(**legend_kwargs)
    if own_figure:
        fig.suptitle(
            format_channel_title(
                "Nav1.5 | selected QQQ mask samples expanded pore geometry"
            ),
            fontsize=16,
            fontweight="semibold",
            y=0.99,
        )
        fig.subplots_adjust(left=0.14, right=0.98, top=0.82, bottom=0.34)
    return fig, ax


def _draw_rmsd_intervals(
    ax: plt.Axes,
    summary: pd.DataFrame,
    measurement: str,
    *,
    references: Sequence[str] = OPEN_REFERENCE_ORDER,
    annotate_difference: bool = False,
) -> None:
    subset = summary[
        summary["measurement"].eq(measurement)
        & summary["reference_id"].isin(references)
        & summary["protocol"].isin(["vanilla", "masked"])
    ].copy()
    offsets = {"vanilla": -0.11, "masked": 0.11}
    colors = {
        "vanilla": NAV15_PALETTE["QQQ_VAN"],
        "masked": NAV15_PALETTE["QQQ_HM"],
    }
    edgecolors = {"vanilla": "#5A4A62", "masked": "#35233D"}

    for x, reference in enumerate(references):
        rows = subset.loc[subset["reference_id"].eq(reference)]
        for protocol in ("vanilla", "masked"):
            row = rows.loc[rows["protocol"].eq(protocol)]
            if row.empty:
                continue
            row = row.iloc[0]
            xpos = x + offsets[protocol]
            ax.vlines(
                xpos,
                row["p05"],
                row["p95"],
                color=colors[protocol],
                linewidth=1.4,
                alpha=0.85,
                zorder=2,
            )
            ax.vlines(
                xpos,
                row["q25"],
                row["q75"],
                color=colors[protocol],
                linewidth=7.0,
                alpha=0.95,
                zorder=3,
            )
            ax.scatter(
                xpos,
                row["median"],
                s=54,
                color=colors[protocol],
                edgecolor=edgecolors[protocol],
                linewidth=0.8,
                zorder=4,
            )
        if annotate_difference and set(rows["protocol"]) >= {"vanilla", "masked"}:
            med = rows.set_index("protocol")["median"]
            y = max(rows["p95"]) + 0.04
            ax.text(
                x,
                y,
                f"Δ {med['masked'] - med['vanilla']:+.2f} Å",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#5F5666",
            )

    labels = [REFERENCE_LABELS[pdb_id].replace(" | ", "\n", 1) for pdb_id in references]
    ax.set_xticks(range(len(references)), labels)
    ax.set_ylabel("Core-aligned Cα RMSD (Å)")
    ax.set_xlabel("")
    _style_axis(ax)


def plot_pore_rmsd_summary(summary: pd.DataFrame) -> plt.Figure:
    """Final-allOK3 pore RMSD intervals with consistent protocol colors."""
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2), sharex=True)
    measurements = (
        ("pore_domain__ca__core_aligned_rmsd_A", "Pore domain"),
        ("DII_s6__ca__core_aligned_rmsd_A", "DII S6"),
    )
    for ax, (measurement, title) in zip(axes, measurements):
        _draw_rmsd_intervals(
            ax,
            summary,
            measurement,
            annotate_difference=measurement.startswith("DII"),
        )
        ax.set_title(title)

    protocol_legend = fig.legend(
        handles=_protocol_handles(),
        title="QQQ ensemble",
        loc="lower center",
        bbox_to_anchor=(0.28, -0.015),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.add_artist(protocol_legend)
    fig.legend(
        handles=_reference_handles(OPEN_REFERENCE_ORDER),
        title="Experimental RMSD reference",
        loc="lower center",
        bbox_to_anchor=(0.72, -0.015),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.suptitle(
        format_channel_title(
            "Nav1.5 QQQ | selected mask improves open-pore resemblance"
        ),
        fontsize=16.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.135,
        "Points: median; thick bars: IQR; thin bars: 5th–95th percentiles. Lower RMSD indicates greater resemblance.",
        ha="center",
        va="bottom",
        fontsize=8.8,
        color="#5F5666",
    )
    fig.tight_layout(rect=(0, 0.23, 1, 0.93), w_pad=2.0)
    return fig


def plot_figure5_pore_candidates(
    ensemble_shape: pd.DataFrame,
    experimental_shape: pd.DataFrame,
    rmsd_summary: pd.DataFrame,
) -> plt.Figure:
    """Recommended E/F pair: joint gate geometry and DII-S6 resemblance."""
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.8))
    plot_focused_pore_shape(
        ensemble_shape,
        experimental_shape,
        ax=axes[0],
        sample_per_dataset=1800,
        show_legend=False,
    )
    _draw_rmsd_intervals(
        axes[1],
        rmsd_summary,
        "DII_s6__ca__core_aligned_rmsd_A",
        annotate_difference=True,
    )
    axes[1].set_title("DII S6 resemblance to open references")

    ensemble_legend = fig.legend(
        handles=_protocol_handles(),
        title="QQQ ensemble",
        loc="lower center",
        bbox_to_anchor=(0.25, -0.035),
        ncol=2,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.add_artist(ensemble_legend)
    fig.legend(
        handles=_reference_handles(MAIN_REFERENCE_ORDER),
        title="Experimental structures",
        loc="lower center",
        bbox_to_anchor=(0.70, -0.065),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.suptitle(
        format_channel_title(
            "Nav1.5 QQQ | masking shifts intracellular-pore sampling"
        ),
        fontsize=16.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.tight_layout(rect=(0, 0.19, 1, 0.93), w_pad=2.2)
    return fig


def plot_orthogonal_gate_geometry(
    datasets: Mapping[str, tuple[pd.DataFrame, str]],
    experimental_shape: pd.DataFrame,
    *,
    sample_per_dataset: int = 1600,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Compare the two opposing-domain gate axes and engagement coordination.

    The engagement coordinate is the mean of the central IFM/QQQ residue's
    two terminal-side-chain distances to the sequence-verified receptor
    asparagines.  It is reported as a correlation annotation rather than a
    color scale so the canonical WT/QQQ and vanilla/masked colors remain
    consistent with the rest of the project.
    """
    colors = {
        "WT | vanilla": NAV15_PALETTE["WT_VAN"],
        "WT | masked v2": NAV15_PALETTE["WT_MASKED_V2"],
        "QQQ | vanilla": NAV15_PALETTE["QQQ_VAN"],
        "QQQ | masked": NAV15_PALETTE["QQQ_HM"],
    }
    order = ["WT | vanilla", "WT | masked v2", "QQQ | vanilla", "QQQ | masked"]
    records: list[dict[str, float | int | str]] = []
    prepared: dict[str, pd.DataFrame] = {}

    for label in order:
        frame, condition = datasets[label]
        motif = "PHE" if condition.upper() == "WT" else "GLN"
        engagement_columns = [
            f"shortest_{motif}1170-ASN1343",
            f"shortest_{motif}1170-ASN1449",
        ]
        required = [*ORTHOGONAL_GATE_COLUMNS.values(), *engagement_columns]
        missing = [column for column in required if column not in frame]
        if missing:
            raise KeyError(f"{label}: missing orthogonal-gate columns {missing}")

        work = pd.DataFrame(
            {
                "DI–DIII (Å)": pd.to_numeric(
                    frame[ORTHOGONAL_GATE_COLUMNS["DI–DIII"]], errors="coerce"
                ),
                "DII–DIV (Å)": pd.to_numeric(
                    frame[ORTHOGONAL_GATE_COLUMNS["DII–DIV"]], errors="coerce"
                ),
                "Engagement distance (Å)": frame[engagement_columns]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1),
            }
        ).dropna()
        work["Mean gate diagonal (Å)"] = (
            work["DI–DIII (Å)"] + work["DII–DIV (Å)"]
        ) / 2.0
        work["Absolute diagonal mismatch (Å)"] = (
            work["DI–DIII (Å)"] - work["DII–DIV (Å)"]
        ).abs()
        prepared[label] = work
        records.append(
            {
                "Dataset": label,
                "n": len(work),
                "DI–DIII median (Å)": work["DI–DIII (Å)"].median(),
                "DII–DIV median (Å)": work["DII–DIV (Å)"].median(),
                "Mean gate median (Å)": work["Mean gate diagonal (Å)"].median(),
                "Diagonal mismatch median (Å)": work[
                    "Absolute diagonal mismatch (Å)"
                ].median(),
                "Engagement-distance median (Å)": work[
                    "Engagement distance (Å)"
                ].median(),
                "Spearman engagement–gate": work["Engagement distance (Å)"].corr(
                    work["Mean gate diagonal (Å)"], method="spearman"
                ),
            }
        )

    summary = pd.DataFrame(records)
    all_x = pd.concat([work["DI–DIII (Å)"] for work in prepared.values()])
    all_y = pd.concat([work["DII–DIV (Å)"] for work in prepared.values()])
    experimental_x = pd.to_numeric(experimental_shape["DI–DIII"], errors="coerce")
    experimental_y = pd.to_numeric(experimental_shape["DII–DIV"], errors="coerce")
    lower = min(
        all_x.quantile(0.005),
        all_y.quantile(0.005),
        experimental_x.min(),
        experimental_y.min(),
    ) - 0.35
    upper = max(
        all_x.quantile(0.995),
        all_y.quantile(0.995),
        experimental_x.max(),
        experimental_y.max(),
    ) + 0.35

    fig, axes = plt.subplots(2, 2, figsize=(11.8, 10.0), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, order):
        work = prepared[label]
        draw = work.sample(min(len(work), sample_per_dataset), random_state=73)
        ax.scatter(
            draw["DI–DIII (Å)"],
            draw["DII–DIV (Å)"],
            s=13,
            alpha=0.22,
            color=colors[label],
            edgecolor="none",
            rasterized=True,
        )
        ax.scatter(
            work["DI–DIII (Å)"].median(),
            work["DII–DIV (Å)"].median(),
            s=92,
            color=colors[label],
            edgecolor="white",
            linewidth=1.35,
            zorder=7,
        )
        for pdb_id in MAIN_REFERENCE_ORDER:
            row = experimental_shape.loc[
                experimental_shape["dataset"].astype(str).eq(pdb_id)
            ]
            if row.empty:
                continue
            style = NAV15_EXPERIMENTAL_STYLES[pdb_id]
            ax.scatter(
                float(row.iloc[0]["DI–DIII"]),
                float(row.iloc[0]["DII–DIV"]),
                marker=style["marker"],
                s=48,
                facecolor="white",
                edgecolor=style["color"],
                linewidth=1.15,
                zorder=8,
            )
        ax.plot(
            [lower, upper],
            [lower, upper],
            color="#8C7A96",
            linestyle="--",
            linewidth=0.9,
            zorder=1,
        )
        row = summary.loc[summary["Dataset"].eq(label)].iloc[0]
        ax.text(
            0.03,
            0.97,
            (
                f"median ({row['DI–DIII median (Å)']:.2f}, "
                f"{row['DII–DIV median (Å)']:.2f}) Å\n"
                f"engagement–gate ρ = {row['Spearman engagement–gate']:.2f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.8,
            color="#5F5666",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#DED7E2",
                "alpha": 0.88,
            },
        )
        ax.set_title(label)
        ax.set_xlim(lower, upper)
        ax.set_ylim(lower, upper)
        ax.grid(color="#EEE9F2", linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    fig.supxlabel("DI–DIII activation-gate diagonal (Å)", y=0.165)
    fig.supylabel("DII–DIV activation-gate diagonal (Å)", x=0.035)
    ensemble_handles = [
        Patch(
            facecolor=colors[label],
            edgecolor="#4A3C50",
            linewidth=0.7,
            label=label,
        )
        for label in order
    ]
    first_legend = fig.legend(
        handles=ensemble_handles,
        title="Ensembles",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=4,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.add_artist(first_legend)
    fig.legend(
        handles=_reference_handles(MAIN_REFERENCE_ORDER),
        title="Experimental structures",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.suptitle(
        format_channel_title(
            "Nav1.5 | orthogonal activation-gate geometry and IFM/QQQ engagement"
        ),
        fontsize=16.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.tight_layout(rect=(0.05, 0.23, 1, 0.95), h_pad=1.8, w_pad=1.6)
    return summary, fig


def plot_gate_engagement_coupling(
    datasets: Mapping[str, tuple[pd.DataFrame, str]],
    *,
    sample_per_dataset: int = 2200,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Plot the per-model coupling between gate span and motif separation.

    Gate openness is the maximum of the six Cα distances connecting the four
    intracellular-gate landmarks.  IFM/QQQ separation is the mean of the two
    Cα distances from the central motif residue to the sequence-verified
    receptor asparagines.  Both coordinates therefore remain transparent
    distance summaries rather than inferred functional-state assignments.
    """
    order = ["WT | vanilla", "WT | masked v2", "QQQ | vanilla", "QQQ | masked"]
    colors = {
        "WT | vanilla": NAV15_PALETTE["WT_VAN"],
        "WT | masked v2": NAV15_PALETTE["WT_MASKED_V2"],
        "QQQ | vanilla": NAV15_PALETTE["QQQ_VAN"],
        "QQQ | masked": NAV15_PALETTE["QQQ_HM"],
    }
    prepared: dict[str, pd.DataFrame] = {}
    records: list[dict[str, float | int | str]] = []

    for label in order:
        frame, condition = datasets[label]
        motif = "PHE" if condition.upper() == "WT" else "GLN"
        engagement_columns = (
            f"CA_{motif}1170_CA-ASN1343_CA",
            f"CA_{motif}1170_CA-ASN1449_CA",
        )
        required = [*GATE_SPAN_COLUMNS, *engagement_columns]
        missing = [column for column in required if column not in frame]
        if missing:
            raise KeyError(f"{label}: missing gate–engagement columns {missing}")

        work = pd.DataFrame(
            {
                "Gate openness (Å)": frame[list(GATE_SPAN_COLUMNS)]
                .apply(pd.to_numeric, errors="coerce")
                .max(axis=1, skipna=True),
                "IFM/QQQ separation (Å)": frame[list(engagement_columns)]
                .apply(pd.to_numeric, errors="coerce")
                .mean(axis=1, skipna=True),
            }
        ).dropna()
        rho = work["Gate openness (Å)"].corr(
            work["IFM/QQQ separation (Å)"], method="spearman"
        )
        prepared[label] = work
        records.append(
            {
                "Dataset": label,
                "n": len(work),
                "Gate median (Å)": work["Gate openness (Å)"].median(),
                "Gate IQR (Å)": (
                    work["Gate openness (Å)"].quantile(0.75)
                    - work["Gate openness (Å)"].quantile(0.25)
                ),
                "IFM/QQQ separation median (Å)": work[
                    "IFM/QQQ separation (Å)"
                ].median(),
                "IFM/QQQ separation IQR (Å)": (
                    work["IFM/QQQ separation (Å)"].quantile(0.75)
                    - work["IFM/QQQ separation (Å)"].quantile(0.25)
                ),
                "Spearman gate–engagement": rho,
            }
        )

    summary = pd.DataFrame(records)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8), sharex=True, sharey=True)
    for ax, label in zip(axes.flat, order):
        work = prepared[label]
        color = colors[label]
        draw = work.sample(min(len(work), sample_per_dataset), random_state=91)
        ax.scatter(
            draw["Gate openness (Å)"],
            draw["IFM/QQQ separation (Å)"],
            s=13,
            alpha=0.22,
            color=color,
            edgecolor="none",
            rasterized=True,
        )
        if work["Gate openness (Å)"].nunique() > 1:
            slope, intercept = np.polyfit(
                work["Gate openness (Å)"],
                work["IFM/QQQ separation (Å)"],
                1,
            )
            xline = np.linspace(
                work["Gate openness (Å)"].quantile(0.01),
                work["Gate openness (Å)"].quantile(0.99),
                100,
            )
            ax.plot(
                xline,
                slope * xline + intercept,
                color=color,
                linewidth=1.8,
                alpha=0.95,
            )
        row = summary.loc[summary["Dataset"].eq(label)].iloc[0]
        ax.text(
            0.04,
            0.95,
            f"Spearman ρ = {row['Spearman gate–engagement']:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.4,
            color="#4F4555",
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "#DED7E2",
                "alpha": 0.9,
            },
        )
        ax.set_title(label)
        ax.grid(color="#EEE9F2", linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    fig.supxlabel(
        "Intracellular-gate openness (maximum six-pair Cα span, Å)", y=0.055
    )
    fig.supylabel(
        "IFM/QQQ–receptor separation (mean of two Cα distances, Å)", x=0.03
    )
    fig.suptitle(
        format_channel_title("Nav1.5 | IFM/QQQ separation and gate opening"),
        fontsize=16.5,
        fontweight="semibold",
        y=0.99,
    )
    fig.text(
        0.5,
        0.012,
        (
            "Each point is one final-QC model. Larger x indicates a wider "
            "intracellular gate; larger y indicates greater motif separation."
        ),
        ha="center",
        va="bottom",
        fontsize=9.3,
        color="#665B6D",
    )
    fig.tight_layout(rect=(0.05, 0.08, 1, 0.95), h_pad=1.6, w_pad=1.4)
    return summary, fig


def plot_qqq_activation_gate_diagonals(
    datasets: Mapping[str, tuple[pd.DataFrame, str]],
    experimental_shape: pd.DataFrame,
    *,
    sample_per_dataset: int = 2500,
) -> tuple[pd.DataFrame, plt.Figure]:
    """Focused QQQ comparison of the two orthogonal activation-gate spans."""
    order = ["QQQ | vanilla", "QQQ | masked"]
    colors = {
        "QQQ | vanilla": NAV15_PALETTE["QQQ_VAN"],
        "QQQ | masked": NAV15_PALETTE["QQQ_HM"],
    }
    prepared: dict[str, pd.DataFrame] = {}
    records: list[dict[str, float | int | str]] = []

    for label in order:
        frame, _ = datasets[label]
        work = pd.DataFrame(
            {
                "DI–DIII (Å)": pd.to_numeric(
                    frame[ORTHOGONAL_GATE_COLUMNS["DI–DIII"]], errors="coerce"
                ),
                "DII–DIV (Å)": pd.to_numeric(
                    frame[ORTHOGONAL_GATE_COLUMNS["DII–DIV"]], errors="coerce"
                ),
            }
        ).dropna()
        work["Absolute diagonal mismatch (Å)"] = (
            work["DI–DIII (Å)"] - work["DII–DIV (Å)"]
        ).abs()
        prepared[label] = work
        records.append(
            {
                "Dataset": label,
                "n": len(work),
                "DI–DIII median (Å)": work["DI–DIII (Å)"].median(),
                "DII–DIV median (Å)": work["DII–DIV (Å)"].median(),
                "Diagonal mismatch median (Å)": work[
                    "Absolute diagonal mismatch (Å)"
                ].median(),
            }
        )

    summary = pd.DataFrame(records)
    all_x = pd.concat([work["DI–DIII (Å)"] for work in prepared.values()])
    all_y = pd.concat([work["DII–DIV (Å)"] for work in prepared.values()])
    experimental_x = pd.to_numeric(experimental_shape["DI–DIII"], errors="coerce")
    experimental_y = pd.to_numeric(experimental_shape["DII–DIV"], errors="coerce")
    x_lower = min(all_x.quantile(0.003), experimental_x.min()) - 0.30
    x_upper = max(all_x.quantile(0.997), experimental_x.max()) + 0.30
    y_lower = min(all_y.quantile(0.003), experimental_y.min()) - 0.30
    y_upper = max(all_y.quantile(0.997), experimental_y.max()) + 0.30
    equality_lower = max(x_lower, y_lower)
    equality_upper = min(x_upper, y_upper)

    fig, axes = plt.subplots(1, 2, figsize=(13.4, 6.5), sharex=True, sharey=True)
    for ax, label in zip(axes, order):
        work = prepared[label]
        draw = work.sample(min(len(work), sample_per_dataset), random_state=73)
        ax.scatter(
            draw["DI–DIII (Å)"],
            draw["DII–DIV (Å)"],
            s=14,
            alpha=0.24,
            color=colors[label],
            edgecolor="none",
            rasterized=True,
        )
        ax.scatter(
            work["DI–DIII (Å)"].median(),
            work["DII–DIV (Å)"].median(),
            s=105,
            color=colors[label],
            edgecolor="white",
            linewidth=1.4,
            zorder=7,
        )
        for pdb_id in MAIN_REFERENCE_ORDER:
            row = experimental_shape.loc[
                experimental_shape["dataset"].astype(str).eq(pdb_id)
            ]
            if row.empty:
                continue
            style = NAV15_EXPERIMENTAL_STYLES[pdb_id]
            ax.scatter(
                float(row.iloc[0]["DI–DIII"]),
                float(row.iloc[0]["DII–DIV"]),
                marker=style["marker"],
                s=51,
                facecolor="white",
                edgecolor=style["color"],
                linewidth=1.2,
                zorder=8,
            )
        ax.plot(
            [equality_lower, equality_upper],
            [equality_lower, equality_upper],
            color="#8C7A96",
            linestyle="--",
            linewidth=0.95,
            zorder=1,
        )
        ax.set_title(label, pad=13)
        ax.set_xlim(x_lower, x_upper)
        ax.set_ylim(y_lower, y_upper)
        ax.grid(color="#EEE9F2", linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        sns.despine(ax=ax)

    fig.supxlabel(
        "DI–DIII span: M415–I1154 Cα distance (Å)",
        y=0.175,
    )
    fig.supylabel(
        "DII–DIV span: A742–I1455 Cα distance (Å)",
        x=0.035,
    )
    fig.legend(
        handles=_reference_handles(MAIN_REFERENCE_ORDER),
        title="Experimental structures",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=3,
        frameon=True,
        facecolor="white",
        edgecolor="#D8D1DC",
    )
    fig.suptitle(
        format_channel_title(
            "Nav1.5 QQQ | orthogonal activation-gate spans"
        ),
        fontsize=16.5,
        fontweight="semibold",
        y=0.985,
    )
    fig.tight_layout(rect=(0.05, 0.21, 1, 0.93), w_pad=1.8)
    return summary, fig
