"""Joint intracellular-gate and channel-state analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared.plotting import (
    CAV12_PALETTE,
    KV21_PALETTE,
    NAV15_PALETTE,
    format_channel_title,
)


def _numeric_block(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    available = [column for column in columns if column in frame.columns]
    if not available:
        raise KeyError(f"None of the requested distance columns are available: {list(columns)}")
    return frame[available].apply(pd.to_numeric, errors="coerce")


def channel_state_coordinates(
    frame: pd.DataFrame,
    gate_aliases: Mapping[str, str],
    state_columns: Sequence[str],
) -> pd.DataFrame:
    """Return one gate-openness and one state coordinate per predicted model.

    Gate openness is the maximum of the six inter-domain intracellular-gate
    Cα distances. The state coordinate is the mean of the supplied,
    channel-specific distances.
    """
    gate = _numeric_block(frame, list(gate_aliases.values())).max(axis=1, skipna=True)
    state = _numeric_block(frame, state_columns).mean(axis=1, skipna=True)
    result = pd.DataFrame(
        {"Gate openness (Å)": gate, "State coordinate (Å)": state},
        index=frame.index,
    )
    return result.dropna()


def summarize_channel_state(
    coordinates: pd.DataFrame, ensemble: str, protocol: str
) -> dict[str, float | int | str]:
    gate = coordinates["Gate openness (Å)"]
    state = coordinates["State coordinate (Å)"]
    return {
        "Ensemble": ensemble,
        "Protocol": protocol,
        "n": len(coordinates),
        "Gate median (Å)": gate.median(),
        "Gate IQR (Å)": gate.quantile(0.75) - gate.quantile(0.25),
        "Gate SD (Å)": gate.std(),
        "State median (Å)": state.median(),
        "State IQR (Å)": state.quantile(0.75) - state.quantile(0.25),
        "Spearman gate–state": gate.corr(state, method="spearman"),
    }


def plot_gate_state_comparison(
    wt: pd.DataFrame,
    mutant: pd.DataFrame,
    gate_aliases: Mapping[str, str],
    wt_state_columns: Sequence[str],
    mutant_state_columns: Sequence[str],
    channel: str,
    mutant_label: str,
    protocol: str,
    state_label: str,
    colors: Sequence[str],
    max_scatter_points: int = 2200,
):
    """Plot gate distributions and their coordination with a distal state metric."""
    coordinates = {
        "WT": channel_state_coordinates(wt, gate_aliases, wt_state_columns),
        mutant_label: channel_state_coordinates(mutant, gate_aliases, mutant_state_columns),
    }
    long_rows = []
    for ensemble, part in coordinates.items():
        long_rows.extend(
            {"Ensemble": ensemble, "Gate openness (Å)": value}
            for value in part["Gate openness (Å)"]
        )

    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 5.4), gridspec_kw={"width_ratios": [0.78, 1.55]}
    )
    sns.violinplot(
        data=pd.DataFrame(long_rows),
        x="Ensemble",
        y="Gate openness (Å)",
        hue="Ensemble",
        order=["WT", mutant_label],
        palette={"WT": colors[0], mutant_label: colors[1]},
        inner="quartile",
        cut=0,
        linewidth=0.7,
        legend=False,
        ax=axes[0],
    )

    for ensemble, color in zip(("WT", mutant_label), colors):
        part = coordinates[ensemble]
        draw = part.sample(min(len(part), max_scatter_points), random_state=17)
        axes[1].scatter(
            draw["Gate openness (Å)"],
            draw["State coordinate (Å)"],
            s=11,
            alpha=0.24,
            color=color,
            edgecolor="none",
            rasterized=True,
            label=ensemble,
        )
        if len(part) > 2 and part["Gate openness (Å)"].nunique() > 1:
            slope, intercept = np.polyfit(
                part["Gate openness (Å)"], part["State coordinate (Å)"], 1
            )
            xline = np.linspace(
                part["Gate openness (Å)"].quantile(0.01),
                part["Gate openness (Å)"].quantile(0.99),
                80,
            )
            axes[1].plot(xline, slope * xline + intercept, color=color, linewidth=1.5)
        rho = part["Gate openness (Å)"].corr(
            part["State coordinate (Å)"], method="spearman"
        )
        axes[1].plot([], [], color=color, label=f"{ensemble} ρ={rho:.2f}")

    axes[0].set_xlabel("")
    axes[0].set_title("Intracellular-gate sampling")
    axes[1].set_xlabel("Intracellular-gate openness (maximum Cα span, Å)")
    axes[1].set_ylabel(state_label)
    axes[1].set_title("Coordination with the broader channel state")
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, labels, title="Ensemble and Spearman correlation", frameon=True)
    for ax in axes:
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        format_channel_title(f"{channel} | WT vs {mutant_label} | {protocol}"),
        fontweight="bold",
    )
    fig.tight_layout()

    summary = pd.DataFrame(
        [
            summarize_channel_state(coordinates["WT"], "WT", protocol),
            summarize_channel_state(
                coordinates[mutant_label], mutant_label, protocol
            ),
        ]
    )
    wt_iqr = float(summary.loc[summary["Ensemble"].eq("WT"), "Gate IQR (Å)"].iloc[0])
    summary["Gate IQR / WT"] = summary["Gate IQR (Å)"] / wt_iqr

    def coordination_label(rho: float) -> str:
        magnitude = abs(rho)
        strength = "strong" if magnitude >= 0.5 else "moderate" if magnitude >= 0.3 else "weak"
        direction = "positive" if rho > 0 else "negative" if rho < 0 else "none"
        return f"{strength} {direction}"

    summary["Gate–state coordination"] = summary["Spearman gate–state"].map(
        coordination_label
    )
    summary["Sampling interpretation"] = "WT reference"
    mutant_row = summary["Ensemble"].eq(mutant_label)
    ratio = float(summary.loc[mutant_row, "Gate IQR / WT"].iloc[0])
    rho = float(summary.loc[mutant_row, "Spearman gate–state"].iloc[0])
    spread = "broader" if ratio >= 1.10 else "narrower" if ratio <= 0.90 else "similar"
    coherence = (
        "coordinated" if abs(rho) >= 0.3 else "not clearly coordinated"
    )
    summary.loc[mutant_row, "Sampling interpretation"] = (
        f"{spread} gate sampling; {coherence}"
    )
    return summary, fig


def vsd_bias_table(
    comparisons: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    aliases: Mapping[str, str],
) -> pd.DataFrame:
    """Calculate protocol-matched mutant-minus-WT VSD median shifts."""
    rows = []
    for protocol, (wt, mutant) in comparisons.items():
        for alias, column in aliases.items():
            if column not in wt.columns or column not in mutant.columns:
                continue
            wt_values = pd.to_numeric(wt[column], errors="coerce").dropna()
            mutant_values = pd.to_numeric(mutant[column], errors="coerce").dropna()
            if wt_values.empty or mutant_values.empty:
                continue
            wt_iqr = wt_values.quantile(0.75) - wt_values.quantile(0.25)
            mutant_iqr = mutant_values.quantile(0.75) - mutant_values.quantile(0.25)
            rows.append(
                {
                    "Protocol": protocol,
                    "VSD landmark": alias,
                    "WT median (Å)": wt_values.median(),
                    "Mutant median (Å)": mutant_values.median(),
                    "Mutant − WT (Å)": mutant_values.median() - wt_values.median(),
                    "WT IQR (Å)": wt_iqr,
                    "Mutant IQR (Å)": mutant_iqr,
                    "Mutant / WT IQR": mutant_iqr / wt_iqr if wt_iqr else np.nan,
                }
            )
    return pd.DataFrame(rows)


def plot_vsd_bias(
    comparisons: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    aliases: Mapping[str, str],
    channel: str,
    mutant_label: str,
    colors: Mapping[str, str],
):
    """Show full WT/mutant VSD distributions and annotate median shifts."""
    table = vsd_bias_table(comparisons, aliases)
    order = list(dict.fromkeys(table["VSD landmark"]))
    protocols = list(comparisons)
    if not protocols or not order:
        raise ValueError("No shared voltage-sensor distances are available")

    channel_key = str(channel).lower().replace(".", "")

    def wt_color(protocol: str) -> str:
        low = str(protocol).lower()
        if channel_key.startswith("kv"):
            return KV21_PALETTE["WT_VAN" if "vanilla" in low else "WT_HM"]
        if channel_key.startswith("cav"):
            return CAV12_PALETTE["WT_VAN" if "vanilla" in low else "WT_HM"]
        if "no" in low and "ifm" in low:
            return NAV15_PALETTE["WT_MASKED_V2_NOIFM"]
        if "v2" in low:
            return NAV15_PALETTE["WT_MASKED_V2"]
        return NAV15_PALETTE["WT_VAN" if "vanilla" in low else "WT_HM"]

    records = []
    for protocol, (wt, mutant) in comparisons.items():
        for landmark, column in aliases.items():
            if column not in wt or column not in mutant:
                continue
            for ensemble, frame in (("WT", wt), (mutant_label, mutant)):
                records.extend(
                    {
                        "Protocol": protocol,
                        "VSD landmark": landmark,
                        "Ensemble": ensemble,
                        "Cα distance (Å)": value,
                    }
                    for value in pd.to_numeric(
                        frame[column], errors="coerce"
                    ).dropna()
                )
    plot_frame = pd.DataFrame(records)
    fig, axes = plt.subplots(
        1, len(protocols),
        figsize=(5.4 * len(protocols), 5.8),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]
    for index, protocol in enumerate(protocols):
        ax = axes[index]
        part = plot_frame[plot_frame["Protocol"].eq(protocol)]
        palette = {"WT": wt_color(protocol), mutant_label: colors[protocol]}
        sns.violinplot(
            data=part,
            x="VSD landmark",
            y="Cα distance (Å)",
            hue="Ensemble",
            order=order,
            hue_order=["WT", mutant_label],
            split=True,
            inner="quartile",
            cut=0,
            linewidth=0.7,
            palette=palette,
            ax=ax,
        )
        ax.set_title(protocol)
        ax.set_xlabel("Voltage-sensor landmark")
        ax.tick_params(axis="x", rotation=35)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        shifts = table[table["Protocol"].eq(protocol)].set_index("VSD landmark")
        upper = part["Cα distance (Å)"].quantile(0.995)
        lower = part["Cα distance (Å)"].quantile(0.005)
        padding = max((upper - lower) * 0.12, 0.15)
        ax.set_ylim(lower - padding * 0.25, upper + padding)
        for position, landmark in enumerate(order):
            if landmark not in shifts.index:
                continue
            delta = float(shifts.at[landmark, "Mutant − WT (Å)"])
            ax.text(
                position,
                upper + padding * 0.28,
                f"Δ={delta:+.2f} Å",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#4E4A50",
            )
        if index:
            ax.set_ylabel("")
            if ax.get_legend() is not None:
                ax.get_legend().remove()
        else:
            ax.set_ylabel("Cα distance (Å)")
            ax.legend(title="Ensemble", frameon=True)
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.45, linestyle="--")
        ax.grid(axis="x", visible=False)
        sns.despine(ax=ax)
    fig.suptitle(
        format_channel_title(
            f"{channel} | WT vs {mutant_label} | voltage-sensor distributions"
        ),
        fontweight="bold",
    )
    fig.text(
        0.5, 0.01,
        "Δ denotes the mutant − WT median distance within each protocol.",
        ha="center",
        fontsize=9,
        color="#665A70",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    return table, fig
