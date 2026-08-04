"""Focused figures for Cav1.2 Timothy-mutation IS6 coupling."""

from __future__ import annotations

from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .plotting import experimental_reference_style


CAV12_GATE_COLUMNS = (
    "CA_LEU401_CA-LEU749_CA",
    "CA_LEU401_CA-VAL1182_CA",
    "CA_LEU401_CA-ILE1516_CA",
    "CA_LEU749_CA-VAL1182_CA",
    "CA_LEU749_CA-ILE1516_CA",
    "CA_VAL1182_CA-ILE1516_CA",
)


def gate_openness(frame: pd.DataFrame) -> pd.Series:
    """Maximum of the six intracellular-gate Cα spans."""
    missing = [column for column in CAV12_GATE_COLUMNS if column not in frame]
    if missing:
        raise KeyError(f"Missing Cav1.2 gate columns: {missing}")
    return frame[list(CAV12_GATE_COLUMNS)].apply(
        pd.to_numeric, errors="coerce"
    ).max(axis=1)


def _contact_fraction(
    frame: pd.DataFrame, residue: str, partner: str, cutoff: float = 4.0
) -> float:
    column = f"shortest_{residue}-{partner}"
    if column not in frame:
        return np.nan
    return float((pd.to_numeric(frame[column], errors="coerce") <= cutoff).mean())


def contact_occupancy_table(
    datasets: Mapping[str, tuple[pd.DataFrame, str, str]]
) -> pd.DataFrame:
    """Contact occupancies for the interpretable mutation-interface pairs."""
    definitions = {
        "G402/S402–I1523": ("GLY402", "SER402", "ILE1523"),
        "G402/S402–M1524": ("GLY402", "SER402", "MET1524"),
        "G406/R406–D1528": ("GLY406", "ARG406", "ASP1528"),
        "G406/R406–D1533": ("GLY406", "ARG406", "ASP1533"),
    }
    rows = []
    for label, (frame, condition, protocol) in datasets.items():
        for contact, (wt_residue, mutant_residue, partner) in definitions.items():
            residue = wt_residue if condition == "WT" else mutant_residue
            # G402S should not be substituted at position 406, or vice versa.
            if condition == "G402S" and contact.startswith("G406"):
                residue = "GLY406"
            if condition == "G406R" and contact.startswith("G402"):
                residue = "GLY402"
            rows.append(
                {
                    "Dataset": label,
                    "Condition": condition,
                    "Protocol": protocol,
                    "Contact": contact,
                    "Contact fraction": _contact_fraction(frame, residue, partner),
                }
            )
    return pd.DataFrame(rows)


def plot_timothy_contact_occupancy(
    datasets: Mapping[str, tuple[pd.DataFrame, str, str]],
    palette: Mapping[str, str],
):
    """Compare the local contact-network changes caused by both mutations."""
    table = contact_occupancy_table(datasets)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharey=True)
    panels = [
        ("G402S", ["G402/S402–I1523", "G402/S402–M1524"]),
        ("G406R", ["G406/R406–D1528", "G406/R406–D1533"]),
    ]
    for ax, (mutant, contacts) in zip(axes, panels):
        selected = table[
            table["Condition"].isin(["WT", mutant])
            & table["Contact"].isin(contacts)
        ]
        order = [
            "WT | vanilla", f"{mutant} | vanilla",
            "WT | masked", f"{mutant} | masked",
        ]
        sns.barplot(
            data=selected, x="Contact", y="Contact fraction", hue="Dataset",
            hue_order=order, palette=palette, ax=ax,
        )
        ax.set_title(f"{mutant} local IS6-interface contacts")
        ax.set_xlabel("")
        ax.set_ylabel("Fraction with shortest distance ≤4 Å")
        ax.tick_params(axis="x", rotation=18)
        ax.legend(title="", fontsize=8, frameon=True)
        ax.grid(axis="y", color="#E8EEF5", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        r"$\mathrm{Ca}_{\mathrm{V}}1.2$ | Timothy mutations produce distinct local IS6 repacking",
        fontsize=15, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return table, fig


def plot_gate_and_local_coupling(
    datasets: Mapping[str, tuple[pd.DataFrame, str, str]],
    palette: Mapping[str, str],
):
    """Show gate distributions and mutation-specific gate coupling."""
    gate_rows = []
    for label, (frame, _, _) in datasets.items():
        gate_rows.extend(
            {"Dataset": label, "Gate openness (Å)": value}
            for value in gate_openness(frame).dropna()
        )
    gate_frame = pd.DataFrame(gate_rows)
    order = list(datasets)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.2))
    sns.violinplot(
        data=gate_frame, x="Dataset", y="Gate openness (Å)", order=order,
        palette=[palette[label] for label in order], inner="quartile",
        cut=0, linewidth=0.65, ax=axes[0],
    )
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].set_xlabel("")
    axes[0].set_title("Global intracellular-gate span")

    coupling = [
        ("G402S", "CA_LEU401_CA-ASP1533_CA", "L401–D1533 (Å)"),
        ("G406R", "CA_GLU407_CA-ARG1532_CA", "E407–R1532 (Å)"),
    ]
    for ax, (mutant, column, ylabel) in zip(axes[1:], coupling):
        for protocol in ("vanilla", "masked"):
            for condition in ("WT", mutant):
                label = f"{condition} | {protocol}"
                frame = datasets[label][0]
                draw = pd.DataFrame(
                    {
                        "gate": gate_openness(frame),
                        "local": pd.to_numeric(frame[column], errors="coerce"),
                    }
                ).dropna()
                sample = draw.sample(min(900, len(draw)), random_state=42)
                ax.scatter(
                    sample["gate"], sample["local"], s=9, alpha=0.16,
                    color=palette[label], edgecolor="none", rasterized=True,
                )
                rho = draw["gate"].corr(draw["local"], method="spearman")
                slope, intercept = np.polyfit(draw["gate"], draw["local"], 1)
                xline = np.linspace(
                    draw["gate"].quantile(0.02), draw["gate"].quantile(0.98), 60
                )
                ax.plot(
                    xline, slope * xline + intercept,
                    color=palette[label], linewidth=1.4,
                    label=f"{label}  ρ={rho:.2f}",
                )
        ax.set_xlabel("Gate openness (maximum Cα span, Å)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{mutant}: local–global coupling")
        ax.legend(fontsize=7.5, frameon=True)
        ax.grid(color="#E8EEF5", linewidth=0.4, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        r"$\mathrm{Ca}_{\mathrm{V}}1.2$ | local IS6 rearrangement is stronger than median pore opening",
        fontsize=15, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return gate_frame, fig


def plot_experimental_interface_landscapes(
    datasets: Mapping[str, tuple[pd.DataFrame, str, str]],
    palette: Mapping[str, str],
    experimental: Mapping[str, Mapping[str, float]],
):
    """Place ensemble S6-interface coordinates against three PDB references."""
    panels = [
        (
            "DI coupling frame",
            "CA_LEU401_CA-ILE1186_CA", "L401–I1186 (Å)",
            "CA_LEU401_CA-VAL753_CA", "L401–V753 (Å)",
        ),
        (
            "G406-region S6 interface",
            "CA_GLU407_CA-ARG1532_CA", "E407–R1532 (Å)",
            "CA_PHE408_CA-THR1531_CA", "F408–T1531 (Å)",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4))
    for ax, (title, xcol, xlabel, ycol, ylabel) in zip(axes, panels):
        for label, (frame, _, _) in datasets.items():
            part = frame[[xcol, ycol]].apply(pd.to_numeric, errors="coerce").dropna()
            draw = part.sample(min(800, len(part)), random_state=9)
            ax.scatter(
                draw[xcol], draw[ycol], s=9, alpha=0.14,
                color=palette[label], edgecolor="none", rasterized=True,
            )
            ax.scatter(
                part[xcol].median(), part[ycol].median(), s=55,
                color=palette[label], edgecolor="white", linewidth=0.9,
                label=label, zorder=5,
            )
        for reference_index, (pdb_id, values) in enumerate(experimental.items()):
            if xcol not in values or ycol not in values:
                continue
            style = experimental_reference_style(pdb_id, reference_index)
            ax.scatter(
                values[xcol], values[ycol], marker=style["marker"], s=58,
                facecolor="white", edgecolor=style["color"], linewidth=1.2,
                label=f"{pdb_id} experimental", zorder=7,
            )
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.grid(color="#E8EEF5", linewidth=0.4, linestyle="--")
        sns.despine(ax=ax)
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Ensembles and structural references",
        loc="lower center", bbox_to_anchor=(0.5, -0.10), ncol=3,
        fontsize=8, frameon=True,
    )
    fig.suptitle(
        r"$\mathrm{Ca}_{\mathrm{V}}1.2$ | experimental S6-interface reference landscapes",
        fontsize=15, fontweight="semibold",
    )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    return fig
