"""Reusable calculations and figures for Nav1.5 IFM-latch analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .structure_distances import read_ca_residues


def calculate_contacts(pdb_path, condition, *, contact_aliases):
    atoms = read_ca_residues(pdb_path)
    values = {}
    for label, definition in contact_aliases.items():
        first, second = definition[condition]
        values[label] = (
            np.linalg.norm(atoms[first] - atoms[second])
            if first in atoms and second in atoms
            else np.nan
        )
    return values


def resolve_pdb_path(manifest_path, condition, protocol, *, root_overrides):
    original = Path(manifest_path)
    if original.is_file():
        return original
    override = root_overrides[condition, protocol]
    return Path(override) / original.name if override is not None else original


def explicit_experimental_distances(
    *, repo_root, residue_map, experimental_directory="nav15/experimental"
):
    result = {}
    for pdb_id, mapping in residue_map.items():
        atoms = read_ca_residues(
            Path(repo_root) / experimental_directory / f"{pdb_id}.pdb"
        )
        result[pdb_id] = {
            label: round(float(np.linalg.norm(atoms[first] - atoms[second])), 3)
            for label, (first, second) in mapping.items()
        }
    return result


def plot_ifm_joint(
    data,
    ensemble_a,
    ensemble_b,
    *,
    experimental_distances,
    ensemble_colors,
    latch_threshold=12.0,
    guessed_definitions=True,
):
    """Plot Cα IFM-latch coordinates with marginal histograms."""
    requested = [ensemble_a, ensemble_b]
    title_a, title_b = (
        ensemble_a.replace(" | ", " "),
        ensemble_b.replace(" | ", " "),
    )
    plot_df = data[data["Ensemble"].isin(requested)].dropna(
        subset=["IFM1", "IFM2"]
    )
    grid = sns.JointGrid(
        data=plot_df, x="IFM1", y="IFM2", height=8.0, ratio=5, space=0.08
    )
    figure, axis = grid.figure, grid.ax_joint
    colors = {
        ensemble_a: ensemble_colors[ensemble_a],
        ensemble_b: ensemble_colors[ensemble_b],
    }
    limit = max(
        34.0, float(plot_df[["IFM1", "IFM2"]].quantile(0.995).max()) + 1.0
    )
    axis.plot(
        [0, latch_threshold, latch_threshold],
        [latch_threshold, latch_threshold, 0],
        color="#A895B6", linewidth=0.9, linestyle=":", zorder=1,
    )
    for ensemble in requested:
        part = plot_df[plot_df["Ensemble"].eq(ensemble)]
        color = colors[ensemble]
        axis.scatter(
            part["IFM1"], part["IFM2"], s=13, alpha=0.38, color=color,
            edgecolor="none", rasterized=True, label=ensemble,
        )
        sns.histplot(
            part["IFM1"], bins=38, stat="density", element="step", fill=True,
            alpha=0.28, color=color, linewidth=1.0, ax=grid.ax_marg_x,
        )
        sns.histplot(
            y=part["IFM2"], bins=38, stat="density", element="step", fill=True,
            alpha=0.28, color=color, linewidth=1.0, ax=grid.ax_marg_y,
        )
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    point_colors = [
        "#F28E8E", "#F2A65A", "#E76FAD", "#7E57C2",
        "#26A69A", "#D4A017", "#5C6BC0", "#8D6E63",
    ]
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        axis.scatter(
            values["IFM1"], values["IFM2"],
            marker=markers[index % len(markers)], s=46,
            facecolor="white",
            edgecolor=point_colors[index % len(point_colors)], linewidth=1.2,
            zorder=12, label=f"Experimental | {pdb_id} | latched",
        )
    axis.text(
        1.2, 1.0, "Latch-like reference region", fontsize=8,
        color="#75657F", va="bottom",
    )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(
        "IFM1 | central IFM/QQQ residue–N1659 Cα distance (Å)", fontsize=12
    )
    axis.set_ylabel(
        "IFM2 | central IFM/QQQ residue–N1765 Cα distance (Å)", fontsize=12
    )
    figure.suptitle(
        rf"$\mathrm{{Na}}_{{\mathrm{{V}}}}1.5$ | IFM latch | {title_a} vs {title_b}",
        fontsize=15, weight="bold", x=0.08, y=0.985, ha="left",
    )
    axis.legend(
        title="Ensembles and experimental structures", loc="upper center",
        bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=True,
        fontsize=9, title_fontsize=9,
    )
    axis.grid(color="#EEE9F2", linewidth=0.45, alpha=0.6)
    sns.despine(ax=axis)
    figure.subplots_adjust(bottom=0.2, top=0.8)
    plt.show()
    return axis


def plot_missing_af2_panel(
    protocol,
    *,
    experimental_distances,
    contact_aliases,
    subset,
):
    figure, axis = plt.subplots(figsize=(7, 6))
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    colors = ["#D55E00", "#0072B2", "#009E73", "#CC79A7"]
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        axis.scatter(
            values["IFM1"], values["IFM2"],
            marker=markers[index % len(markers)], s=110,
            color=colors[index % len(colors)], edgecolor="white", linewidth=0.7,
        )
        axis.annotate(
            pdb_id, (values["IFM1"], values["IFM2"]),
            xytext=(6, 6), textcoords="offset points",
        )
    axis.set_xlabel(f"IFM1 | {contact_aliases['IFM1']['display_alias']} (Å)")
    axis.set_ylabel(f"IFM2 | {contact_aliases['IFM2']['display_alias']} (Å)")
    axis.set_title(
        rf"$\mathrm{{Na}}_{{\mathrm{{V}}}}1.5$ | WT vs IFM→QQQ | {protocol} | IFM latching | {subset}\n"
        "AF2 distributions unavailable — experimental references only"
    )
    axis.text(
        0.5, 0.08, "Set PDB_ROOT_OVERRIDES to the AF2 PDB folders and rerun",
        transform=axis.transAxes, ha="center", color="#B22222",
        bbox={"facecolor": "white", "edgecolor": "#B22222", "alpha": 0.9},
    )
    axis.grid(linestyle="--", linewidth=0.4, alpha=0.5)
    sns.despine(ax=axis)
    figure.tight_layout()
    plt.show()
    return axis


def plot_shortest_latch_contacts(
    data,
    ensemble_a,
    ensemble_b,
    *,
    experimental_distances,
    ensemble_colors,
):
    """Plot direct heavy-atom latch contacts and whole-motif distributions."""
    requested = [ensemble_a, ensemble_b]
    x, y = "Shortest central–N1659", "Shortest central–N1765"
    part = data[data["Ensemble"].isin(requested)].dropna(subset=[x, y])
    colors = {
        ensemble_a: ensemble_colors[ensemble_a],
        ensemble_b: ensemble_colors[ensemble_b],
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(11.5, 4.8), gridspec_kw={"width_ratios": [1.08, 1]}
    )
    axis = axes[0]
    for ensemble in requested:
        values = part[part["Ensemble"].eq(ensemble)]
        axis.scatter(
            values[x], values[y], s=14, alpha=0.38, color=colors[ensemble],
            edgecolor="none", rasterized=True, label=ensemble,
        )
    markers = ["o", "s", "D", "^", "v", "P", "X", "*"]
    point_colors = [
        "#F28E8E", "#F2A65A", "#E76FAD", "#7E57C2",
        "#26A69A", "#D4A017", "#5C6BC0", "#8D6E63",
    ]
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        axis.scatter(
            values[x], values[y], marker=markers[index % len(markers)], s=38,
            facecolor="white",
            edgecolor=point_colors[index % len(point_colors)], linewidth=1.1,
            zorder=10, label=f"{pdb_id} latched",
        )
    axis.set_xlabel(
        "Central IFM/QQQ residue–N1659\nshortest heavy-atom distance (Å)"
    )
    axis.set_ylabel(
        "Central IFM/QQQ residue–N1765\nshortest heavy-atom distance (Å)"
    )
    axis.grid(color="#EEE9F2", linewidth=0.45, alpha=0.65)
    long = part[
        ["Ensemble", "Shortest whole motif–N1659"]
    ].rename(columns={"Shortest whole motif–N1659": "Distance"})
    sns.violinplot(
        data=long, x="Ensemble", y="Distance", order=requested,
        palette=colors, inner="quartile", cut=0, linewidth=0.8, ax=axes[1],
    )
    experimental_offsets = np.linspace(
        -0.18, 0.18, len(experimental_distances)
    )
    for index, (_, values) in enumerate(experimental_distances.items()):
        axes[1].scatter(
            [experimental_offsets[index]],
            [values["Shortest whole motif–N1659"]],
            marker=markers[index % len(markers)], s=32, facecolor="white",
            edgecolor=point_colors[index % len(point_colors)],
            linewidth=1.0, zorder=10,
        )
    axes[1].set_xlabel("Ensemble")
    axes[1].set_ylabel(
        "Whole IFM/QQQ motif–N1659\nminimum heavy-atom distance (Å)"
    )
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", color="#EEE9F2", linewidth=0.45, alpha=0.65)
    figure.suptitle(
        r"$\mathrm{Na}_{\mathrm{V}}1.5$ | shortest IFM-latch contacts | "
        f"{ensemble_a.replace(' | ', ' ')} vs {ensemble_b.replace(' | ', ' ')}",
        x=0.06, ha="left", fontsize=14, weight="bold",
    )
    axis.legend(
        title="Ensembles and latched structures", loc="upper center",
        bbox_to_anchor=(1.03, -0.25), ncol=3, fontsize=8,
        title_fontsize=8, frameon=True,
    )
    sns.despine(fig=figure)
    figure.subplots_adjust(bottom=0.3, top=0.84, wspace=0.34)
    plt.show()
    return axes
