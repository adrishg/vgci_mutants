"""Reusable calculations and figures for Nav1.5 IFM-engagement analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from .plotting import experimental_reference_style
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
    engagement_threshold=12.0,
    guessed_definitions=True,
):
    """Plot Cα IFM-engagement coordinates with marginal histograms."""
    requested = [ensemble_a, ensemble_b]
    title_a, title_b = (
        ensemble_a.replace(" | ", " "),
        ensemble_b.replace(" | ", " "),
    )
    plot_df = data[data["Ensemble"].isin(requested)].dropna(
        subset=["IFM1", "IFM2"]
    )
    grid = sns.JointGrid(
        data=plot_df, x="IFM1", y="IFM2", height=7.6, ratio=4, space=0.06
    )
    figure, axis = grid.figure, grid.ax_joint
    figure.set_size_inches(12.2, 7.8)
    colors = {
        ensemble_a: ensemble_colors[ensemble_a],
        ensemble_b: ensemble_colors[ensemble_b],
    }
    limit = max(
        34.0, float(plot_df[["IFM1", "IFM2"]].quantile(0.995).max()) + 1.0
    )
    axis.plot(
        [0, engagement_threshold, engagement_threshold],
        [engagement_threshold, engagement_threshold, 0],
        color="#A895B6", linewidth=0.9, linestyle=":", zorder=1,
    )
    histogram_annotations = []
    for ensemble in requested:
        part = plot_df[plot_df["Ensemble"].eq(ensemble)]
        color = colors[ensemble]
        short_ifm1 = part["IFM1"].le(engagement_threshold)
        short_ifm2 = part["IFM2"].le(engagement_threshold)
        engaged = short_ifm1 & short_ifm2
        partially_engaged = short_ifm1 ^ short_ifm2
        disengaged = ~short_ifm1 & ~short_ifm2
        total = int(len(part))
        if total:
            percentages = [
                100.0 * int(state.sum()) / total
                for state in (engaged, partially_engaged, disengaged)
            ]
            histogram_annotations.append(
                (
                    ensemble,
                    (
                        f"{ensemble.replace(' | ', ' ')} (n={total:,}): "
                        f"engaged {percentages[0]:.1f}%  ·  "
                        f"partial {percentages[1]:.1f}%  ·  "
                        f"disengaged {percentages[2]:.1f}%"
                    ),
                )
            )
            ensemble_label = ensemble
        else:
            histogram_annotations.append(
                (ensemble, f"{ensemble.replace(' | ', ' ')}: not available")
            )
            ensemble_label = f"{ensemble} | engagement unavailable"
        axis.scatter(
            part["IFM1"], part["IFM2"], s=13, alpha=0.38, color=color,
            edgecolor="none", rasterized=True, label=ensemble_label,
        )
        sns.histplot(
            part["IFM1"], bins=38, stat="count", element="step", fill=True,
            alpha=0.28, color=color, linewidth=1.0, ax=grid.ax_marg_x,
        )
        sns.histplot(
            y=part["IFM2"], bins=38, stat="count", element="step", fill=True,
            alpha=0.28, color=color, linewidth=1.0, ax=grid.ax_marg_y,
        )
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        style = experimental_reference_style(pdb_id, index)
        axis.scatter(
            values["IFM1"], values["IFM2"],
            marker=style["marker"], s=46,
            facecolor="white",
            edgecolor=style["color"], linewidth=1.2,
            zorder=12, label=f"Experimental | {pdb_id} | WT IFM engaged",
        )
    for index, (ensemble, label) in enumerate(histogram_annotations):
        annotation_color = sns.set_hls_values(colors[ensemble], l=0.36)
        grid.ax_marg_x.text(
            0.02, 0.94 - index * 0.22, label,
            transform=grid.ax_marg_x.transAxes,
            ha="left", va="top", fontsize=8.1, color=annotation_color,
            bbox={
                "boxstyle": "round,pad=0.2",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
    axis.set_xlim(0, limit)
    axis.set_ylim(0, limit)
    grid.ax_marg_x.set_xlim(0, limit)
    grid.ax_marg_y.set_ylim(0, limit)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(
        "IFM1 | central IFM/QQQ residue–N1659 Cα distance (Å)", fontsize=12
    )
    axis.set_ylabel(
        "IFM2 | central IFM/QQQ residue–N1765 Cα distance (Å)", fontsize=12
    )
    figure.suptitle(
        rf"$\mathrm{{Na}}_{{\mathrm{{V}}}}1.5$ | IFM engagement | {title_a} vs {title_b}",
        fontsize=15, weight="bold", x=0.5, y=0.985, ha="center",
    )
    grid.ax_marg_x.set_ylabel("Model count", fontsize=8.5)
    grid.ax_marg_y.set_xlabel("Model count", fontsize=8.5)
    grid.ax_marg_x.tick_params(
        axis="y", left=True, labelleft=True, labelsize=7, length=2.5
    )
    grid.ax_marg_y.tick_params(
        axis="x", bottom=True, labelbottom=True, labelsize=7, length=2.5
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels,
        title="Ensembles and WT IFM references",
        loc="center left", bbox_to_anchor=(0.77, 0.46),
        ncol=1, frameon=True, fontsize=8.5, title_fontsize=8.5,
    )
    axis.grid(color="#EEE9F2", linewidth=0.45, alpha=0.6)
    sns.despine(ax=axis)
    figure.subplots_adjust(left=0.08, right=0.74, bottom=0.12, top=0.83)
    # Equal scaling shrinks the joint axis inside its grid cell. Re-anchor both
    # marginals to the final joint-axis bounds so histogram bins line up exactly.
    figure.canvas.draw()
    joint_bounds = axis.get_position()
    top_bounds = grid.ax_marg_x.get_position()
    right_bounds = grid.ax_marg_y.get_position()
    grid.ax_marg_x.set_position(
        [joint_bounds.x0, top_bounds.y0, joint_bounds.width, top_bounds.height]
    )
    grid.ax_marg_y.set_position(
        [right_bounds.x0, joint_bounds.y0, right_bounds.width, joint_bounds.height]
    )
    top_bounds = grid.ax_marg_x.get_position()
    right_bounds = grid.ax_marg_y.get_position()
    figure.text(
        top_bounds.x0 - 0.038,
        top_bounds.y0 + top_bounds.height / 2,
        "Model count",
        rotation=90, ha="center", va="center", fontsize=8.5,
    )
    figure.text(
        right_bounds.x0 + right_bounds.width / 2,
        right_bounds.y0 - 0.055,
        "Model count",
        ha="center", va="top", fontsize=8.5,
    )
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
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        style = experimental_reference_style(pdb_id, index)
        axis.scatter(
            values["IFM1"], values["IFM2"],
            marker=style["marker"], s=110,
            color=style["color"], edgecolor="white", linewidth=0.7,
        )
        axis.annotate(
            pdb_id, (values["IFM1"], values["IFM2"]),
            xytext=(6, 6), textcoords="offset points",
        )
    axis.set_xlabel(f"IFM1 | {contact_aliases['IFM1']['display_alias']} (Å)")
    axis.set_ylabel(f"IFM2 | {contact_aliases['IFM2']['display_alias']} (Å)")
    axis.set_title(
        rf"$\mathrm{{Na}}_{{\mathrm{{V}}}}1.5$ | WT vs IFM→QQQ | {protocol} | IFM engagement | {subset}\n"
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


def plot_shortest_engagement_contacts(
    data,
    ensemble_a,
    ensemble_b,
    *,
    experimental_distances,
    ensemble_colors,
):
    """Plot direct terminal-atom contacts with a matched summary distribution.

    When either ensemble lacks the two flanking motif contacts required for the
    whole-motif minimum, the violin falls back to the central IFM/QQQ residue
    for both ensembles. The axis label states which coordinate is displayed.
    """
    requested = [ensemble_a, ensemble_b]
    x, y = "Shortest central–N1659", "Shortest central–N1765"
    part = data[data["Ensemble"].isin(requested)].dropna(subset=[x, y])
    colors = {
        ensemble_a: ensemble_colors[ensemble_a],
        ensemble_b: ensemble_colors[ensemble_b],
    }
    figure, axes = plt.subplots(
        1, 2, figsize=(13.2, 4.9), gridspec_kw={"width_ratios": [1.08, 1]}
    )
    axis = axes[0]
    for ensemble in requested:
        values = part[part["Ensemble"].eq(ensemble)]
        axis.scatter(
            values[x], values[y], s=14, alpha=0.38, color=colors[ensemble],
            edgecolor="none", rasterized=True, label=ensemble,
        )
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        style = experimental_reference_style(pdb_id, index)
        axis.scatter(
            values[x], values[y], marker=style["marker"], s=38,
            facecolor="white",
            edgecolor=style["color"], linewidth=1.1,
            zorder=10, label=f"{pdb_id} | WT IFM reference",
        )
    axis.set_xlabel(
        "Central Phe CZ / Gln NE2–N1659 Asn ND2\n"
        "terminal side-chain distance (Å)"
    )
    axis.set_ylabel(
        "Central Phe CZ / Gln NE2–N1765 Asn ND2\n"
        "terminal side-chain distance (Å)"
    )
    axis.grid(color="#EEE9F2", linewidth=0.45, alpha=0.65)
    whole_motif_available = (
        "Whole motif available" not in part.columns
        or bool(part["Whole motif available"].all())
    )
    summary_metric = (
        "Shortest whole motif–N1659"
        if whole_motif_available
        else "Shortest central–N1659"
    )
    long = part[["Ensemble", summary_metric]].rename(
        columns={summary_metric: "Distance"}
    )
    sns.violinplot(
        data=long, x="Ensemble", y="Distance", hue="Ensemble",
        order=requested, hue_order=requested, palette=colors,
        inner="quartile", cut=0, linewidth=0.8, legend=False, ax=axes[1],
    )
    experimental_offsets = np.linspace(
        -0.18, 0.18, len(experimental_distances)
    )
    for index, (pdb_id, values) in enumerate(experimental_distances.items()):
        style = experimental_reference_style(pdb_id, index)
        axes[1].scatter(
            [experimental_offsets[index]],
            [values[summary_metric]],
            marker=style["marker"], s=32, facecolor="white",
            edgecolor=style["color"],
            linewidth=1.0, zorder=10,
        )
    axes[1].set_xlabel("Ensemble")
    axes[1].set_ylabel(
        (
            "Whole IFM/QQQ motif–N1659\nminimum terminal-atom distance (Å)"
            if whole_motif_available
            else "Central IFM/QQQ residue–N1659\nterminal-atom distance (Å)"
        )
    )
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", color="#EEE9F2", linewidth=0.45, alpha=0.65)
    figure.suptitle(
        r"$\mathrm{Na}_{\mathrm{V}}1.5$ | terminal-atom IFM engagement | "
        f"{ensemble_a.replace(' | ', ' ')} vs {ensemble_b.replace(' | ', ' ')}",
        x=0.5, ha="center", fontsize=14, weight="bold",
    )
    handles, labels = axis.get_legend_handles_labels()
    figure.legend(
        handles, labels,
        title="Ensembles and WT IFM references",
        loc="center left", bbox_to_anchor=(0.79, 0.48),
        ncol=1, fontsize=8, title_fontsize=8, frameon=True,
    )
    sns.despine(fig=figure)
    figure.subplots_adjust(left=0.07, right=0.77, bottom=0.2, top=0.82, wspace=0.34)
    plt.show()
    return axes


# Backward-compatible import for older executed notebooks. New notebook code
# uses the engagement terminology exclusively.
plot_shortest_latch_contacts = plot_shortest_engagement_contacts
