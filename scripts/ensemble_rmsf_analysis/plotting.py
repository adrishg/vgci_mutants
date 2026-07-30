"""Publication-oriented RMSF plots with explicit mask shading."""

from __future__ import annotations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle
import pandas as pd
from shared.plotting import format_channel_title

COLORS = {
    "vanilla": "#C9C5E8",
    "masked": "#514B9B",
    "delta_positive": "#6A3D9A",
    "delta_negative": "#B7A6D4",
}
MASK_SHADE = "#F0B35A"
TOPOLOGY_COLORS = {
    "VSD": "#5B8DB8", "linker": "#87AFC7", "pore": "#386F91", "filter": "#214C66",
    "DI": "#577DA6", "DII": "#477D91", "DIII": "#536A9B", "DIV": "#655B9A",
    "III–IV linker": "#8B6A9E",
}


def shade_mask(ax, positions: set[int], color=MASK_SHADE, alpha=.22):
    for position in sorted(positions):
        ax.axvspan(position - .5, position + .5, color=color, alpha=alpha, lw=0)


def add_topology_strip(ax, segments: list[dict] | None):
    """Add a compact structural-segment track along the bottom of a plot."""
    if not segments:
        return
    xmin, xmax = ax.get_xlim()
    transform = ax.get_xaxis_transform()
    for segment in segments:
        start, end = segment["start"], segment["end"]
        if end < xmin or start > xmax:
            continue
        visible_start, visible_end = max(start, xmin), min(end, xmax)
        color = TOPOLOGY_COLORS.get(segment.get("domain"), "#6485A5")
        ax.add_patch(Rectangle(
            (visible_start, .012), visible_end - visible_start + 1, .052,
            transform=transform, facecolor=color, edgecolor="white", linewidth=.35,
            alpha=.9, clip_on=False, zorder=8,
        ))
        if visible_end - visible_start >= max(3, (xmax - xmin) * .008):
            ax.text(
                (visible_start + visible_end) / 2, .038, segment["label"],
                transform=transform, ha="center", va="center", color="white",
                fontsize=6.2, weight="semibold", clip_on=False, zorder=9,
            )


def plot_whole_protein(
    comparison: pd.DataFrame, mask: set[int], title: str, output: Path,
    topology: list[dict] | None = None, colors: dict[str, str] | None = None,
):
    colors = colors or {"Vanilla": COLORS["vanilla"], "Masked": COLORS["masked"]}
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[1.15, 1])
    x = comparison.raw_residue_number
    for ax in axes:
        shade_mask(ax, mask)
    # The wider vanilla bars remain visible behind the narrower masked bars.
    axes[0].bar(
        x, comparison.vanilla_rmsf_A, width=.92, color=colors["Vanilla"],
        linewidth=0, label="Vanilla", zorder=2,
    )
    axes[0].bar(
        x, comparison.masked_rmsf_A, width=.52, color=colors["Masked"],
        linewidth=0, label="Masked", zorder=3,
    )
    axes[0].set_ylabel("Ensemble RMSF (Å)")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=MASK_SHADE, edgecolor="none", alpha=.35))
    labels.append("Directly masked residues")
    axes[0].legend(handles, labels, frameon=False, ncol=3)
    delta = comparison.masked_minus_vanilla_rmsf_A
    delta_colors = [
        colors["Masked"] if value >= 0 else colors["Vanilla"]
        for value in delta
    ]
    axes[1].bar(x, delta, width=.82, color=delta_colors, linewidth=0, zorder=2)
    axes[1].axhline(0, color=".35", lw=.8)
    axes[1].set(xlabel="Raw query residue number", ylabel="Masked − vanilla RMSF (Å)")
    for ax in axes:
        ax.grid(axis="y", alpha=.14)
    add_topology_strip(axes[0], topology)
    axes[0].set_title(format_channel_title(title), loc="left", weight="bold")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_zoom(
    comparison: pd.DataFrame, mask: set[int], start: int, end: int,
    title: str, output: Path, annotations: dict[int, str] | None = None,
    topology: list[dict] | None = None, colors: dict[str, str] | None = None,
):
    colors = colors or {"Vanilla": COLORS["vanilla"], "Masked": COLORS["masked"]}
    part = comparison.query("@start <= raw_residue_number <= @end")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    x = part.raw_residue_number
    local_mask = {position for position in mask if start <= position <= end}
    for ax in axes:
        shade_mask(ax, local_mask, alpha=.3)
    axes[0].bar(
        x - .2, part.vanilla_rmsf_A, width=.4, color=colors["Vanilla"],
        linewidth=0, label="Vanilla", zorder=2,
    )
    axes[0].bar(
        x + .2, part.masked_rmsf_A, width=.4, color=colors["Masked"],
        linewidth=0, label="Masked", zorder=2,
    )
    delta = part.masked_minus_vanilla_rmsf_A
    delta_colors = [
        colors["Masked"] if value >= 0 else colors["Vanilla"]
        for value in delta
    ]
    axes[1].bar(x, delta, width=.72, color=delta_colors, linewidth=0, zorder=2)
    axes[1].axhline(0, color=".35", lw=.8)
    if annotations:
        for position, label in annotations.items():
            axes[0].axvline(position, color="#D55E00", ls="--", lw=.9)
            axes[0].text(
                position, .075, label, rotation=90, va="bottom", ha="right",
                transform=axes[0].get_xaxis_transform(), color="#7B3D00",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=.18", "facecolor": "#FFF6E8",
                      "edgecolor": "#D55E00", "linewidth": .55, "alpha": .92},
                clip_on=False,
            )
    axes[0].set(ylabel="Ensemble RMSF (Å)", title=title)
    axes[1].set(xlabel="Raw query residue number", ylabel="Masked − vanilla (Å)")
    add_topology_strip(axes[0], topology)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=MASK_SHADE, edgecolor="none", alpha=.4))
    labels.append("Directly masked residues")
    axes[0].legend(
        handles, labels, frameon=False, ncol=1, title="Profile",
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    fig.tight_layout(rect=(0, 0, .86, .97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_whole_protein_lines(
    comparison: pd.DataFrame, mask: set[int], title: str, output: Path,
    topology: list[dict] | None = None, colors: dict[str, str] | None = None,
):
    """Line alternative with magnitude fills; fills are not uncertainty intervals."""
    colors = colors or {"Vanilla": COLORS["vanilla"], "Masked": COLORS["masked"]}
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, height_ratios=[1.15, 1])
    x = comparison.raw_residue_number.to_numpy()
    vanilla = comparison.vanilla_rmsf_A.to_numpy()
    masked = comparison.masked_rmsf_A.to_numpy()
    delta = comparison.masked_minus_vanilla_rmsf_A.to_numpy()
    for ax in axes:
        shade_mask(ax, mask)
        ax.grid(axis="y", alpha=.14)
    axes[0].fill_between(x, 0, vanilla, color=colors["Vanilla"], alpha=.34, linewidth=0)
    axes[0].plot(x, vanilla, color=colors["Vanilla"], lw=1.1, label="Vanilla")
    axes[0].fill_between(x, 0, masked, color=colors["Masked"], alpha=.18, linewidth=0)
    axes[0].plot(x, masked, color=colors["Masked"], lw=1.1, label="Masked")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=MASK_SHADE, edgecolor="none", alpha=.35))
    labels.append("Directly masked residues")
    axes[0].legend(handles, labels, frameon=False, ncol=3)
    axes[0].set(ylabel="Ensemble RMSF (Å)", title=title + " | line profile")
    add_topology_strip(axes[0], topology)
    axes[1].fill_between(
        x, 0, delta, where=delta >= 0, color=colors["Masked"], alpha=.28,
        interpolate=True, linewidth=0,
    )
    axes[1].fill_between(
        x, 0, delta, where=delta < 0, color=colors["Vanilla"], alpha=.55,
        interpolate=True, linewidth=0,
    )
    axes[1].plot(x, delta, color=colors["Masked"], lw=.85)
    axes[1].axhline(0, color=".35", lw=.8)
    axes[1].set(xlabel="Raw query residue number", ylabel="Masked − vanilla RMSF (Å)")
    fig.text(
        .995, .01, "Shaded curve area shows RMSF magnitude; it is not a confidence interval.",
        ha="right", va="bottom", fontsize=8.5, color=".38",
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_zoom_lines(
    comparison: pd.DataFrame, mask: set[int], start: int, end: int,
    title: str, output: Path, annotations: dict[int, str] | None = None,
    topology: list[dict] | None = None, colors: dict[str, str] | None = None,
):
    """Mutation-centered line alternative with RMSF-magnitude fills."""
    colors = colors or {"Vanilla": COLORS["vanilla"], "Masked": COLORS["masked"]}
    part = comparison.query("@start <= raw_residue_number <= @end")
    x = part.raw_residue_number.to_numpy()
    vanilla = part.vanilla_rmsf_A.to_numpy()
    masked = part.masked_rmsf_A.to_numpy()
    delta = part.masked_minus_vanilla_rmsf_A.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    local_mask = {position for position in mask if start <= position <= end}
    for ax in axes:
        shade_mask(ax, local_mask, alpha=.3)
        ax.grid(axis="y", alpha=.14)
    axes[0].fill_between(x, 0, vanilla, color=colors["Vanilla"], alpha=.38, linewidth=0)
    axes[0].plot(x, vanilla, color=colors["Vanilla"], lw=1.35, label="Vanilla")
    axes[0].fill_between(x, 0, masked, color=colors["Masked"], alpha=.18, linewidth=0)
    axes[0].plot(x, masked, color=colors["Masked"], lw=1.35, label="Masked")
    axes[1].fill_between(
        x, 0, delta, where=delta >= 0, color=colors["Masked"], alpha=.3,
        interpolate=True, linewidth=0,
    )
    axes[1].fill_between(
        x, 0, delta, where=delta < 0, color=colors["Vanilla"], alpha=.6,
        interpolate=True, linewidth=0,
    )
    axes[1].plot(x, delta, color=colors["Masked"], lw=1)
    axes[1].axhline(0, color=".35", lw=.8)
    if annotations:
        for position, label in annotations.items():
            axes[0].axvline(position, color="#D55E00", ls="--", lw=.9)
            axes[0].text(
                position, .075, label, rotation=90, va="bottom", ha="right",
                transform=axes[0].get_xaxis_transform(), color="#7B3D00",
                fontsize=8.5,
                bbox={"boxstyle": "round,pad=.18", "facecolor": "#FFF6E8",
                      "edgecolor": "#D55E00", "linewidth": .55, "alpha": .92},
                clip_on=False,
            )
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor=MASK_SHADE, edgecolor="none", alpha=.4))
    labels.append("Directly masked residues")
    axes[0].legend(
        handles, labels, frameon=False, ncol=1, title="Profile",
        bbox_to_anchor=(1.01, 1), loc="upper left",
    )
    axes[0].set(ylabel="Ensemble RMSF (Å)", title=title + " | line profile")
    add_topology_strip(axes[0], topology)
    axes[1].set(xlabel="Raw query residue number", ylabel="Masked − vanilla (Å)")
    fig.text(
        .86, .012, "Shaded curve area shows RMSF magnitude; it is not a confidence interval.",
        ha="right", va="bottom", fontsize=8.5, color=".38",
    )
    fig.tight_layout(rect=(0, .045, .86, .97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
