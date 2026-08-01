"""Matched WT RMSF comparisons across Kv2.1, Nav1.5, and Cav1.2."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import yaml

from scripts.ensemble_rmsf_analysis.io import load_primary_profile
from scripts.ensemble_rmsf_analysis.masks import parse_ranges
from scripts.ensemble_rmsf_analysis.topology import SEQUENCE_NUMBERING, TOPOLOGY
from shared.plotting import (
    CAV12_PALETTE,
    KV21_PALETTE,
    NAV15_PALETTE,
    format_channel_title,
)


CHANNEL_ORDER = ("kv21", "nav15", "cav12")
CHANNEL_NAMES = {"kv21": "Kv2.1", "nav15": "Nav1.5", "cav12": "Cav1.2"}

# These windows are a display choice, not an additional QC filter. Kv2.1 uses
# a continuous S1–S6-centered window with short flanks. Nav1.5 and Cav1.2 retain
# 30-residue flanks around their first and last mapped transmembrane segments to
# keep the full four-domain cores readable without including terminal tails.
CORE_WINDOWS = {
    "kv21": (170, 440),
    "nav15": (101, 1485),
    "cav12": (95, 1554),
}

WT_COLORS = {
    "kv21": {"vanilla": KV21_PALETTE["WT_VAN"], "masked": KV21_PALETTE["WT_HM"]},
    "nav15": {"vanilla": NAV15_PALETTE["WT_VAN"], "masked": NAV15_PALETTE["WT_HM"]},
    "cav12": {"vanilla": CAV12_PALETTE["WT_VAN"], "masked": CAV12_PALETTE["WT_HM"]},
}

MASK_SHADE = "#FFE082"
TOPOLOGY_COLORS = {
    "VSD": "#6D9EC1",
    "linker": "#9AB9CC",
    "pore": "#3F7899",
    "filter": "#23526C",
    "DI": "#5E89B2",
    "DII": "#4D8398",
    "DIII": "#6075A5",
    "DIV": "#7567A7",
    "III–IV linker": "#9875A9",
}


def _load_wt_mask(repo_root: Path, channel: str) -> set[int]:
    path = (
        repo_root
        / "scripts"
        / "ensemble_rmsf_analysis"
        / "config"
        / "generated_mask_definitions.yaml"
    )
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    specification = config["channels"][channel]["datasets"]["wt_masked"][
        "directly_masked_ranges"
    ]
    return parse_ranges(str(specification))


def load_wt_core_profiles(repo_root: str | Path) -> dict[str, pd.DataFrame]:
    """Load final allOK3 WT profiles and retain the main vanilla/masked pair."""
    root = Path(repo_root)
    result: dict[str, pd.DataFrame] = {}
    for channel in CHANNEL_ORDER:
        profiles, schema, source = load_primary_profile(root, channel)
        part = profiles.loc[
            profiles[schema["condition"]].astype(str).str.lower().eq("wt")
            & profiles[schema["protocol"]].astype(str).str.lower().isin(("vanilla", "masked"))
        ].copy()
        part = part.rename(
            columns={
                schema["residue"]: "raw_residue_number",
                schema["rmsf"]: "ensemble_rmsf_A",
                schema["protocol"]: "protocol",
            }
        )
        start, end = CORE_WINDOWS[channel]
        part = part.loc[part.raw_residue_number.between(start, end)].copy()
        if set(part.protocol.str.lower()) != {"vanilla", "masked"}:
            raise ValueError(f"{channel}: final WT vanilla/masked RMSF pair is incomplete")
        duplicate = part.duplicated(["protocol", "raw_residue_number"])
        if duplicate.any():
            raise ValueError(f"{channel}: duplicate WT RMSF residue rows in {source}")
        part["channel"] = channel
        part["source_profile"] = str(source.relative_to(root))
        part["directly_masked"] = part.raw_residue_number.astype(int).isin(
            _load_wt_mask(root, channel)
        )
        result[channel] = part[
            [
                "channel",
                "protocol",
                "raw_residue_number",
                "ensemble_rmsf_A",
                "directly_masked",
                "source_profile",
            ]
        ].sort_values(["protocol", "raw_residue_number"])
    return result


def summarize_wt_core_profiles(profiles: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize residue-wise RMSF and the paired masking effect."""
    rows = []
    for channel in CHANNEL_ORDER:
        frame = profiles[channel]
        pivot = frame.pivot(
            index="raw_residue_number", columns="protocol", values="ensemble_rmsf_A"
        )
        delta = pivot["masked"] - pivot["vanilla"]
        for protocol in ("vanilla", "masked"):
            values = pivot[protocol].dropna()
            rows.append(
                {
                    "channel": CHANNEL_NAMES[channel],
                    "channel_residue_window": (
                        f"{CORE_WINDOWS[channel][0] + SEQUENCE_NUMBERING[channel]['display_shift']}"
                        f"–{CORE_WINDOWS[channel][1] + SEQUENCE_NUMBERING[channel]['display_shift']}"
                    ),
                    "raw_model_window": f"{CORE_WINDOWS[channel][0]}–{CORE_WINDOWS[channel][1]}",
                    "protocol": protocol,
                    "residues": len(values),
                    "median_rmsf_A": values.median(),
                    "mean_rmsf_A": values.mean(),
                    "p90_rmsf_A": values.quantile(0.90),
                    "p95_rmsf_A": values.quantile(0.95),
                    "median_masked_minus_vanilla_A": delta.median(),
                    "fraction_residues_increased_by_masking": delta.gt(0).mean(),
                }
            )
    return pd.DataFrame(rows)


def _contiguous_ranges(positions: set[int]) -> list[tuple[int, int]]:
    values = sorted(positions)
    if not values:
        return []
    ranges: list[tuple[int, int]] = []
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    return ranges


def _shade_mask(ax, positions: set[int], start: int, end: int) -> None:
    local = {position for position in positions if start <= position <= end}
    for left, right in _contiguous_ranges(local):
        ax.axvspan(left - 0.5, right + 0.5, color=MASK_SHADE, alpha=0.27, lw=0, zorder=0)


def _compact_segment_label(channel: str, label: str) -> str:
    """Return a short label that remains legible inside four-domain tracks."""
    if channel == "kv21":
        return label
    if label == "IFM":
        return label
    if label.endswith("S6 core"):
        return "S6"
    if label.endswith("S6 extension"):
        return "e"
    return label.split()[-1]


def _add_topology_strip(ax, channel: str, *, y: float = -0.19) -> None:
    """Draw a labeled structural-segment track below the residue-number ticks."""
    transform = ax.get_xaxis_transform()
    start, end = CORE_WINDOWS[channel]
    shift = int(SEQUENCE_NUMBERING[channel]["display_shift"])
    height = 0.055
    visible_segments: list[tuple[dict, int, int]] = []
    for segment in TOPOLOGY[channel]:
        left = max(start, int(segment["start"])) + shift
        right = min(end, int(segment["end"])) + shift
        if right < left:
            continue
        visible_segments.append((segment, left, right))
        color = TOPOLOGY_COLORS.get(str(segment["domain"]), "#6F8FA5")
        ax.add_patch(
            Rectangle(
                (left, y),
                right - left + 1,
                height,
                transform=transform,
                facecolor=color,
                edgecolor="white",
                linewidth=0.35,
                clip_on=False,
                zorder=8,
            )
        )
        label = _compact_segment_label(channel, str(segment["label"]))
        minimum_width = 4 if channel != "kv21" else 8
        if right - left >= minimum_width:
            ax.text(
                (left + right) / 2,
                y + height / 2,
                label,
                transform=transform,
                ha="center",
                va="center",
                fontsize=7.2 if channel == "kv21" else 5.8,
                weight="semibold",
                color="white",
                clip_on=False,
                zorder=9,
            )
    if channel == "kv21":
        return

    # Domain names provide the missing DI–DIV context without repeating the
    # domain prefix inside every narrow transmembrane-segment box.
    for domain in ("DI", "DII", "DIII", "DIV"):
        members = [
            (left, right)
            for segment, left, right in visible_segments
            if str(segment["domain"]) == domain
        ]
        if not members:
            continue
        left = min(item[0] for item in members)
        right = max(item[1] for item in members)
        ax.text(
            (left + right) / 2,
            y + height + 0.018,
            domain,
            transform=transform,
            ha="center",
            va="bottom",
            fontsize=7.4,
            weight="bold",
            color=TOPOLOGY_COLORS[domain],
            clip_on=False,
            zorder=9,
        )


def plot_wt_core_profiles(
    profiles: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot matched WT vanilla/masked RMSF profiles on a shared Å scale."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14.2), sharey=True)
    upper = max(frame.ensemble_rmsf_A.quantile(0.999) for frame in profiles.values())
    upper = max(5.0, np.ceil(upper + 1.0))
    for ax, channel in zip(axes, CHANNEL_ORDER):
        frame = profiles[channel]
        start, end = CORE_WINDOWS[channel]
        shift = int(SEQUENCE_NUMBERING[channel]["display_shift"])
        display_start, display_end = start + shift, end + shift
        mask = {
            value + shift
            for value in frame.loc[frame.directly_masked, "raw_residue_number"].astype(int)
        }
        _shade_mask(ax, mask, display_start, display_end)
        for protocol in ("vanilla", "masked"):
            part = frame.loc[frame.protocol.str.lower().eq(protocol)]
            # Reindexing prevents lines and fills from bridging loop intervals
            # intentionally omitted from a segment-only channel profile.
            curve = (
                part.set_index("raw_residue_number")["ensemble_rmsf_A"]
                .reindex(range(start, end + 1))
            )
            curve.index = curve.index + shift
            color = WT_COLORS[channel][protocol]
            ax.fill_between(
                curve.index,
                0,
                curve.values,
                color=color,
                alpha=0.16 if protocol == "masked" else 0.24,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                curve.index,
                curve.values,
                color=color,
                lw=1.35,
                label=protocol.capitalize(),
                zorder=3,
            )
        _add_topology_strip(ax, channel)
        ax.set_xlim(display_start, display_end)
        ax.set_ylim(0, upper)
        ax.set_title(
            format_channel_title(f"{CHANNEL_NAMES[channel]} | WT"),
            loc="left",
            fontsize=17,
            weight="bold",
            pad=8,
        )
        ax.text(
            0.995,
            0.94,
            f"channel-core display window: {display_start}–{display_end}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#555555",
        )
        ax.set_ylabel("Ensemble RMSF (Å)", fontsize=13)
        ax.set_xlabel(
            str(SEQUENCE_NUMBERING[channel]["axis_label"]),
            fontsize=12,
            labelpad=48,
        )
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", color="#E8EDF0", linewidth=0.65)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
    legend = []
    for channel in CHANNEL_ORDER:
        for protocol in ("vanilla", "masked"):
            legend.append(
                Line2D(
                    [0],
                    [0],
                    color=WT_COLORS[channel][protocol],
                    lw=2.4,
                    label=f"{CHANNEL_NAMES[channel]} | {protocol}",
                )
            )
    legend.extend(
        [
            Patch(
                facecolor=MASK_SHADE,
                alpha=0.35,
                edgecolor="none",
                label="Directly masked residues",
            ),
            Patch(
                facecolor=TOPOLOGY_COLORS["VSD"],
                edgecolor="none",
                label="Mapped structural segments",
            ),
        ]
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.subplots_adjust(left=0.065, right=0.992, top=0.985, bottom=0.16, hspace=0.72)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_wt_core_bar_profiles(
    profiles: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot the same WT comparison as paired residue-wise RMSF bars."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 14.2), sharey=True)
    upper = max(frame.ensemble_rmsf_A.quantile(0.999) for frame in profiles.values())
    upper = max(5.0, np.ceil(upper + 1.0))

    for ax, channel in zip(axes, CHANNEL_ORDER):
        frame = profiles[channel]
        start, end = CORE_WINDOWS[channel]
        shift = int(SEQUENCE_NUMBERING[channel]["display_shift"])
        display_start, display_end = start + shift, end + shift
        mask = {
            value + shift
            for value in frame.loc[frame.directly_masked, "raw_residue_number"].astype(int)
        }
        _shade_mask(ax, mask, display_start, display_end)

        pivot = (
            frame.assign(protocol=frame.protocol.str.lower())
            .pivot(
                index="raw_residue_number",
                columns="protocol",
                values="ensemble_rmsf_A",
            )
            .reindex(range(start, end + 1))
        )
        x = pivot.index.to_numpy(dtype=float) + shift
        ax.bar(
            x - 0.22,
            pivot["vanilla"].to_numpy(dtype=float),
            width=0.42,
            color=WT_COLORS[channel]["vanilla"],
            edgecolor="none",
            alpha=0.88,
            label="Vanilla",
            zorder=2,
        )
        ax.bar(
            x + 0.22,
            pivot["masked"].to_numpy(dtype=float),
            width=0.42,
            color=WT_COLORS[channel]["masked"],
            edgecolor="none",
            alpha=0.95,
            label="Masked",
            zorder=3,
        )

        _add_topology_strip(ax, channel)
        ax.set_xlim(display_start, display_end)
        ax.set_ylim(0, upper)
        ax.set_title(
            format_channel_title(f"{CHANNEL_NAMES[channel]} | WT"),
            loc="left",
            fontsize=17,
            weight="bold",
            pad=8,
        )
        ax.text(
            0.995,
            0.94,
            f"channel-core display window: {display_start}–{display_end}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#555555",
        )
        ax.set_ylabel("Ensemble RMSF (Å)", fontsize=13)
        ax.set_xlabel(
            str(SEQUENCE_NUMBERING[channel]["axis_label"]),
            fontsize=12,
            labelpad=48,
        )
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", color="#E8EDF0", linewidth=0.65)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)

    legend = []
    for channel in CHANNEL_ORDER:
        for protocol in ("vanilla", "masked"):
            legend.append(
                Patch(
                    facecolor=WT_COLORS[channel][protocol],
                    edgecolor="none",
                    label=f"{CHANNEL_NAMES[channel]} | {protocol}",
                )
            )
    legend.extend(
        [
            Patch(
                facecolor=MASK_SHADE,
                alpha=0.35,
                edgecolor="none",
                label="Directly masked residues",
            ),
            Patch(
                facecolor=TOPOLOGY_COLORS["VSD"],
                edgecolor="none",
                label="Mapped structural segments",
            ),
        ]
    )
    fig.legend(
        handles=legend,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.subplots_adjust(left=0.065, right=0.992, top=0.985, bottom=0.16, hspace=0.76)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    return fig


def plot_wt_core_delta_profiles(
    profiles: dict[str, pd.DataFrame],
    output_path: str | Path | None = None,
) -> plt.Figure:
    """Plot paired residue-wise RMSF changes, masked minus vanilla."""
    fig, axes = plt.subplots(3, 1, figsize=(16, 13.6))
    deltas = {}
    for channel, frame in profiles.items():
        pivot = frame.pivot(
            index="raw_residue_number", columns="protocol", values="ensemble_rmsf_A"
        )
        deltas[channel] = pivot["masked"] - pivot["vanilla"]
    bound = max(np.abs(delta).quantile(0.995) for delta in deltas.values())
    bound = max(2.0, float(np.ceil(bound)))
    for ax, channel in zip(axes, CHANNEL_ORDER):
        start, end = CORE_WINDOWS[channel]
        shift = int(SEQUENCE_NUMBERING[channel]["display_shift"])
        display_start, display_end = start + shift, end + shift
        frame = profiles[channel]
        mask = {
            value + shift
            for value in frame.loc[frame.directly_masked, "raw_residue_number"].astype(int)
        }
        _shade_mask(ax, mask, display_start, display_end)
        delta = deltas[channel].reindex(range(start, end + 1))
        delta.index = delta.index + shift
        color = WT_COLORS[channel]["masked"]
        ax.fill_between(
            delta.index,
            0,
            delta.values,
            where=delta.values >= 0,
            color=color,
            alpha=0.42,
            interpolate=True,
            linewidth=0,
        )
        ax.fill_between(
            delta.index,
            0,
            delta.values,
            where=delta.values < 0,
            color=WT_COLORS[channel]["vanilla"],
            alpha=0.78,
            interpolate=True,
            linewidth=0,
        )
        ax.plot(delta.index, delta.values, color=color, lw=0.9)
        ax.axhline(0, color="#444444", lw=0.8)
        _add_topology_strip(ax, channel)
        ax.set_xlim(display_start, display_end)
        ax.set_ylim(-bound, bound)
        ax.set_title(
            format_channel_title(f"{CHANNEL_NAMES[channel]} | WT"),
            loc="left",
            fontsize=17,
            weight="bold",
            pad=8,
        )
        ax.set_ylabel("Masked − vanilla\nRMSF (Å)", fontsize=13)
        ax.set_xlabel(
            str(SEQUENCE_NUMBERING[channel]["axis_label"]),
            fontsize=12,
            labelpad=48,
        )
        ax.tick_params(labelsize=10)
        ax.grid(axis="y", color="#E8EDF0", linewidth=0.65)
        ax.grid(axis="x", visible=False)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor="#555555", alpha=0.42, edgecolor="none", label="Higher RMSF after masking"),
            Patch(facecolor="#D9D9D9", alpha=0.9, edgecolor="none", label="Lower RMSF after masking"),
            Patch(facecolor=MASK_SHADE, alpha=0.35, edgecolor="none", label="Directly masked residues"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=11,
        bbox_to_anchor=(0.5, 0.006),
    )
    fig.subplots_adjust(left=0.075, right=0.992, top=0.985, bottom=0.16, hspace=0.72)
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    return fig
