"""Reusable plotting helpers for channel-distance notebooks.

Keep channel/protocol colors and figure styling here so notebooks contain only
data loading, analysis choices, and plot execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns


STYLE_VERSION = "channel-mathtext-v6-high-contrast-ensembles"


KV21_PALETTE = {
    # WT stays sage, L403A uses a clearer leaf green, and F412L uses deep
    # blue-green. Light/dark pairs still encode vanilla/masked consistently.
    "WT_VAN": "#DDEBDD",
    "WT_HM": "#789A80",
    "L403A_VAN": "#A9D9AE",
    "L403A_HM": "#168A43",
    "F412L_VAN": "#9CCFC1",
    "F412L_HM": "#006B57",
    "experimental": "#222222",
    "shift_closer": "#F48FB1",
    "shift_farther": "#FF8A65",
}

CAV12_PALETTE = {
    "WT_VAN": "#DCEBF8",
    "WT_HM": "#3978B5",
    "G402S_VAN": "#C8DAF6",
    "G402S_HM": "#315FAF",
    "G406R_VAN": "#B8E2F0",
    "G406R_HM": "#217D9A",
    "G490R_VAN": "#D2E4F4",
    "G490R_HM": "#296A9C",
}

NAV15_PALETTE = {
    "WT_VAN": "#EBDCF2",
    "WT_HM": "#855094",
    "QQQ_VAN": "#E6D0F2",
    "QQQ_HM": "#7046A0",
    # Reserved second mutant or alternative-mask family.
    "M2_VAN": "#EBC9E3",
    "M2_HM": "#8F3F75",
    # Supplemental mask variants remain in the same family and increase in
    # saturation without changing the main WT-versus-mutant hue encoding.
    "WT_MASKED_V2": "#6F3F83",
    "WT_MASKED_V2_NOIFM": "#542D68",
    "QQQ_MASKED": "#7046A0",
    "QQQ_MASKED_V2": "#56327E",
}

ACCENT_PALETTE = {
    "RED": "#E57373", "PINK": "#F48FB1", "RASPBERRY": "#F06292",
    "CORAL": "#FF8A65", "ORANGE": "#FFB74D", "PEACH": "#FFCC99",
    "APRICOT": "#FFD89A", "YELLOW": "#FFE082", "LEMON": "#FFF176",
    "CREAM": "#FFF6B3",
}

CHANNEL_ENSEMBLE_PALETTES = {
    ("cav12", "wt"): {"Vanilla": CAV12_PALETTE["WT_VAN"], "Masked": CAV12_PALETTE["WT_HM"]},
    ("cav12", "g402s"): {"Vanilla": CAV12_PALETTE["G402S_VAN"], "Masked": CAV12_PALETTE["G402S_HM"]},
    ("cav12", "g406r"): {"Vanilla": CAV12_PALETTE["G406R_VAN"], "Masked": CAV12_PALETTE["G406R_HM"]},
    ("cav12", "g490r"): {"Vanilla": CAV12_PALETTE["G490R_VAN"], "Masked": CAV12_PALETTE["G490R_HM"]},
    ("kv21", "wt"): {"Vanilla": KV21_PALETTE["WT_VAN"], "Masked": KV21_PALETTE["WT_HM"]},
    ("kv21", "l403a"): {"Vanilla": KV21_PALETTE["L403A_VAN"], "Masked": KV21_PALETTE["L403A_HM"]},
    ("kv21", "f412l"): {"Vanilla": KV21_PALETTE["F412L_VAN"], "Masked": KV21_PALETTE["F412L_HM"]},
    ("nav15", "wt"): {"Vanilla": NAV15_PALETTE["WT_VAN"], "Masked": NAV15_PALETTE["WT_HM"]},
    ("nav15", "qqq"): {"Vanilla": NAV15_PALETTE["QQQ_VAN"], "Masked": NAV15_PALETTE["QQQ_HM"]},
}

RMSD_REFERENCE_STYLES = {
    "8SD3": {"color": ACCENT_PALETTE["RED"], "marker": "o"},
    "8SDA": {"color": ACCENT_PALETTE["PINK"], "marker": "s"},
    "9O10": {"color": ACCENT_PALETTE["RASPBERRY"], "marker": "D"},
    "9O11": {"color": ACCENT_PALETTE["CORAL"], "marker": "^"},
    "9O12": {"color": ACCENT_PALETTE["ORANGE"], "marker": "v"},
    "9O13": {"color": ACCENT_PALETTE["PEACH"], "marker": "P"},
    "8HLP": {"color": ACCENT_PALETTE["RED"], "marker": "o"},
    "8HMA": {"color": ACCENT_PALETTE["PINK"], "marker": "s"},
    "8HMB": {"color": ACCENT_PALETTE["RASPBERRY"], "marker": "D"},
    "8WEA": {"color": ACCENT_PALETTE["CORAL"], "marker": "^"},
    "8WE9": {"color": ACCENT_PALETTE["ORANGE"], "marker": "v"},
    "8WE8": {"color": ACCENT_PALETTE["PEACH"], "marker": "P"},
    "8WE7": {"color": ACCENT_PALETTE["APRICOT"], "marker": "X"},
    "8WE6": {"color": ACCENT_PALETTE["YELLOW"], "marker": "*"},
    "8FD7": {"color": ACCENT_PALETTE["LEMON"], "marker": "<"},
    "8EOG": {"color": ACCENT_PALETTE["CREAM"], "marker": ">"},
}


def ensemble_protocol_palette(channel: str, condition: str) -> dict[str, str]:
    """Return the canonical vanilla/masked colors for one channel condition."""
    key = (str(channel).lower(), str(condition).lower())
    if key not in CHANNEL_ENSEMBLE_PALETTES:
        raise KeyError(f"No canonical ensemble palette for {key}")
    return CHANNEL_ENSEMBLE_PALETTES[key].copy()

# Experimental structures must remain distinguishable in grayscale and for
# readers who cannot reliably separate nearby hues.  Color and shape therefore
# encode the same structure redundantly in every shared plotting helper.
EXPERIMENTAL_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*", "<", ">")

# Canonical NaV1.5 experimental-reference encoding.  These assignments match
# the distance-distribution figures and should also be used in RMSD, pore-shape,
# and latching panels.  Shape and color redundantly identify each structure.
NAV15_EXPERIMENTAL_STYLES = {
    "7FBS": {"color": "#D55E00", "marker": "o"},
    "6UZ3": {"color": "#0072B2", "marker": "s"},
    "8VYJ": {"color": "#009E73", "marker": "D"},
    "8VYK": {"color": "#CC79A7", "marker": "^"},
    "7DTC": {"color": "#E69F00", "marker": "v"},
    "8T6L": {"color": "#8C6D31", "marker": "P"},
}


def experimental_reference_style(structure: str, fallback_index: int = 0) -> dict[str, str]:
    """Return the project-wide color/marker assigned to an experimental PDB."""
    text = str(structure).upper()
    match = re.search(r"\b[0-9][A-Z0-9]{3}\b", text)
    pdb_id = match.group(0) if match else text.strip()
    if pdb_id in NAV15_EXPERIMENTAL_STYLES:
        return NAV15_EXPERIMENTAL_STYLES[pdb_id].copy()
    if pdb_id in RMSD_REFERENCE_STYLES:
        return RMSD_REFERENCE_STYLES[pdb_id].copy()
    return {
        "color": tuple(ACCENT_PALETTE.values())[fallback_index % len(ACCENT_PALETTE)],
        "marker": EXPERIMENTAL_MARKERS[fallback_index % len(EXPERIMENTAL_MARKERS)],
    }

KV21_STYLE = {
    # 12.8 × 7.2 inches exports at exactly 3840 × 2160 pixels with dpi=300.
    # Notebook canvas leaves a right-hand strip for legends; the data panel stays near-square.
    "figure.figsize": (10.5, 8.0),
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "savefig.bbox": None,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "axes.facecolor": "#FFFFFF",
    "axes.titlesize": 16,
    "axes.titleweight": "semibold",
    "axes.labelsize": 12,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "axes.grid": True,
    "grid.alpha": 0.75,
    "grid.color": "#EDF5EF",
    "grid.linewidth": 0.4,
    "grid.linestyle": "--",
    "lines.linewidth": 0.8,
    "patch.linewidth": 0.6,
    "font.family": "DejaVu Sans",
}

SELECTIVITY_FILTER_ALIASES = {
    "G375 A-B": "CA_CA_A_GLY377_CA-B_GLY377_CA",
    "G375 A-C": "CA_CA_A_GLY377_CA-C_GLY377_CA",
    "G375 A-D": "CA_CA_A_GLY377_CA-D_GLY377_CA",
    "G375 B-C": "CA_CA_B_GLY377_CA-C_GLY377_CA",
    "G375 B-D": "CA_CA_B_GLY377_CA-D_GLY377_CA",
    "G375 C-D": "CA_CA_C_GLY377_CA-D_GLY377_CA",
}

GATE_ALIASES = {
    "A402 A-B": "CA_CA_A_ALA404_CA-B_ALA404_CA",
    "A402 A-C": "CA_CA_A_ALA404_CA-C_ALA404_CA",
    "A402 A-D": "CA_CA_A_ALA404_CA-D_ALA404_CA",
    "A402 B-C": "CA_CA_B_ALA404_CA-C_ALA404_CA",
    "A402 B-D": "CA_CA_B_ALA404_CA-D_ALA404_CA",
    "A402 C-D": "CA_CA_C_ALA404_CA-D_ALA404_CA",
}

# S6 pore-axis profile.  For each model, the maximum of all six inter-subunit
# Cα distances is used as a chain-order-independent cross-pore diameter proxy.
# Kv2.1 model/CSV numbering is +2 relative to rat structures 8SD3/8SDA:
# model L405 == experimental L403 and model F414 == experimental F412.
S6_RING_RESIDUES = {
    "V398": (("VAL",), 400),
    "I401": (("ILE",), 403),
    "A402": (("ALA",), 404),
    "L403/A403": (("LEU", "ALA"), 405),
    "I405": (("ILE",), 407),
    "V409": (("VAL",), 411),
}


def add_s6_cross_pore_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add per-model S6 cross-pore diameter proxies from six Cα ring distances."""
    result = df.copy()
    aliases = {}
    chain_pairs = ("A-B", "A-C", "A-D", "B-C", "B-D", "C-D")
    for label, (residue_names, residue_number) in S6_RING_RESIDUES.items():
        columns = []
        for pair in chain_pairs:
            candidates = [
                f"CA_CA_{pair[0]}_{resname}{residue_number}_CA-{pair[2]}_{resname}{residue_number}_CA"
                for resname in residue_names
            ]
            matching = [column for column in candidates if column in result.columns]
            if len(matching) != 1:
                raise KeyError(f"Expected one S6 ring column for {label} {pair}; found {matching}")
            columns.append(matching[0])
        output_column = f"S6_cross_pore_max_{residue_number}"
        result[output_column] = result[columns].apply(pd.to_numeric, errors="coerce").max(axis=1)
        aliases[label] = output_column
    return result, aliases


VSD_CHAIN_A_ALIASES = {
    "F236-R289 A": "CA_CA_A_ARG291_CA-A_PHE238_CA",
    "F236-R308 A": "CA_CA_A_ARG310_CA-A_PHE238_CA",
}

COMPARISON_REGIONS = {
    "intracellular gate (A402 Cα)": GATE_ALIASES,
    "selectivity filter (G375 Cα)": SELECTIVITY_FILTER_ALIASES,
    "voltage sensor chain A (F236-R289/R308 Cα)": VSD_CHAIN_A_ALIASES,
}


def apply_kv21_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", rc=KV21_STYLE)


def format_channel_title(title: object) -> str:
    """Apply concise publication-facing channel notation to plot titles."""
    text = str(title)
    # These describe rendering or data provenance and belong in the caption or
    # notebook text, not in the visible title.
    text = re.sub(
        r"\s*\|\s*(?:half-half|split violin|allOk3\s*\+\s*structural QC|"
        r"experimental distances)\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\s*;\s*experimental numbering(?=\))",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*\|\s*\|", " | ", text)
    text = re.sub(r"\s{2,}", " ", text).strip(" |")
    replacements = {
        "Kv2.1": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        "KV2.1": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        "Nav1.5": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        "NAV1.5": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        "Cav1.2": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
        "CAV1.2": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
        r"K$_{\mathrm{V}}$2.1": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        r"Na$_{\mathrm{V}}$1.5": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        r"Ca$_{\mathrm{V}}$1.2": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
    }
    for source, replacement in replacements.items():
        text = text.replace(source, replacement)
    return text


def channel_math_label(channel: object) -> str:
    """Return the canonical visible channel name with a capital subscript V."""
    key = re.sub(r"[^a-z0-9]", "", str(channel).lower())
    labels = {
        "kv21": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        "nav15": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        "cav12": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
    }
    if key not in labels:
        raise KeyError(f"Unknown channel label: {channel!r}")
    return labels[key]


def save_figure_4k(fig, path, *, transparent: bool = False) -> None:
    """Export a 16:9 figure at 3840 × 2160 pixels without changing notebook display."""
    original_size = fig.get_size_inches().copy()
    fig.set_size_inches(12.8, 7.2)
    fig.savefig(path, dpi=300, bbox_inches=None, facecolor="none" if transparent else "white",
                transparent=transparent)
    fig.set_size_inches(original_size)


def finish_axes(ax, title: str, ylabel: str = "Cα distance (Å)") -> None:
    ax.set_title(format_channel_title(title), pad=14)
    ax.set_xlabel("Residue-pair alias")
    ax.set_ylabel(ylabel)
    number_offset = -2 if re.search(r"Kv?2\.1", str(title), re.I) else 0
    # Preserve the underlying categorical keys used for experimental overlays;
    # only the visible tick text is compacted.
    ax.set_xticklabels([
        format_distance_alias(label.get_text(), number_offset)
        for label in ax.get_xticklabels()
    ])
    ax.tick_params(axis="x", rotation=40)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    ax.grid(axis="x", visible=False)
    ax.set_box_aspect(0.88)
    for spine in ax.spines.values():
        spine.set_color("#DDE9E0")
        spine.set_linewidth(0.6)
    # Seaborn's inner quartile marks default to gray; use a softer green-charcoal instead.
    for line in ax.lines:
        line.set_color("#52685A")
        line.set_alpha(0.72)
        line.set_linewidth(0.65)
    sns.despine(ax=ax)


def finish_figure(fig, *, legend_addon: bool = False) -> None:
    """Keep the data panel fixed and reserve optional canvas space for an outside legend."""
    right = 0.76 if legend_addon else 0.96
    fig.subplots_adjust(left=0.11, right=right, bottom=0.20, top=0.86)


def available_aliases(df: pd.DataFrame, aliases: Mapping[str, str]) -> dict[str, str]:
    """Return display aliases whose source columns exist in a dataframe."""
    return {label: column for label, column in aliases.items() if column in df.columns}


def _tint_violin_borders(ax, darken: float = 0.72) -> None:
    """Replace Seaborn's gray violin borders with a darker tint of each fill color."""
    for collection in ax.collections:
        if not isinstance(collection, PolyCollection):
            continue
        facecolors = collection.get_facecolors()
        if not len(facecolors):
            continue
        red, green, blue, alpha = facecolors[0]
        collection.set_edgecolor((red * darken, green * darken, blue * darken, alpha))
        collection.set_linewidth(0.65)


def shared_aliases(
    df1: pd.DataFrame, df2: pd.DataFrame, aliases: Mapping[str, str]
) -> dict[str, str]:
    """Return aliases available in both dataframes."""
    return {
        label: column for label, column in aliases.items()
        if column in df1.columns and column in df2.columns
    }


def _long_distances(df: pd.DataFrame, aliases: Mapping[str, str]) -> pd.DataFrame:
    records = [
        {"Alias": alias, "Distance": value}
        for alias, column in available_aliases(df, aliases).items()
        for value in pd.to_numeric(df[column], errors="coerce").dropna()
    ]
    return pd.DataFrame(records)


def plot_distances_by_alias_violin(
    df: pd.DataFrame,
    alias_dict: Mapping[str, str],
    exp_distances_list: Sequence[Mapping[str, Sequence[float]]] | None = None,
    title_custom_add: str = "",
    colors: Sequence[str] | None = None,
    fig_width: float = 10.5,
    fig_height: float = 8.0,
    dataset_labels: Sequence[str] | None = None,
    pdb_colors: Sequence[str] | None = None,
):
    """Plot one ensemble and optionally overlay experimental distances."""
    aliases = available_aliases(df, alias_dict)
    missing = [column for column in alias_dict.values() if column not in df.columns]
    if missing:
        print(f"Skipped {len(missing)} unavailable distance columns.")
    plot_df = _long_distances(df, aliases)
    if plot_df.empty:
        print("No numeric distances were available to plot.")
        return None
    plot_df["Alias"] = pd.Categorical(plot_df["Alias"], list(aliases), ordered=True)
    palette = list(colors) if colors and len(colors) == len(aliases) else None
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.violinplot(
        data=plot_df, x="Alias", y="Distance", order=list(aliases), palette=palette,
        color=None if palette else KV21_PALETTE["WT_HM"], inner="quartile", cut=0,
        linewidth=0.6, saturation=0.82, ax=ax,
    )
    _tint_violin_borders(ax)
    if exp_distances_list and dataset_labels:
        marker_styles = EXPERIMENTAL_MARKERS
        point_colors = list(pdb_colors) if pdb_colors and len(pdb_colors) == len(exp_distances_list) else [KV21_PALETTE["experimental"]] * len(exp_distances_list)
        used = [False] * len(exp_distances_list)
        for i, distances in enumerate(exp_distances_list):
            offset = (i - (len(exp_distances_list) - 1) / 2) * 0.07
            for alias, values in distances.items():
                if alias not in aliases:
                    continue
                xpos = list(aliases).index(alias) + offset
                for value in values:
                    ax.scatter(xpos, value, facecolors="white", edgecolors=point_colors[i],
                               marker=marker_styles[i % len(marker_styles)], s=34, linewidths=0.8, zorder=6,
                               label=(dataset_labels[i] if str(dataset_labels[i]).startswith("Experimental |") else "Experimental | " + str(dataset_labels[i])) if not used[i] else None)
                    used[i] = True
    has_legend = bool(ax.get_legend_handles_labels()[1])
    if has_legend:
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, title="Experimental structures", loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=min(4, len(labels)), fontsize=9, title_fontsize=9, frameon=True)
        fig.subplots_adjust(bottom=0.16)
    finish_axes(ax, title_custom_add.lstrip("— "))
    finish_figure(fig, legend_addon=has_legend)
    plt.show()
    return ax


def plot_split_violin(
    df: pd.DataFrame, alias_dict: Mapping[str, tuple[str, str]], title_custom_add: str = "",
    fig_width: float = 10.5, fig_height: float = 8.0, colors: Sequence[str] | None = None,
    split_labels: tuple[str, str] = ("First distance", "Second distance"),
    legend_title: str = "Distance",
):
    records = []
    label_a, label_b = split_labels
    for alias, (column_a, column_b) in alias_dict.items():
        for dataset, column in ((label_a, column_a), (label_b, column_b)):
            if column in df.columns:
                records.extend({"Alias": alias, "Distance": value, "Dataset": dataset}
                               for value in pd.to_numeric(df[column], errors="coerce").dropna())
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("No numeric distances were available to plot.")
        return None
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.violinplot(data=plot_df, x="Alias", y="Distance", hue="Dataset",
                   order=list(alias_dict), hue_order=[label_a, label_b], split=True,
                   inner="quartile", cut=0, linewidth=0.6, palette=colors, ax=ax)
    _tint_violin_borders(ax)
    finish_axes(ax, title_custom_add.lstrip("— "))
    ax.legend(title=legend_title, bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_figure(fig, legend_addon=True); plt.show()
    return ax


def plot_protocol_split_with_experimentals(
    df_vanilla: pd.DataFrame,
    df_masked: pd.DataFrame,
    aliases: Mapping[str, str],
    title: str,
    colors: Sequence[str],
    exp_distances_list: Sequence[Mapping[str, Sequence[float]]] | None = None,
    dataset_labels: Sequence[str] | None = None,
    pdb_colors: Sequence[str] | None = None,
    fig_width: float = 10.5,
    fig_height: float = 8.0,
):
    """Plot vanilla/masked split violins with optional experimental markers."""
    aliases = shared_aliases(df_vanilla, df_masked, aliases)
    records = []
    for protocol, frame in (("vanilla", df_vanilla), ("masked", df_masked)):
        for alias, column in aliases.items():
            records.extend(
                {"Distance alias": alias, "Distance (Å)": value, "Protocol": protocol}
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("No shared numeric distances were available for the protocol split plot.")
        return None
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.violinplot(
        data=plot_df, x="Distance alias", y="Distance (Å)", hue="Protocol",
        order=list(aliases), hue_order=["vanilla", "masked"], split=True,
        inner="quartile", cut=0, linewidth=0.6,
        palette={"vanilla": colors[0], "masked": colors[1]},
        saturation=0.82, ax=ax,
    )
    _tint_violin_borders(ax)
    if exp_distances_list and dataset_labels:
        point_colors = (
            list(pdb_colors)
            if pdb_colors and len(pdb_colors) == len(exp_distances_list)
            else [KV21_PALETTE["experimental"]] * len(exp_distances_list)
        )
        used = [False] * len(exp_distances_list)
        for index, distances in enumerate(exp_distances_list):
            offset = (index - (len(exp_distances_list) - 1) / 2) * 0.07
            for alias, values in distances.items():
                if alias not in aliases:
                    continue
                xpos = list(aliases).index(alias) + offset
                for value in values:
                    label = str(dataset_labels[index])
                    if not label.startswith("Experimental |"):
                        label = "Experimental | " + label
                    ax.scatter(
                        xpos, value, facecolors="white", edgecolors=point_colors[index],
                        marker=EXPERIMENTAL_MARKERS[index % len(EXPERIMENTAL_MARKERS)],
                        s=34, linewidths=0.8, zorder=6,
                        label=label if not used[index] else None,
                    )
                    used[index] = True
    finish_axes(ax, title)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.legend(
        handles, labels, title="Protocols and experimental structures",
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(labels)), fontsize=9, title_fontsize=9, frameon=True,
    )
    finish_figure(fig, legend_addon=False)
    fig.subplots_adjust(bottom=0.32)
    plt.show()
    return ax


def plot_single_ensemble(
    df: pd.DataFrame, aliases: Mapping[str, str], label: str, color: str, title: str,
):
    aliases = available_aliases(df, aliases)
    plot_df = _long_distances(df, aliases).rename(columns={"Alias": "Landmark"})
    fig, ax = plt.subplots(figsize=KV21_STYLE["figure.figsize"])
    sns.violinplot(data=plot_df, x="Landmark", y="Distance", order=list(aliases),
                   color=color, inner="quartile", cut=0, linewidth=0.6,
                   saturation=0.82, ax=ax)
    _tint_violin_borders(ax)
    finish_axes(ax, title)
    finish_figure(fig); plt.show()
    return ax


def plot_transparent_overlay(
    df_wt: pd.DataFrame, df_mutant: pd.DataFrame, aliases: Mapping[str, str],
    mutant_label: str, protocol: str, region_label: str, alpha: float = 0.42,
):
    aliases = shared_aliases(df_wt, df_mutant, aliases)
    records = []
    for dataset, frame in (("WT", df_wt), (mutant_label, df_mutant)):
        for alias, column in aliases.items():
            records.extend({"Landmark": alias, "Distance": value, "Dataset": dataset}
                           for value in pd.to_numeric(frame[column], errors="coerce").dropna())
    suffix = "VAN" if protocol == "vanilla" else "HM"
    palette = {
        "WT": KV21_PALETTE[f"WT_{suffix}"],
        mutant_label: KV21_PALETTE[f"{mutant_label}_{suffix}"],
    }
    plot_df = pd.DataFrame(records)
    fig, ax = plt.subplots(figsize=KV21_STYLE["figure.figsize"])
    sns.violinplot(data=plot_df, x="Landmark", y="Distance", hue="Dataset",
                   order=list(aliases), hue_order=["WT", mutant_label], dodge=False,
                   palette=palette, inner="quartile", cut=0, linewidth=0.6, ax=ax)
    _tint_violin_borders(ax)
    for artist in ax.collections:
        if isinstance(artist, PolyCollection):
            artist.set_alpha(alpha)
    finish_axes(ax, f"Kv2.1 | WT vs {mutant_label} | {protocol} | {region_label}")
    ax.legend(title="Ensemble", bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_figure(fig, legend_addon=True); plt.show()
    return ax


def plot_distances_by_alias_violin_overlay(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    alias_dict: Mapping[str, str],
    df1_label: str = "Dataset 1",
    df2_label: str = "Dataset 2",
    exp_distances_list=None,
    dataset_labels=None,
    pdb_colors=None,
    title_custom_add: str = "",
    colors=None,
    alpha: float = 0.45,
    fig_width: float = 10.5,
    fig_height: float = 8.0,
    linewidth: float = 0.8,
):
    """Compatibility overlay used by the Cav1.2 and Nav1.5 notebooks."""
    aliases = shared_aliases(df1, df2, alias_dict)
    records = []
    for label, frame in ((df1_label, df1), (df2_label, df2)):
        for alias, column in aliases.items():
            records.extend(
                {"Alias": alias, "Distance": value, "Dataset": label}
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("No shared numeric distances were available to plot.")
        return None
    palette = (
        {df1_label: colors[0], df2_label: colors[1]}
        if colors is not None and len(colors) == 2
        else dict(zip((df1_label, df2_label), sns.color_palette("Set2", 2)))
    )
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.violinplot(
        data=plot_df, x="Alias", y="Distance", hue="Dataset",
        order=list(aliases), palette=palette, dodge=False, cut=0,
        inner="quartile", linewidth=linewidth, ax=ax,
    )
    _tint_violin_borders(ax)
    for collection in ax.collections:
        if isinstance(collection, PolyCollection):
            collection.set_alpha(alpha)
    if exp_distances_list and dataset_labels:
        point_colors = (
            list(pdb_colors)
            if pdb_colors and len(pdb_colors) == len(exp_distances_list)
            else [KV21_PALETTE["experimental"]] * len(exp_distances_list)
        )
        used = [False] * len(exp_distances_list)
        for index, distances in enumerate(exp_distances_list):
            offset = (index - (len(exp_distances_list) - 1) / 2) * 0.07
            for alias, values in distances.items():
                if alias not in aliases:
                    continue
                for value in values:
                    ax.scatter(
                        list(aliases).index(alias) + offset, value,
                        facecolors="white", edgecolors=point_colors[index],
                        marker=EXPERIMENTAL_MARKERS[index % len(EXPERIMENTAL_MARKERS)],
                        s=34, linewidths=0.8, zorder=6,
                        label=dataset_labels[index] if not used[index] else None,
                    )
                    used[index] = True
    finish_axes(ax, title_custom_add)
    handles, labels = ax.get_legend_handles_labels()
    if labels:
        ax.legend(handles, labels, bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_figure(fig, legend_addon=bool(labels))
    plt.show()
    return ax


def plot_pairwise_split_violin(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    aliases: Mapping[str, str],
    label1: str,
    label2: str,
    title: str,
    colors: Sequence[str],
):
    """Generic two-ensemble split violin for channel-specific aliases."""
    shared = shared_aliases(df1, df2, aliases)
    records = []
    for label, frame in ((label1, df1), (label2, df2)):
        for alias, column in shared.items():
            records.extend(
                {"Alias": alias, "Distance": value, "Dataset": label}
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    fig, ax = plt.subplots(figsize=KV21_STYLE["figure.figsize"])
    sns.violinplot(
        data=pd.DataFrame(records), x="Alias", y="Distance", hue="Dataset",
        order=list(shared), hue_order=[label1, label2], split=True,
        inner="quartile", cut=0, linewidth=0.6,
        palette={label1: colors[0], label2: colors[1]}, ax=ax,
    )
    _tint_violin_borders(ax)
    finish_axes(ax, title)
    ax.legend(title="Ensemble", bbox_to_anchor=(1.02, 1), loc="upper left")
    finish_figure(fig, legend_addon=True)
    plt.show()
    return ax


def plot_split_comparison_with_experimentals(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    aliases: Mapping[str, str],
    label1: str,
    label2: str,
    title: str,
    colors: Sequence[str],
    experimental_rows: Sequence[Mapping] | None = None,
):
    """Generic split violin with tidy experimental marker rows."""
    shared = shared_aliases(df1, df2, aliases)
    records = []
    for label, frame in ((label1, df1), (label2, df2)):
        for alias, column in shared.items():
            records.extend(
                {"Alias": alias, "Distance": value, "Dataset": label}
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print(f"Skipped empty split comparison: {title}")
        return None
    figure, axis = plt.subplots(figsize=KV21_STYLE["figure.figsize"])
    sns.violinplot(
        data=plot_df, x="Alias", y="Distance", hue="Dataset",
        order=list(shared), hue_order=[label1, label2], split=True,
        inner="quartile", cut=0, linewidth=0.6,
        palette={label1: colors[0], label2: colors[1]}, ax=axis,
    )
    _tint_violin_borders(axis)
    used_structures = set()
    structure_order = list(
        dict.fromkeys(row["Structure"] for row in (experimental_rows or []))
    )
    for row in experimental_rows or []:
        alias = row["Alias"]
        if alias not in shared:
            continue
        structure = row["Structure"]
        index = structure_order.index(structure)
        style = experimental_reference_style(structure, index)
        axis.scatter(
            list(shared).index(alias),
            row["Distance"],
            marker=style["marker"],
            s=30, facecolors="white", edgecolors=style["color"],
            linewidths=0.8, zorder=7,
            label=structure if structure not in used_structures else None,
        )
        used_structures.add(structure)
    finish_axes(axis, title)
    handles, labels = axis.get_legend_handles_labels()
    axis.get_legend().remove()
    figure.legend(
        handles, labels, title="Ensembles and experimental structures",
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(labels)), fontsize=8, title_fontsize=8, frameon=True,
    )
    finish_figure(figure)
    figure.subplots_adjust(bottom=0.32)
    plt.show()
    return axis


def plot_split_ensemble_with_experimentals(
    df_wt: pd.DataFrame,
    df_mutant: pd.DataFrame,
    aliases: Mapping[str, str],
    mutant_label: str,
    protocol: str,
    region_label: str,
    exp_distances_list: Sequence[Mapping[str, Sequence[float]]] | None = None,
    dataset_labels: Sequence[str] | None = None,
    pdb_colors: Sequence[str] | None = None,
):
    """Plot WT/mutant split violins with coordinate-derived experimental markers."""
    aliases = shared_aliases(df_wt, df_mutant, aliases)
    records = []
    for dataset, frame in (("WT", df_wt), (mutant_label, df_mutant)):
        for alias, column in aliases.items():
            records.extend(
                {"Landmark": alias, "Distance": value, "Dataset": dataset}
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    plot_df = pd.DataFrame(records)
    suffix = "VAN" if protocol == "vanilla" else "HM"
    palette = {
        "WT": KV21_PALETTE[f"WT_{suffix}"],
        mutant_label: KV21_PALETTE[f"{mutant_label}_{suffix}"],
    }
    fig, ax = plt.subplots(figsize=KV21_STYLE["figure.figsize"])
    sns.violinplot(
        data=plot_df, x="Landmark", y="Distance", hue="Dataset",
        order=list(aliases), hue_order=["WT", mutant_label], split=True,
        palette=palette, inner="quartile", cut=0, linewidth=0.6,
        saturation=0.82, ax=ax,
    )
    _tint_violin_borders(ax)

    if exp_distances_list and dataset_labels:
        point_colors = (
            list(pdb_colors)
            if pdb_colors and len(pdb_colors) == len(exp_distances_list)
            else [KV21_PALETTE["experimental"]] * len(exp_distances_list)
        )
        used = [False] * len(exp_distances_list)
        for i, distances in enumerate(exp_distances_list):
            for alias, values in distances.items():
                if alias not in aliases:
                    continue
                xpos = list(aliases).index(alias)
                for value in values:
                    label = dataset_labels[i]
                    if not str(label).startswith("Experimental |"):
                        label = "Experimental | " + str(label)
                    ax.scatter(
                        xpos, value, facecolors="white", edgecolors=point_colors[i],
                        marker=EXPERIMENTAL_MARKERS[i % len(EXPERIMENTAL_MARKERS)],
                        s=34, linewidths=0.8, zorder=6,
                        label=label if not used[i] else None,
                    )
                    used[i] = True

    finish_axes(
        ax,
        f"Kv2.1 | WT vs {mutant_label} | {protocol} | {region_label} | experimental distances",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Ensembles and experimental structures",
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(labels)), fontsize=9, title_fontsize=9, frameon=True,
    )
    ax.get_legend().remove()
    finish_figure(fig, legend_addon=False)
    fig.subplots_adjust(bottom=0.32)
    plt.show()
    return ax


def plot_l403a_chain_c_interface_figure(
    vanilla: pd.DataFrame,
    masked: pd.DataFrame,
    aliases: Mapping[str, str],
    experimental_wt: Mapping[str, Sequence[float]],
    experimental_l403a: Mapping[str, Sequence[float]],
):
    """Publication-focused L403A pore–VSD figure without pooling subunits.

    The left panel resolves the six chain-C interface coordinates.  The right
    panel places the focal E423–N179 result in its A–D context using ensemble
    medians.  Keeping chains separate avoids pseudo-replication and makes the
    asymmetric 8SDA displacement explicit.
    """
    chain_c_order = [
        f"{pore}C-{vsd}C"
        for pore in ("K427", "E423", "K420")
        for vsd in ("N179", "V182")
    ]
    chain_c = {
        alias: aliases[alias]
        for alias in chain_c_order
        if alias in aliases
    }
    if not chain_c:
        raise KeyError("No chain-C pore–VSD aliases are available")

    records = []
    for protocol, frame in (("vanilla", vanilla), ("masked", masked)):
        for alias, column in chain_c.items():
            records.extend(
                {
                    "Distance": value,
                    "Alias": alias.replace("C-", "–").removesuffix("C"),
                    "Protocol": protocol,
                }
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    display_order = [
        alias.replace("C-", "–").removesuffix("C") for alias in chain_c
    ]

    figure, (main_axis, context_axis) = plt.subplots(
        1, 2, figsize=(12.8, 6.9),
        gridspec_kw={"width_ratios": (2.25, 1), "wspace": 0.28},
    )
    sns.violinplot(
        data=pd.DataFrame(records), x="Alias", y="Distance",
        hue="Protocol", order=display_order,
        hue_order=["vanilla", "masked"], split=True, inner="quartile",
        cut=0, linewidth=0.65,
        palette={
            "vanilla": KV21_PALETTE["L403A_VAN"],
            "masked": KV21_PALETTE["L403A_HM"],
        },
        ax=main_axis,
    )
    _tint_violin_borders(main_axis)
    for index, alias in enumerate(chain_c):
        for values, color, marker in (
            (experimental_wt.get(alias, []), "#D55E00", "o"),
            (experimental_l403a.get(alias, []), "#0072B2", "s"),
        ):
            for value in values:
                main_axis.scatter(
                    index, value, s=31, marker=marker, facecolors="white",
                    edgecolors=color, linewidths=0.9, zorder=8,
                )
    main_axis.set(
        title="Chain C pore–VSD interface",
        xlabel="Residue pair (experimental numbering)",
        ylabel="Cα distance (Å)",
    )
    main_axis.tick_params(axis="x", rotation=28)
    if main_axis.get_legend() is not None:
        main_axis.get_legend().remove()

    focal_aliases = [f"E423{chain}-N179{chain}" for chain in "ABCD"]
    context_rows = []
    for alias in focal_aliases:
        if alias not in aliases:
            continue
        column = aliases[alias]
        chain = alias[4]
        for protocol, frame in (("vanilla", vanilla), ("masked", masked)):
            context_rows.extend(
                {
                    "Chain": chain,
                    "Protocol": protocol,
                    "Distance": value,
                }
                for value in pd.to_numeric(
                    frame[column], errors="coerce"
                ).dropna()
            )
    context = pd.DataFrame(context_rows)
    chains = [chain for chain in "ABCD" if chain in set(context["Chain"])]
    sns.violinplot(
        data=context, x="Chain", y="Distance", hue="Protocol",
        order=chains, hue_order=["vanilla", "masked"], split=True,
        inner="quartile", cut=0, linewidth=0.65,
        palette={
            "vanilla": KV21_PALETTE["L403A_VAN"],
            "masked": KV21_PALETTE["L403A_HM"],
        },
        ax=context_axis,
    )
    _tint_violin_borders(context_axis)
    for chain_index, chain in enumerate(chains):
        alias = f"E423{chain}-N179{chain}"
        for values, color, marker in (
            (experimental_wt.get(alias, []), "#D55E00", "o"),
            (experimental_l403a.get(alias, []), "#0072B2", "s"),
        ):
            for value in values:
                context_axis.scatter(
                    chain_index, value, s=31, marker=marker,
                    facecolors="white", edgecolors=color,
                    linewidths=0.9, zorder=8,
                )
    context_axis.set(
        title="E423–N179 across chains",
        xlabel="Subunit chain", ylabel="Cα distance (Å)",
    )
    if context_axis.get_legend() is not None:
        context_axis.get_legend().remove()
    for axis in (main_axis, context_axis):
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)

    legend_handles = [
        Patch(facecolor=KV21_PALETTE["L403A_VAN"], edgecolor="#78977F",
              label="L403A | vanilla"),
        Patch(facecolor=KV21_PALETTE["L403A_HM"], edgecolor="#356D46",
              label="L403A | masked"),
        Line2D([0], [0], color="none", marker="o", markersize=6,
               markerfacecolor="white", markeredgecolor="#D55E00",
               label="Experimental | 8SD3 WT"),
        Line2D([0], [0], color="none", marker="s", markersize=6,
               markerfacecolor="white", markeredgecolor="#0072B2",
               label="Experimental | 8SDA L403A"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=4, title="L403A predictions and experimental references",
        frameon=True,
    )
    figure.suptitle(
        r"$\mathrm{K}_{\mathrm{V}}2.1$ | L403A | masking shifts the chain-C pore–VSD interface toward 8SDA",
        fontsize=17, fontweight="semibold", y=0.985,
    )
    figure.subplots_adjust(bottom=0.22, top=0.86)
    plt.show()
    return main_axis, context_axis


def plot_l403a_e423_n179_figure(
    vanilla: pd.DataFrame,
    masked: pd.DataFrame,
    aliases: Mapping[str, str],
    experimental_wt: Mapping[str, Sequence[float]],
    experimental_l403a: Mapping[str, Sequence[float]],
):
    """Show full L403A E423–N179 distributions separately for chains A–D."""
    focal_aliases = [f"E423{chain}-N179{chain}" for chain in "ABCD"]
    records = []
    available_chains = []
    for alias in focal_aliases:
        if alias not in aliases:
            continue
        chain = alias[4]
        available_chains.append(chain)
        column = aliases[alias]
        for protocol, frame in (("vanilla", vanilla), ("masked", masked)):
            records.extend(
                {
                    "Chain": chain,
                    "Protocol": protocol,
                    "Distance": value,
                }
                for value in pd.to_numeric(
                    frame[column], errors="coerce"
                ).dropna()
            )
    if not records:
        raise KeyError("No E423–N179 chain-resolved distances are available")

    figure, axis = plt.subplots(figsize=(8.4, 6.6))
    sns.violinplot(
        data=pd.DataFrame(records), x="Chain", y="Distance",
        hue="Protocol", order=available_chains,
        hue_order=["vanilla", "masked"], split=True,
        inner="quartile", cut=0, linewidth=0.7,
        palette={
            "vanilla": KV21_PALETTE["L403A_VAN"],
            "masked": KV21_PALETTE["L403A_HM"],
        },
        ax=axis,
    )
    _tint_violin_borders(axis)
    for chain_index, chain in enumerate(available_chains):
        alias = f"E423{chain}-N179{chain}"
        for values, color, marker in (
            (experimental_wt.get(alias, []), "#D55E00", "o"),
            (experimental_l403a.get(alias, []), "#0072B2", "s"),
        ):
            for value in values:
                axis.scatter(
                    chain_index, value, s=34, marker=marker,
                    facecolors="white", edgecolors=color,
                    linewidths=0.9, zorder=8,
                )
    if axis.get_legend() is not None:
        axis.get_legend().remove()
    axis.set(
        title=r"$\mathrm{K}_{\mathrm{V}}2.1$ | L403A | E423–N179 pore–VSD distance",
        xlabel="Subunit chain",
        ylabel="Cα distance (Å)",
    )
    axis.grid(axis="x", visible=False)
    sns.despine(ax=axis)
    figure.legend(
        handles=[
            Patch(
                facecolor=KV21_PALETTE["L403A_VAN"],
                edgecolor="#78977F", label="L403A | vanilla",
            ),
            Patch(
                facecolor=KV21_PALETTE["L403A_HM"],
                edgecolor="#356D46", label="L403A | masked",
            ),
            Line2D(
                [0], [0], color="none", marker="o", markersize=6,
                markerfacecolor="white", markeredgecolor="#D55E00",
                label="Experimental | 8SD3 WT",
            ),
            Line2D(
                [0], [0], color="none", marker="s", markersize=6,
                markerfacecolor="white", markeredgecolor="#0072B2",
                label="Experimental | 8SDA L403A",
            ),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.01),
        ncol=2, title="Ensembles and experimental structures",
        frameon=True,
    )
    figure.subplots_adjust(bottom=0.24, top=0.88)
    plt.show()
    return axis


def plot_l403a_e423_n179_asymmetry(
    vanilla: pd.DataFrame,
    masked: pd.DataFrame,
    aliases: Mapping[str, str],
    experimental_wt: Mapping[str, Sequence[float]],
    experimental_l403a: Mapping[str, Sequence[float]],
):
    """Test whether E423–N179 elongation affects two or all four subunits.

    Chain letters are not treated as physical identities because cyclic chain
    registration can rotate among otherwise equivalent tetramers. Distances
    are therefore sorted within every model. A shifted subunit is defined by
    the midpoint between the longest WT-reference distance and the shortest of
    the two elongated 8SDA distances.
    """
    focal = {
        chain: aliases[f"E423{chain}-N179{chain}"]
        for chain in "ABCD"
        if f"E423{chain}-N179{chain}" in aliases
    }
    if len(focal) != 4:
        raise KeyError("All four E423–N179 chain distances are required")

    wt_values = np.asarray([
        experimental_wt[f"E423{chain}-N179{chain}"][0] for chain in "ABCD"
    ], dtype=float)
    mutant_values = np.asarray([
        experimental_l403a[f"E423{chain}-N179{chain}"][0] for chain in "ABCD"
    ], dtype=float)
    wt_sorted = np.sort(wt_values)
    mutant_sorted = np.sort(mutant_values)
    # 8SDA contains two WT-like and two elongated values. The boundary is
    # placed halfway across the experimentally observed separation.
    threshold = float((wt_sorted[-1] + mutant_sorted[-2]) / 2)

    rank_rows = []
    occupancy_rows = []
    model_rows = []
    for protocol, frame in (("vanilla", vanilla), ("masked", masked)):
        values = frame[list(focal.values())].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()
        sorted_values = np.sort(values.to_numpy(dtype=float), axis=1)
        shifted_counts = (sorted_values > threshold).sum(axis=1)
        for rank_index in range(4):
            rank_rows.extend(
                {
                    "Protocol": protocol,
                    "Rank": f"{rank_index + 1}",
                    "Distance": value,
                }
                for value in sorted_values[:, rank_index]
            )
        counts = pd.Series(shifted_counts).value_counts().reindex(
            range(5), fill_value=0
        )
        for count, frequency in counts.items():
            occupancy_rows.append({
                "Protocol": protocol,
                "Shifted subunits": int(count),
                "Fraction of models": float(frequency / len(values)),
            })
        raw_values = values.to_numpy(dtype=float)
        for model, count, row, raw_row in zip(
            values.index, shifted_counts, sorted_values, raw_values
        ):
            shifted_chains = [
                chain for chain, value in zip(focal, raw_row)
                if value > threshold
            ]
            model_rows.append({
                "Protocol": protocol,
                "Model": model,
                **{
                    f"Chain {chain} distance": float(value)
                    for chain, value in zip(focal, raw_row)
                },
                "Shifted chains": ",".join(shifted_chains) or "none",
                "Shifted subunits": int(count),
                "Minimum distance": float(row[0]),
                "Maximum distance": float(row[-1]),
                "Within-model span": float(row[-1] - row[0]),
            })

    rank_table = pd.DataFrame(rank_rows)
    occupancy_table = pd.DataFrame(occupancy_rows)
    model_table = pd.DataFrame(model_rows)
    colors = {
        "vanilla": KV21_PALETTE["L403A_VAN"],
        "masked": KV21_PALETTE["L403A_HM"],
    }
    figure, (rank_axis, occupancy_axis) = plt.subplots(
        1, 2, figsize=(12.4, 6.5),
        gridspec_kw={"width_ratios": (1.45, 1), "wspace": 0.28},
    )
    sns.violinplot(
        data=rank_table, x="Rank", y="Distance", hue="Protocol",
        hue_order=["vanilla", "masked"], split=True, inner="quartile",
        cut=0, linewidth=0.65, palette=colors, ax=rank_axis,
    )
    _tint_violin_borders(rank_axis)
    for index, (wt_value, mutant_value) in enumerate(
        zip(wt_sorted, mutant_sorted)
    ):
        rank_axis.scatter(
            index - 0.06, wt_value, marker=RMSD_REFERENCE_STYLES["8SD3"]["marker"],
            s=34, facecolors="white",
            edgecolors=RMSD_REFERENCE_STYLES["8SD3"]["color"],
            linewidths=0.9, zorder=8,
        )
        rank_axis.scatter(
            index + 0.06, mutant_value,
            marker=RMSD_REFERENCE_STYLES["8SDA"]["marker"],
            s=34, facecolors="white",
            edgecolors=RMSD_REFERENCE_STYLES["8SDA"]["color"],
            linewidths=0.9, zorder=8,
        )
    rank_axis.axhline(
        threshold, color="#7A6A86", linestyle="--", linewidth=0.9,
    )
    rank_axis.set(
        title="Distances ranked within each tetramer",
        xlabel="Subunit rank (shortest → longest)",
        ylabel="E423–N179 Cα distance (Å)",
    )
    if rank_axis.get_legend() is not None:
        rank_axis.get_legend().remove()

    sns.barplot(
        data=occupancy_table, x="Shifted subunits", y="Fraction of models",
        hue="Protocol", hue_order=["vanilla", "masked"],
        palette=colors, edgecolor="#38413B", linewidth=0.55,
        ax=occupancy_axis,
    )
    occupancy_axis.scatter(
        2, 1.0, marker=RMSD_REFERENCE_STYLES["8SDA"]["marker"],
        s=42, facecolors="white",
        edgecolors=RMSD_REFERENCE_STYLES["8SDA"]["color"],
        linewidths=1.0, zorder=8,
    )
    occupancy_axis.set(
        title="Shifted-subunit occupancy per model",
        xlabel=f"Subunits above {threshold:.2f} Å",
        ylabel="Fraction of models",
        ylim=(0, 1.08),
    )
    if occupancy_axis.get_legend() is not None:
        occupancy_axis.get_legend().remove()
    for axis in (rank_axis, occupancy_axis):
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)

    figure.legend(
        handles=[
            Patch(facecolor=colors["vanilla"], edgecolor="#78977F",
                  label="L403A | vanilla"),
            Patch(facecolor=colors["masked"], edgecolor="#356D46",
                  label="L403A | masked"),
            Line2D([0], [0], linestyle="none",
                   marker=RMSD_REFERENCE_STYLES["8SD3"]["marker"],
                   markerfacecolor="white",
                   markeredgecolor=RMSD_REFERENCE_STYLES["8SD3"]["color"],
                   label="8SD3 | WT"),
            Line2D([0], [0], linestyle="none",
                   marker=RMSD_REFERENCE_STYLES["8SDA"]["marker"],
                   markerfacecolor="white",
                   markeredgecolor=RMSD_REFERENCE_STYLES["8SDA"]["color"],
                   label="8SDA | L403A (two shifted subunits)"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 0.005),
        ncol=4, title="Ensembles and experimental references",
        frameon=True,
    )
    figure.suptitle(
        r"$\mathrm{K}_{\mathrm{V}}2.1$ | L403A | E423–N179 subunit asymmetry",
        fontsize=17, fontweight="semibold", y=0.985,
    )
    figure.subplots_adjust(bottom=0.22, top=0.86)
    return model_table, occupancy_table, threshold, figure


_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def _compact_residue_token(name: str, number: str, chain: str = "",
                           number_offset: int = 0) -> str:
    residue = _THREE_TO_ONE.get(name.upper(), name.title())
    return f"{residue}{int(number) + number_offset}{chain}"


def format_distance_alias(value: object, residue_number_offset: int = 0) -> str:
    """Render a raw distance column as compact one-letter residue pairs.

    The source column remains untouched. ``residue_number_offset`` affects only
    the visible number and is ``-2`` for Kv2.1 model columns displayed in the
    rat experimental/paper numbering.
    """
    text = str(value)
    measurement = ""
    if text.startswith("shortest_"):
        body = text.removeprefix("shortest_")
        measurement = "shortest"
    elif text.startswith("CA_CA_"):
        body = text.removeprefix("CA_CA_").replace("_CA", "")
        measurement = "Cα"
    elif text.startswith("CA_"):
        body = text.removeprefix("CA_").replace("_CA", "")
        measurement = "Cα"
    else:
        body = text

    # Raw Cα columns place chain before the residue: A_TYR378.
    body = re.sub(
        r"([A-D])_([A-Z]{3})(\d+)",
        lambda match: _compact_residue_token(
            match.group(2), match.group(3), match.group(1),
            residue_number_offset,
        ),
        body,
    )
    # Shortest-distance columns commonly place chain after the residue:
    # TYR378C. Unchained residue tokens are compacted by the same expression.
    body = re.sub(
        r"([A-Z]{3})(\d+)([A-D])?",
        lambda match: _compact_residue_token(
            match.group(1), match.group(2), match.group(3) or "",
            residue_number_offset,
        ),
        body,
    )
    body = body.replace("_", " ").replace("-", "–")
    return f"{body} ({measurement})" if measurement else body


def plot_top_shifts(
    shift_table: pd.DataFrame,
    title: str,
    n: int = 15,
    *,
    xlabel: str = "Median shift, mutant − WT (Å)",
    residue_number_offset: int | None = None,
):
    top = shift_table.head(n).sort_values("median_shift_A").copy()
    if residue_number_offset is None:
        residue_number_offset = -2 if re.search(r"Kv?2\.1", str(title), re.I) else 0
    top["display_distance"] = top["distance"].map(
        lambda value: format_distance_alias(value, residue_number_offset)
    )
    colors = [
        KV21_PALETTE["shift_farther"] if value > 0
        else KV21_PALETTE["shift_closer"]
        for value in top["median_shift_A"]
    ]
    ax = top.plot.barh(x="display_distance", y="median_shift_A", color=colors,
                       legend=False, figsize=(9, 7))
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_title(format_channel_title(title), pad=14)
    ax.set(xlabel=xlabel, ylabel="Distance")
    ax.grid(axis="y", visible=False); sns.despine(ax=ax); plt.tight_layout(); plt.show()
    return ax
