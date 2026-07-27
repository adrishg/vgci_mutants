"""Reusable plotting helpers for channel-distance notebooks.

Keep channel/protocol colors and figure styling here so notebooks contain only
data loading, analysis choices, and plot execution.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import pandas as pd
import seaborn as sns


STYLE_VERSION = "kv21-clean-green-v3-s6-profile"


KV21_PALETTE = {
    "WT_VAN": "#DBEAD7",
    "WT_HM": "#7FAC8A",
    "L403A_VAN": "#C6EAC7",
    "L403A_HM": "#4BAA6A",
    "F412L_VAN": "#B8E2D3",
    "F412L_HM": "#3E9175",
    "experimental": "#222222",
    "shift_closer": "#F48FB1",
    "shift_farther": "#FF8A65",
}

ACCENT_PALETTE = {
    "RED": "#E57373", "PINK": "#F48FB1", "RASPBERRY": "#F06292",
    "CORAL": "#FF8A65", "ORANGE": "#FFB74D", "PEACH": "#FFCC99",
    "APRICOT": "#FFD89A", "YELLOW": "#FFE082", "LEMON": "#FFF176",
    "CREAM": "#FFF6B3",
}

# Experimental structures must remain distinguishable in grayscale and for
# readers who cannot reliably separate nearby hues.  Color and shape therefore
# encode the same structure redundantly in every shared plotting helper.
EXPERIMENTAL_MARKERS = ("o", "s", "D", "^", "v", "P", "X", "*", "<", ">")

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
    "intracellular gate (A402 Cα; experimental numbering)": GATE_ALIASES,
    "selectivity filter (G375 Cα; experimental numbering)": SELECTIVITY_FILTER_ALIASES,
    "voltage sensor chain A (F236-R289/R308 Cα; experimental numbering)": VSD_CHAIN_A_ALIASES,
}


def apply_kv21_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook", rc=KV21_STYLE)


def save_figure_4k(fig, path, *, transparent: bool = False) -> None:
    """Export a 16:9 figure at 3840 × 2160 pixels without changing notebook display."""
    original_size = fig.get_size_inches().copy()
    fig.set_size_inches(12.8, 7.2)
    fig.savefig(path, dpi=300, bbox_inches=None, facecolor="none" if transparent else "white",
                transparent=transparent)
    fig.set_size_inches(original_size)


def finish_axes(ax, title: str, ylabel: str = "Cα distance (Å)") -> None:
    ax.set_title(title, pad=14)
    ax.set_xlabel("Residue-pair alias")
    ax.set_ylabel(ylabel)
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
            for alias, values in distances.items():
                if alias not in aliases:
                    continue
                xpos = list(aliases).index(alias)
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
):
    records = []
    for alias, (column_a, column_b) in alias_dict.items():
        for dataset, column in (("A", column_a), ("B", column_b)):
            if column in df.columns:
                records.extend({"Alias": alias, "Distance": value, "Dataset": dataset}
                               for value in pd.to_numeric(df[column], errors="coerce").dropna())
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        print("No numeric distances were available to plot.")
        return None
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    sns.violinplot(data=plot_df, x="Alias", y="Distance", hue="Dataset",
                   order=list(alias_dict), hue_order=["A", "B"], split=True,
                   inner="quartile", cut=0, linewidth=0.6, palette=colors, ax=ax)
    _tint_violin_borders(ax)
    finish_axes(ax, title_custom_add.lstrip("— "))
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
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(3, len(labels)), fontsize=9, title_fontsize=9, frameon=True,
    )
    fig.subplots_adjust(bottom=0.20)
    finish_figure(fig, legend_addon=True)
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
    # Keep the standard palette for single ensembles, but increase contrast in
    # overlays so WT and mutant distributions remain readable when transparent.
    overlay_mutants = {
        "L403A": {"VAN": "#2F8F55", "HM": "#176B3A"},
        "F412L": {"VAN": "#278A72", "HM": "#145C4A"},
    }
    palette = {"WT": KV21_PALETTE[f"WT_{suffix}"],
               mutant_label: overlay_mutants.get(mutant_label, {}).get(suffix, KV21_PALETTE[f"{mutant_label}_{suffix}"])}
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
        f"Kv2.1 | WT vs {mutant_label} | {protocol} | {region_label} | split violin | experimental distances",
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, title="Ensembles and experimental structures",
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=min(4, len(labels)), fontsize=9, title_fontsize=9, frameon=True,
    )
    ax.get_legend().remove()
    fig.subplots_adjust(bottom=0.18)
    finish_figure(fig, legend_addon=True)
    plt.show()
    return ax


def plot_top_shifts(shift_table: pd.DataFrame, title: str, n: int = 15):
    top = shift_table.head(n).sort_values("median_shift_A")
    colors = [KV21_PALETTE["shift_farther"] if value > 0 else KV21_PALETTE["shift_closer"]
              for value in top["median_shift_A"]]
    ax = top.plot.barh(x="distance", y="median_shift_A", color=colors,
                       legend=False, figsize=(9, 7))
    ax.axvline(0, color="#333333", linewidth=0.7)
    ax.set_title(title, pad=14)
    ax.set(xlabel="Median shift, mutant − WT (Å)", ylabel="Distance alias")
    ax.grid(axis="y", visible=False); sns.despine(ax=ax); plt.tight_layout(); plt.show()
    return ax
