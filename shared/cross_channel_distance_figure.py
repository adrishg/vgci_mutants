"""Publication-style cross-channel distance-sampling figures.

The module keeps figure construction out of the presentation notebook.  Every
panel compares vanilla and masked predictions from the final channel-specific
QC table and uses the canonical experimental-reference styles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import pandas as pd
import seaborn as sns

from shared.experimental_overlays import experimental_rows
from shared.plotting import (
    CAV12_PALETTE,
    KV21_PALETTE,
    NAV15_PALETTE,
    experimental_reference_style,
    format_channel_title,
)


REGION_LABELS = {
    "selectivity_filter": "Selectivity filter",
    "vsds": "Voltage sensors",
    "intracellular_gate": "Intracellular gate",
}


ALIASES = {
    "Cav1.2": {
        "selectivity_filter": {
            "E363–E706": "CA_GLU363_CA-GLU706_CA",
            "E363–E1135": "CA_GLU363_CA-GLU1135_CA",
            "E363–E1464": "CA_GLU363_CA-GLU1464_CA",
            "E706–E1135": "CA_GLU706_CA-GLU1135_CA",
            "E706–E1464": "CA_GLU706_CA-GLU1464_CA",
            "E1135–E1464": "CA_GLU1135_CA-GLU1464_CA",
        },
        "vsds": {
            "F170–R237": "CA_ARG237_CA-PHE170_CA",
            "F170–R246": "CA_ARG246_CA-PHE170_CA",
            "F567–R620": "CA_ARG620_CA-PHE567_CA",
            "F567–K629": "CA_LYS629_CA-PHE567_CA",
            "F940–K1015": "CA_LYS1015_CA-PHE940_CA",
            "F940–R1031": "CA_ARG1031_CA-PHE940_CA",
            "F1278–R1377": "CA_ARG1377_CA-PHE1278_CA",
            "F1278–K1386": "CA_LYS1386_CA-PHE1278_CA",
        },
        "intracellular_gate": {
            "L401–L749": "CA_LEU401_CA-LEU749_CA",
            "L401–V1182": "CA_LEU401_CA-VAL1182_CA",
            "L401–I1516": "CA_LEU401_CA-ILE1516_CA",
            "L749–V1182": "CA_LEU749_CA-VAL1182_CA",
            "L749–I1516": "CA_LEU749_CA-ILE1516_CA",
            "V1182–I1516": "CA_VAL1182_CA-ILE1516_CA",
        },
    },
    "Kv2.1": {
        "selectivity_filter": {
            "G375 A–B": "CA_CA_A_GLY377_CA-B_GLY377_CA",
            "G375 A–C": "CA_CA_A_GLY377_CA-C_GLY377_CA",
            "G375 A–D": "CA_CA_A_GLY377_CA-D_GLY377_CA",
            "G375 B–C": "CA_CA_B_GLY377_CA-C_GLY377_CA",
            "G375 B–D": "CA_CA_B_GLY377_CA-D_GLY377_CA",
            "G375 C–D": "CA_CA_C_GLY377_CA-D_GLY377_CA",
        },
        "vsds": {
            "F236–R308 A": "CA_CA_A_ARG310_CA-A_PHE238_CA",
            "F236–R308 B": "CA_CA_B_ARG310_CA-B_PHE238_CA",
            "F236–R308 C": "CA_CA_C_ARG310_CA-C_PHE238_CA",
            "F236–R308 D": "CA_CA_D_ARG310_CA-D_PHE238_CA",
        },
        "intracellular_gate": {
            "A402 A–B": "CA_CA_A_ALA404_CA-B_ALA404_CA",
            "A402 A–C": "CA_CA_A_ALA404_CA-C_ALA404_CA",
            "A402 A–D": "CA_CA_A_ALA404_CA-D_ALA404_CA",
            "A402 B–C": "CA_CA_B_ALA404_CA-C_ALA404_CA",
            "A402 B–D": "CA_CA_B_ALA404_CA-D_ALA404_CA",
            "A402 C–D": "CA_CA_C_ALA404_CA-D_ALA404_CA",
        },
    },
    "Nav1.5": {
        "selectivity_filter": {
            "D373–E704": "CA_ASP373_CA-GLU704_CA",
            "D373–K1103": "CA_ASP373_CA-LYS1103_CA",
            "D373–A1395": "CA_ASP373_CA-ALA1395_CA",
            "E704–K1103": "CA_GLU704_CA-LYS1103_CA",
            "E704–A1395": "CA_GLU704_CA-ALA1395_CA",
            "K1103–A1395": "CA_LYS1103_CA-ALA1395_CA",
        },
        "vsds": {
            "F165–R220": "CA_ARG220_CA-PHE165_CA",
            "F165–K229": "CA_LYS229_CA-PHE165_CA",
            "F560–R612": "CA_ARG612_CA-PHE560_CA",
            "F560–K624": "CA_LYS624_CA-PHE560_CA",
            "F934–K984": "CA_LYS984_CA-PHE934_CA",
            "F934–R1000": "CA_ARG1000_CA-PHE934_CA",
            "F1251–R1307": "CA_ARG1307_CA-PHE1251_CA",
        },
        "intracellular_gate": {
            "M415–A742": "CA_MET415_CA-ALA742_CA",
            "M415–I1154": "CA_MET415_CA-ILE1154_CA",
            "M415–I1455": "CA_MET415_CA-ILE1455_CA",
            "A742–I1154": "CA_ALA742_CA-ILE1154_CA",
            "A742–I1455": "CA_ALA742_CA-ILE1455_CA",
            "I1154–I1455": "CA_ILE1154_CA-ILE1455_CA",
        },
    },
}


def paper_configs(repo_root: str | Path):
    """Return the three WT final-QC comparisons in manuscript order."""
    root = Path(repo_root)
    return {
        "Kv2.1": {
            "condition": "WT",
            "vanilla": root / "kv21/dataDistances/26-02-11_Kv2.1_wt_vanillaAF2test_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
            "masked": root / "kv21/dataDistances/26-02-11_Kv2.1_wt_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv",
            "colors": (KV21_PALETTE["WT_VAN"], KV21_PALETTE["WT_HM"]),
            "references": ("8SD3:", "9O10:", "9O11:", "9O12:", "9O13:"),
        },
        "Nav1.5": {
            "condition": "WT",
            "protocol_note": "targeted mask v2",
            "vanilla": root / "nav15/dataDistances/26-07-27_Nav15_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
            "masked": root / "nav15/dataDistances/26-07-27_Nav15_wt_maskedv2_AF2_distances_all_ok_rmsd_3A.csv",
            "colors": (NAV15_PALETTE["WT_VAN"], NAV15_PALETTE["WT_HM"]),
            "references": ("6UZ3:", "8VYJ:", "8VYK:"),
        },
        "Cav1.2": {
            "condition": "WT",
            "vanilla": root / "cav12/dataDistances/26-02-10_Cav12_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
            "masked": root / "cav12/dataDistances/26-02-10_Cav12_wt_maskedAF2_distances_all_ok_rmsd_3A.csv",
            "colors": (CAV12_PALETTE["WT_VAN"], CAV12_PALETTE["WT_HM"]),
            "references": ("8HLP", "8WE6", "8FD7"),
        },
    }


def load_paper_data(configs):
    loaded = {}
    for channel, config in configs.items():
        for protocol in ("vanilla", "masked"):
            path = config[protocol]
            if not path.is_file():
                raise FileNotFoundError(f"{channel} {protocol}: {path}")
        loaded[channel] = {
            "vanilla": pd.read_csv(config["vanilla"]),
            "masked": pd.read_csv(config["masked"]),
        }
    return loaded


def _available_aliases(frames, aliases):
    return {
        alias: column for alias, column in aliases.items()
        if all(column in frame.columns for frame in frames.values())
    }


def _selected_references(repo_root, channel, region, aliases, prefixes):
    # The display aliases use typographic en dashes, whereas the validated
    # overlay tables use ASCII hyphens. Preserve the compact display label
    # after matching against the source table.
    display_alias = {alias.replace("–", "-"): alias for alias in aliases}
    # Historical overlay labels retained two lysine names even though the
    # validated model columns and displayed one-letter identities are arginine.
    display_alias.update({
        "F940-K1031": "F940–R1031",
        "F934-K1000": "F934–R1000",
    })
    rows = experimental_rows(
        repo_root, channel, region, list(display_alias)
    )
    selected = []
    for row in rows:
        if not any(
            row["Structure"].startswith(prefix) for prefix in prefixes
        ):
            continue
        item = dict(row)
        if item["Alias"] not in display_alias:
            continue
        item["Alias"] = display_alias[item["Alias"]]
        selected.append(item)
    return selected


def _reference_style(structure, reference_prefixes):
    """Use one stable marker/color for a reference throughout a channel."""
    fallback_index = next(
        (
            index for index, prefix in enumerate(reference_prefixes)
            if str(structure).startswith(prefix)
        ),
        0,
    )
    return experimental_reference_style(structure, fallback_index)


def draw_distance_panel(
    axis,
    *,
    repo_root,
    channel,
    condition,
    region,
    frames,
    colors,
    reference_prefixes,
    show_title=True,
    show_ylabel=True,
):
    """Draw one fixed-format vanilla/masked panel on an existing axis."""
    aliases = _available_aliases(frames, ALIASES[channel][region])
    records = []
    for protocol in ("Vanilla", "Masked"):
        frame = frames[protocol.lower()]
        for alias, column in aliases.items():
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            records.extend(
                {"Alias": alias, "Distance": value, "Protocol": protocol}
                for value in values
            )
    plot_data = pd.DataFrame(records)
    sns.violinplot(
        data=plot_data, x="Alias", y="Distance", hue="Protocol",
        order=list(aliases), hue_order=["Vanilla", "Masked"], split=True,
        inner="quartile", cut=0, density_norm="width", common_norm=False,
        linewidth=0.55, saturation=0.88,
        palette={"Vanilla": colors[0], "Masked": colors[1]}, ax=axis,
    )
    if axis.get_legend() is not None:
        axis.get_legend().remove()

    references = _selected_references(
        repo_root, channel, region, list(aliases), reference_prefixes
    )
    seen = set()
    for row in references:
        structure = row["Structure"]
        style = _reference_style(structure, reference_prefixes)
        position = list(aliases).index(row["Alias"])
        axis.scatter(
            position, row["Distance"], s=131, marker=style["marker"],
            facecolors="white", edgecolors=style["color"], linewidths=1.5,
            zorder=8, label=structure if structure not in seen else None,
        )
        seen.add(structure)

    if show_title:
        axis.set_title(
            format_channel_title(
                f"{channel} | {condition}\n{REGION_LABELS[region]}"
            ),
            fontsize=38, fontweight="semibold", pad=8,
        )
    axis.set_xlabel("")
    axis.set_ylabel(
        "Cα distance (Å)" if show_ylabel else "",
        fontsize=44,
    )
    axis.text(
        0.01, 0.985, "vanilla ◀   ▶ masked",
        transform=axis.transAxes, ha="left", va="top",
        fontsize=26.0, color="#4D4D4D",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    axis.tick_params(axis="x", rotation=42, labelsize=30.0)
    axis.tick_params(axis="y", labelsize=29.0)
    plt.setp(
        axis.get_xticklabels(),
        horizontalalignment="right",
        rotation_mode="anchor",
    )
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", linewidth=0.45, alpha=0.35)
    sns.despine(ax=axis)
    return references


def _legend_handles(channel, config, references):
    protocol = [
        Patch(facecolor=config["colors"][0], edgecolor="#555555", label="Vanilla"),
        Patch(facecolor=config["colors"][1], edgecolor="#555555", label="Masked"),
    ]
    structures = list(dict.fromkeys(row["Structure"] for row in references))
    experimental = []
    for structure in structures:
        style = _reference_style(structure, config["references"])
        experimental.append(Line2D(
            [], [], linestyle="", marker=style["marker"], markersize=12.3,
            markerfacecolor="white", markeredgecolor=style["color"],
            markeredgewidth=1.5, label=structure,
        ))
    return protocol + experimental


def make_individual_panels(repo_root, configs, data, output_dir):
    """Export identically sized panels for flexible manuscript assembly."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for channel, config in configs.items():
        for region in ("selectivity_filter", "vsds", "intracellular_gate"):
            figure, axis = plt.subplots(figsize=(6.2, 5.1))
            references = draw_distance_panel(
                axis, repo_root=repo_root, channel=channel,
                condition=config["condition"], region=region,
                frames=data[channel], colors=config["colors"],
                reference_prefixes=config["references"],
            )
            handles = _legend_handles(channel, config, references)
            figure.legend(
                handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01),
                ncol=min(3, len(handles)), fontsize=7.2, frameon=True,
                title="Ensembles and experimental references",
                title_fontsize=7.8,
            )
            figure.subplots_adjust(bottom=0.31, left=0.12, right=0.97, top=0.86)
            safe_channel = channel.lower().replace(".", "")
            path = output / f"{safe_channel}_{config['condition'].split('|')[0].strip().lower()}_{region}.png"
            # Do not use bbox_inches='tight': preserving the fixed canvas makes
            # every exported panel exactly 3100 × 2550 pixels.
            figure.savefig(path, dpi=500)
            plt.close(figure)
            paths.append(path)
    return paths


def make_grid(
    repo_root,
    configs,
    data,
    *,
    regions: Sequence[str],
    output_path,
):
    """Create a consistently sized cross-channel grid."""
    channels = ("Kv2.1", "Nav1.5", "Cav1.2")
    figure = plt.figure(
        figsize=(8.82 * len(regions), 8.64 * len(channels) + 10.08)
    )
    grid = figure.add_gridspec(
        len(channels) + 1, len(regions),
        height_ratios=[1, 1, 1, 0.90],
    )
    axes = [
        [figure.add_subplot(grid[row, column]) for column in range(len(regions))]
        for row in range(len(channels))
    ]
    legend_axis = figure.add_subplot(grid[-1, :])
    legend_axis.axis("off")

    references_by_channel = {channel: [] for channel in channels}
    for row, channel in enumerate(channels):
        config = configs[channel]
        for column, region in enumerate(regions):
            panel_references = draw_distance_panel(
                axes[row][column], repo_root=repo_root, channel=channel,
                condition=config["condition"], region=region,
                frames=data[channel], colors=config["colors"],
                reference_prefixes=config["references"],
                show_ylabel=column == 0,
            )
            references_by_channel[channel].extend(panel_references)

    ensemble_handles = []
    for channel in channels:
        config = configs[channel]
        ensemble_handles.extend([
            Patch(
                facecolor=config["colors"][0], edgecolor="#555555",
                label=f"{format_channel_title(channel)} | vanilla",
            ),
            Patch(
                facecolor=config["colors"][1], edgecolor="#555555",
                label=f"{format_channel_title(channel)} | masked",
            ),
        ])
    experimental_handles = []
    for channel in channels:
        structures = list(dict.fromkeys(
            row["Structure"] for row in references_by_channel[channel]
        ))
        for structure in structures:
            style = _reference_style(structure, configs[channel]["references"])
            experimental_handles.append(Line2D(
                [], [], linestyle="", marker=style["marker"], markersize=12.75,
                markerfacecolor="white", markeredgecolor=style["color"],
                markeredgewidth=1.5,
                label=(
                    f"{format_channel_title(channel)} | "
                    f"{str(structure).split(':', 1)[0]}"
                ),
            ))

    # Keep each channel visually self-contained. Matplotlib fills multi-column
    # legends down each column, so three invisible entries reserve column 3 as
    # a spacer: Kv2.1 occupies columns 1–2, Nav1.5 column 4, and Cav1.2 column 5.
    spacer = Line2D([], [], linestyle="", marker="", alpha=0, label="")
    experimental_handles = (
        experimental_handles[:5]
        + [spacer]
        + [spacer, spacer, spacer]
        + experimental_handles[5:8]
        + experimental_handles[8:11]
    )

    legend_axis.text(
        0.5, 0.98, "Residue-pair landmark",
        transform=legend_axis.transAxes, ha="center", va="top",
        fontsize=48, fontweight="medium",
    )
    ensemble_legend = legend_axis.legend(
        handles=ensemble_handles,
        loc="upper center", bbox_to_anchor=(0.5, 0.78),
        ncol=3, fontsize=30.0, frameon=True,
        columnspacing=1.35, handletextpad=0.65,
    )
    legend_axis.add_artist(ensemble_legend)
    legend_axis.legend(
        handles=experimental_handles,
        loc="lower center", bbox_to_anchor=(0.5, -0.04),
        ncol=5, fontsize=28.0, frameon=True,
        columnspacing=1.25, handletextpad=0.6,
    )
    figure.subplots_adjust(
        left=0.050, right=0.992, top=0.985, bottom=0.03,
        hspace=0.70, wspace=0.06,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=500, bbox_inches="tight")
    return figure
