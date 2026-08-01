"""Focused analysis of the Kv2.1 F412 hydrophobic coupling nexus.

The experimental benchmark is the asymmetric L403A structure (8SDA), because
an F412L structure could not be reconstructed.  Model residue numbers are two
positions higher than the experimental Kv2.1 labels used in the paper.
"""

from __future__ import annotations

from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser

from shared.rmsd_analysis import apply_kv21_rmsd_qc
from shared.plotting import (
    KV21_PALETTE,
    RMSD_REFERENCE_STYLES,
    ensemble_protocol_palette,
    format_channel_title,
)


REGIONS = {
    "Hydrophobic nexus": "hydrophobic_nexus__ca__core_aligned_rmsd_A",
    "L403 region": "l403_region__ca__core_aligned_rmsd_A",
    "F412 region": "f412_region__ca__core_aligned_rmsd_A",
}
CHAINS = tuple("ABCD")
VANILLA = KV21_PALETTE["F412L_VAN"]
MASKED = KV21_PALETTE["F412L_HM"]

def _basename(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda value: Path(value).name)


def _all_ok3_basenames(repo_root: str | Path) -> set[str]:
    manifest = pd.read_csv(
        Path(repo_root) / "kv21/dataRMSF/qc/kv21_all_ok3_selection_manifest.csv",
        usecols=["pdb_basename", "all_ok_3"],
    )
    return set(manifest.loc[manifest["all_ok_3"].fillna(False), "pdb_basename"])


def load_nexus_rmsd(repo_root: str | Path) -> pd.DataFrame:
    """Load only the RMSD columns needed for the nexus analysis."""
    root = Path(repo_root)
    source = root / "kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD_v2.csv"
    chain_columns = [
        f"{region.split('__', 1)[0]}__chain_{chain}__ca__core_aligned_rmsd_A"
        for region in REGIONS.values()
        for chain in CHAINS
    ]
    usecols = [
        "sequence_condition", "protocol", "pdb_file", "model_path",
        "reference_id", "analysis_status", "selected_core_postfit_rmsd_A",
        *REGIONS.values(), *chain_columns,
    ]
    frame = pd.read_csv(source, usecols=usecols)
    frame = frame.loc[frame["analysis_status"].eq("ok")].copy()
    allowed = _all_ok3_basenames(root)
    frame = frame.loc[_basename(frame["pdb_file"]).isin(allowed)].copy()
    return apply_kv21_rmsd_qc(frame, root)


def reference_preference(rmsd: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize whether each model is locally closer to 8SDA or 8SD3.

    Delta is RMSD(8SDA L403A) minus RMSD(8SD3 WT). Negative values are
    L403A-like; positive values are WT-like.
    """
    identifiers = ["sequence_condition", "protocol", "pdb_file", "model_path"]
    records = []
    for label, column in REGIONS.items():
        wide = rmsd.pivot_table(
            index=identifiers, columns="reference_id", values=column, aggfunc="first"
        ).reset_index()
        wide = wide.dropna(subset=["8SD3", "8SDA"])
        wide["Region"] = label
        wide["Preference delta (Å)"] = wide["8SDA"] - wide["8SD3"]
        wide["L403A-like"] = wide["Preference delta (Å)"] < 0
        records.append(wide)
    paired = pd.concat(records, ignore_index=True)
    summary = (
        paired.groupby(["sequence_condition", "protocol", "Region"], as_index=False)
        .agg(
            n=("Preference delta (Å)", "size"),
            median_delta_A=("Preference delta (Å)", "median"),
            mean_delta_A=("Preference delta (Å)", "mean"),
            fraction_L403A_like=("L403A-like", "mean"),
        )
    )
    return paired, summary


def rank_representatives(rmsd: pd.DataFrame) -> pd.DataFrame:
    """Rank retained models by mean local Cα RMSD to experimental L403A."""
    part = rmsd.loc[rmsd["reference_id"].eq("8SDA")].copy()
    part["combined_local_score_A"] = part[list(REGIONS.values())].mean(axis=1)
    columns = [
        "sequence_condition", "protocol", "pdb_file", "model_path",
        "combined_local_score_A", *REGIONS.values(),
    ]
    for chain in CHAINS:
        columns.append(
            f"hydrophobic_nexus__chain_{chain}__ca__core_aligned_rmsd_A"
        )
    return part[columns].sort_values("combined_local_score_A").reset_index(drop=True)


def plot_preference(summary: pd.DataFrame):
    """Plot the fraction of each ensemble entering an L403A-like local state."""
    plot = summary.loc[
        summary["sequence_condition"].isin(["wt", "l403a", "f412l"])
    ].copy()
    plot["Sequence"] = plot["sequence_condition"].map(
        {"wt": "WT", "l403a": "L403A", "f412l": "F412L"}
    )
    plot["Protocol"] = plot["protocol"].str.capitalize()
    sequence_order = ["WT", "L403A", "F412L"]
    protocol_order = ["Vanilla", "Masked"]
    condition_key = {"WT": "wt", "L403A": "l403a", "F412L": "f412l"}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.2), sharey=True)
    for ax, (region, part) in zip(axes, plot.groupby("Region", sort=False)):
        for sequence_index, sequence in enumerate(sequence_order):
            palette = ensemble_protocol_palette("kv21", condition_key[sequence])
            for offset, protocol in zip((-.18, .18), protocol_order):
                row = part.loc[
                    part["Sequence"].eq(sequence) & part["Protocol"].eq(protocol),
                    "fraction_L403A_like",
                ]
                if not row.empty:
                    ax.bar(
                        sequence_index + offset, row.iloc[0], width=.34,
                        color=palette[protocol], edgecolor="#425047", linewidth=.55,
                    )
        ax.set_xticks(range(len(sequence_order)), sequence_order)
        ax.set_title(region)
        ax.set_xlabel("")
        ax.set_ylim(0, 0.5)
        ax.set_ylabel("Fraction closer to 8SDA (L403A)" if ax is axes[0] else "")
        ax.grid(axis="y", linestyle="--", linewidth=.45, color="#E5E8E6")
        if ax.get_legend() is not None:
            ax.get_legend().remove()
    handles = []
    for sequence in sequence_order:
        palette = ensemble_protocol_palette("kv21", condition_key[sequence])
        for protocol in protocol_order:
            handles.append(Patch(
                facecolor=palette[protocol], edgecolor="#425047",
                label=f"{sequence} | {protocol.lower()}",
            ))
    fig.legend(handles=handles, title="Sequence and prediction protocol",
               frameon=False, loc="lower center", bbox_to_anchor=(.5, .015), ncol=3)
    fig.suptitle(format_channel_title("Kv2.1 | sampling of the L403A-like nexus"),
                 fontweight="bold", y=.99)
    sns.despine(fig=fig)
    fig.tight_layout(rect=(0, .24, 1, .94))
    return fig


_PAIR_RE = re.compile(
    r"shortest_(?P<res1>[A-Z]{3})(?P<num1>\d+)(?P<chain1>[A-D])-"
    r"(?P<res2>[A-Z]{3})(?P<num2>\d+)(?P<chain2>[A-D])"
)


def _nexus_distance_columns(columns) -> list[str]:
    """Select nexus distances while remaining independent of cyclic chain labels."""
    selected = []
    for column in columns:
        match = _PAIR_RE.fullmatch(str(column))
        if not match:
            continue
        numbers = {int(match["num1"]), int(match["num2"])}
        if 414 not in numbers or not numbers.intersection({318, 331, 405}):
            continue
        partner = next(number for number in numbers if number != 414)
        same_chain = match["chain1"] == match["chain2"]
        if (partner == 318 and same_chain) or (partner in {331, 405} and not same_chain):
            selected.append(column)
    return selected


def contact_signature(repo_root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize nearest nexus contacts in WT and F412L allOK3 models.

    Cross-chain labels cannot be compared directly between 8SD3, 8SDA and AF2
    models. We therefore use the closest available partner of each residue
    class in each tetramer. This tests nexus engagement without imposing an
    incorrect cyclic-chain correspondence.
    """
    root = Path(repo_root)
    allowed = _all_ok3_basenames(root)
    records = []
    for sequence_key, sequence_label in (("wt", "WT"), ("f412l", "F412L")):
        for protocol in ("vanilla", "masked"):
            source = (
                root
                / "kv21/dataDistances"
                / (
                    f"26-02-11_Kv2.1_{sequence_key}_{protocol}AF2_distances_"
                    "all_ok_rmsd_3A_structural_interface_alignment_qc.csv"
                )
            )
            if not source.is_file():
                raise FileNotFoundError(
                    f"Missing {sequence_label} {protocol} alignment-QC "
                    f"distance table: {source}"
                )
            header = pd.read_csv(source, nrows=0)
            distance_columns = _nexus_distance_columns(header.columns)
            if not distance_columns:
                raise KeyError(
                    f"No F412/L412-centered nexus distances found in {source}"
                )
            frame = pd.read_csv(
                source, usecols=["pdb_file", *distance_columns]
            )
            frame = frame.loc[_basename(frame["pdb_file"]).isin(allowed)]
            frame["_model"] = _basename(frame["pdb_file"])
            for partner_number, partner in {
                318: "L316 (S4–S5 linker)",
                331: "L329 (neighboring S5)",
                405: "L403 (neighboring S6)",
            }.items():
                partner_columns = [
                    column for column in distance_columns
                    if re.search(
                        fr"(?:^|[A-Z]){partner_number}[A-D](?:-|$)", column
                    )
                ]
                values = frame[partner_columns].apply(
                    pd.to_numeric, errors="coerce"
                )
                nearest = values.min(axis=1)
                for model_name, distance in zip(frame["_model"], nearest):
                    records.append({
                        "Sequence": sequence_label,
                        "Protocol": protocol.capitalize(),
                        "Contact": partner,
                        "Model": model_name,
                        "Nearest distance (Å)": distance,
                        "number_of_available_pairs": len(partner_columns),
                    })
    model_table = pd.DataFrame(records).dropna(subset=["Nearest distance (Å)"])
    summary = (
        model_table.groupby(
            ["Sequence", "Protocol", "Contact"], as_index=False
        )
        .agg(
            n=("Nearest distance (Å)", "size"),
            median_nearest_A=("Nearest distance (Å)", "median"),
            q25_nearest_A=("Nearest distance (Å)", lambda x: x.quantile(.25)),
            q75_nearest_A=("Nearest distance (Å)", lambda x: x.quantile(.75)),
            fraction_models_with_contact=("Nearest distance (Å)", lambda x: x.le(4).mean()),
            fraction_probable_overlap=("Nearest distance (Å)", lambda x: x.lt(2).mean()),
            number_of_available_pairs=("number_of_available_pairs", "max"),
        )
    )
    summary["coverage_note"] = summary["Contact"].map({
        "L316 (S4–S5 linker)": "4 same-chain distances available",
        "L329 (neighboring S5)": (
            "6 directed cross-chain distances available; legacy table does not "
            "contain every reverse orientation"
        ),
        "L403 (neighboring S6)": "12 directed cross-chain distances available",
    })
    return model_table, summary


def experimental_contact_signature(repo_root: str | Path) -> pd.DataFrame:
    """Measure paper-defined shortest heavy-atom contacts in 8SD3 and 8SDA."""
    root = Path(repo_root)
    parser = PDBParser(QUIET=True)
    records = []
    for reference, label in (("8SD3", "8SD3 | WT"), ("8SDA", "8SDA | L403A")):
        model = parser.get_structure(reference, root / f"kv21/experimental/{reference}.pdb")[0]
        for contact, partner_residue in {
            "L316 (S4–S5 linker)": 316,
            "L329 (neighboring S5)": 329,
            "L403 (neighboring S6)": 403,
        }.items():
            for source_chain in CHAINS:
                source = model[source_chain][(" ", 412, " ")]
                candidate_chains = (
                    [source_chain] if partner_residue == 316
                    else [chain for chain in CHAINS if chain != source_chain]
                )
                candidates = []
                for partner_chain in candidate_chains:
                    partner = model[partner_chain][(" ", partner_residue, " ")]
                    atoms_a = [atom for atom in source if atom.element != "H"]
                    atoms_b = [atom for atom in partner if atom.element != "H"]
                    distance = min(float(atom_a - atom_b)
                                   for atom_a in atoms_a for atom_b in atoms_b)
                    candidates.append((distance, partner_chain))
                distance, partner_chain = min(candidates)
                records.append({
                    "Reference": label, "Contact": contact,
                    "Source chain": source_chain, "Partner chain": partner_chain,
                    "Shortest heavy-atom distance (Å)": distance,
                })
    return pd.DataFrame(records)


def plot_contact_signature(summary: pd.DataFrame):
    """Show protocol-dependent retention of the three F412L nexus contacts."""
    summary = summary.loc[summary["Sequence"].eq("F412L")].copy()
    order = [
        "L316 (S4–S5 linker)", "L329 (neighboring S5)", "L403 (neighboring S6)"
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.barplot(
        data=summary, x="Contact", y="fraction_models_with_contact", hue="Protocol",
        order=order, hue_order=["Vanilla", "Masked"],
        palette={"Vanilla": VANILLA, "Masked": MASKED}, ax=ax,
    )
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Fraction of models with a nexus contact (≤4 Å)")
    ax.set_title(format_channel_title(
                 "Kv2.1 | F412L | hydrophobic-nexus contacts"),
                 fontweight="bold")
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    fig.legend(
        handles=handles, labels=labels, title="Prediction protocol",
        frameon=False, ncol=2, loc="lower center",
        bbox_to_anchor=(.5, .015),
    )
    ax.grid(axis="y", linestyle="--", linewidth=.45, color="#E5E8E6")
    sns.despine(ax=ax)
    fig.tight_layout(rect=(0, .16, 1, 1))
    return fig


def plot_contact_distances(model_table: pd.DataFrame, experimental: pd.DataFrame):
    """Compare WT F412 and F412L L412 nexus-distance distributions."""
    order = [
        "L316 (S4–S5 linker)", "L329 (neighboring S5)", "L403 (neighboring S6)"
    ]
    protocol_palettes = {
        "Vanilla": {
            "WT": KV21_PALETTE["WT_VAN"],
            "F412L": KV21_PALETTE["F412L_VAN"],
        },
        "Masked": {
            "WT": KV21_PALETTE["WT_HM"],
            "F412L": KV21_PALETTE["F412L_HM"],
        },
    }
    fig, axes = plt.subplots(
        1, 2, figsize=(12.4, 5.7), sharey=True, squeeze=False
    )
    offsets = {"8SD3 | WT": -.18, "8SDA | L403A": .18}
    colors = {
        "8SD3 | WT": RMSD_REFERENCE_STYLES["8SD3"]["color"],
        "8SDA | L403A": RMSD_REFERENCE_STYLES["8SDA"]["color"],
    }
    markers = {
        "8SD3 | WT": RMSD_REFERENCE_STYLES["8SD3"]["marker"],
        "8SDA | L403A": RMSD_REFERENCE_STYLES["8SDA"]["marker"],
    }
    for panel_index, protocol in enumerate(("Vanilla", "Masked")):
        ax = axes[0, panel_index]
        part = model_table.loc[model_table["Protocol"].eq(protocol)]
        sns.violinplot(
            data=part,
            x="Contact",
            y="Nearest distance (Å)",
            hue="Sequence",
            order=order,
            hue_order=["WT", "F412L"],
            split=True,
            inner="quartile",
            cut=0,
            palette=protocol_palettes[protocol],
            linewidth=.65,
            ax=ax,
        )
        for reference, reference_part in experimental.groupby("Reference"):
            for index, contact in enumerate(order):
                values = reference_part.loc[
                    reference_part["Contact"].eq(contact),
                    "Shortest heavy-atom distance (Å)",
                ]
                ax.scatter(
                    np.full(len(values), index + offsets[reference]),
                    values,
                    s=28,
                    marker=markers[reference],
                    facecolor="white",
                    edgecolor=colors[reference],
                    linewidth=1.15,
                    zorder=5,
                )
        ax.axhline(4, color="#71637A", linestyle="--", linewidth=.8)
        ax.set_xticks(range(len(order)), [
            "F412/L412→L316\nsame-chain S4–S5 linker",
            "F412/L412→L329\nclosest neighboring S5",
            "F412/L412→L403\nclosest neighboring S6",
        ])
        ax.set_xlabel("Hydrophobic-nexus interaction")
        ax.set_ylabel(
            "Shortest heavy-atom distance (Å)" if panel_index == 0 else ""
        )
        ax.set_title(protocol.lower(), fontweight="bold")
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.grid(axis="y", linestyle="--", linewidth=.45, color="#E5E8E6")
        sns.despine(ax=ax)
    legend_handles = [
        Patch(
            facecolor=KV21_PALETTE["WT_VAN"],
            edgecolor="#425047",
            linewidth=.65,
            label="WT | vanilla",
        ),
        Patch(
            facecolor=KV21_PALETTE["F412L_VAN"],
            edgecolor="#425047",
            linewidth=.65,
            label="F412L | vanilla",
        ),
        Patch(
            facecolor=KV21_PALETTE["WT_HM"],
            edgecolor="#425047",
            linewidth=.65,
            label="WT | masked",
        ),
        Patch(
            facecolor=KV21_PALETTE["F412L_HM"],
            edgecolor="#425047",
            linewidth=.65,
            label="F412L | masked",
        ),
    ]
    for reference in ("8SD3 | WT", "8SDA | L403A"):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="",
                marker=markers[reference],
                markerfacecolor="white",
                markeredgecolor=colors[reference],
                markeredgewidth=1.15,
                markersize=6,
                label=reference,
            )
        )
    fig.legend(
        handles=legend_handles,
        title="F412L ensembles and Kv2.1 references",
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(.5, .065),
    )
    fig.suptitle(
        format_channel_title(
            "Kv2.1 | WT vs F412L | hydrophobic-nexus geometry"
        ),
        fontweight="bold",
        y=.99,
    )
    fig.text(
        .5, .012,
        "Left violin half: WT F412; right half: F412L L412. "
        "Experimental markers: F412 in 8SD3 WT and chain-resolved 8SDA "
        "L403A. Dashed line: 4 Å proximity guide.",
        ha="center", fontsize=8.7, color="#665A70",
    )
    fig.tight_layout(rect=(0, .17, 1, .93))
    return fig


def plot_f412l_contact_distances(
    model_table: pd.DataFrame, experimental: pd.DataFrame
):
    """Show F412L vanilla and masked ensembles as paired split violins."""
    order = [
        "L316 (S4–S5 linker)", "L329 (neighboring S5)",
        "L403 (neighboring S6)",
    ]
    protocol_colors = {
        "Vanilla": KV21_PALETTE["F412L_VAN"],
        "Masked": KV21_PALETTE["F412L_HM"],
    }
    offsets = {"8SD3 | WT": -.10, "8SDA | L403A": .10}
    colors = {
        "8SD3 | WT": RMSD_REFERENCE_STYLES["8SD3"]["color"],
        "8SDA | L403A": RMSD_REFERENCE_STYLES["8SDA"]["color"],
    }
    markers = {
        "8SD3 | WT": RMSD_REFERENCE_STYLES["8SD3"]["marker"],
        "8SDA | L403A": RMSD_REFERENCE_STYLES["8SDA"]["marker"],
    }
    mutant = model_table.loc[model_table["Sequence"].eq("F412L")].copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    sns.violinplot(
        data=mutant,
        x="Contact",
        y="Nearest distance (Å)",
        hue="Protocol",
        order=order,
        hue_order=["Vanilla", "Masked"],
        split=True,
        inner="quartile",
        cut=0,
        palette=protocol_colors,
        linewidth=.70,
        ax=ax,
    )
    for reference, reference_part in experimental.groupby("Reference"):
        for index, contact in enumerate(order):
            values = reference_part.loc[
                reference_part["Contact"].eq(contact),
                "Shortest heavy-atom distance (Å)",
            ]
            ax.scatter(
                np.full(len(values), index + offsets[reference]),
                values,
                s=28,
                marker=markers[reference],
                facecolor="white",
                edgecolor=colors[reference],
                linewidth=1.15,
                zorder=5,
            )
    ax.axhline(4, color="#71637A", linestyle="--", linewidth=.8)
    ax.set_xticks(range(len(order)), [
        "L412→L316\nsame-chain S4–S5 linker",
        "L412→L329\nclosest neighboring S5",
        "L412→L403\nclosest neighboring S6",
    ])
    ax.set_xlabel("Hydrophobic-nexus interaction")
    ax.set_ylabel("Shortest heavy-atom distance (Å)")
    automatic_legend = ax.get_legend()
    if automatic_legend is not None:
        automatic_legend.remove()
    ax.grid(axis="y", linestyle="--", linewidth=.45, color="#E5E8E6")
    sns.despine(ax=ax)
    legend_handles = [
        Patch(
            facecolor=protocol_colors["Vanilla"],
            edgecolor="#425047",
            linewidth=.65,
            label="F412L | vanilla",
        ),
        Patch(
            facecolor=protocol_colors["Masked"],
            edgecolor="#425047",
            linewidth=.65,
            label="F412L | masked",
        ),
    ]
    for reference in ("8SD3 | WT", "8SDA | L403A"):
        legend_handles.append(
            Line2D(
                [0], [0], linestyle="", marker=markers[reference],
                markerfacecolor="white", markeredgecolor=colors[reference],
                markeredgewidth=1.15, markersize=6, label=reference,
            )
        )
    fig.legend(
        handles=legend_handles,
        title="F412L ensembles and Kv2.1 references",
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(.5, .065),
    )
    fig.suptitle(
        format_channel_title("Kv2.1 | F412L | hydrophobic-nexus geometry"),
        fontweight="bold",
        y=.99,
    )
    fig.text(
        .5, .012,
        "Left violin half: F412L vanilla; right half: F412L masked. "
        "Experimental markers: F412 in 8SD3 WT and chain-resolved 8SDA "
        "L403A. Dashed line: 4 Å proximity guide.",
        ha="center", fontsize=8.7, color="#665A70",
    )
    fig.tight_layout(rect=(0, .17, 1, .93))
    return fig


def write_nexus_outputs(repo_root: str | Path):
    """Recompute and save the focused tables and publication-resolution figures."""
    root = Path(repo_root)
    output = root / "kv21/dataRMSD/analysis/nexus"
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rmsd = load_nexus_rmsd(root)
    paired, preference = reference_preference(rmsd)
    representatives = rank_representatives(rmsd)
    contact_pairs, contacts = contact_signature(root)
    experimental = experimental_contact_signature(root)

    preference.to_csv(output / "reference_preference_summary.csv", index=False)
    representatives.to_csv(output / "representative_model_ranking.csv", index=False)
    contact_pairs.to_csv(
        output / "wt_f412l_nexus_contact_chain_pairs.csv", index=False
    )
    contacts.to_csv(
        output / "wt_f412l_nexus_contact_summary.csv", index=False
    )
    contact_pairs.loc[contact_pairs["Sequence"].eq("F412L")].to_csv(
        output / "f412l_nexus_contact_chain_pairs.csv", index=False
    )
    contacts.loc[contacts["Sequence"].eq("F412L")].to_csv(
        output / "f412l_nexus_contact_summary.csv", index=False
    )
    experimental.to_csv(output / "experimental_nexus_contacts.csv", index=False)

    preference_figure = plot_preference(preference)
    preference_figure.savefig(
        figures / "nexus_l403a_like_fraction.png", dpi=400, bbox_inches="tight"
    )
    contact_figure = plot_contact_signature(contacts)
    contact_figure.savefig(
        figures / "f412l_nexus_contacts.png", dpi=400, bbox_inches="tight"
    )
    geometry_figure = plot_contact_distances(contact_pairs, experimental)
    geometry_figure.savefig(
        figures / "f412l_nexus_geometry_vs_experiment.png",
        dpi=400, bbox_inches="tight",
    )
    mutant_geometry_figure = plot_f412l_contact_distances(
        contact_pairs, experimental
    )
    mutant_geometry_figure.savefig(
        figures / "f412l_only_nexus_geometry_vs_experiment.png",
        dpi=400, bbox_inches="tight",
    )
    return {
        "rmsd": rmsd, "paired": paired, "preference": preference,
        "representatives": representatives, "contact_pairs": contact_pairs,
        "contacts": contacts, "preference_figure": preference_figure,
        "experimental": experimental, "contact_figure": contact_figure,
        "geometry_figure": geometry_figure,
        "mutant_geometry_figure": mutant_geometry_figure,
    }
