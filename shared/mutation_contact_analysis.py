"""Shortest-distance analysis for newly introduced mutation side chains."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser

from shared.plotting import (
    RMSD_REFERENCE_STYLES,
    format_channel_title,
    format_distance_alias,
)


def _display_residue(value: object, channel: str) -> str:
    """Use one-letter residue codes and the channel's publication numbering."""
    offset = -2 if str(channel).lower().replace(" ", "").startswith("kv2.1") else 0
    return format_distance_alias(value, offset)


def _display_residue_with_chain(value: object, channel: str) -> str:
    """Spell out a terminal chain ID so it is not mistaken for an amino acid."""
    match = re.fullmatch(r"([A-Z]{3}\d+)([A-Z])", str(value))
    if not match:
        return _display_residue(value, channel)
    residue, chain = match.groups()
    return f"{_display_residue(residue, channel)} (chain {chain})"


def mutation_contact_table(
    wt: pd.DataFrame,
    mutant: pd.DataFrame,
    wt_residue: str,
    mutant_residue: str,
    protocol: str,
) -> pd.DataFrame:
    """Compare partner-specific shortest distances for WT and mutant residues."""
    rows = []
    prefix = f"shortest_{mutant_residue}-"
    for mutant_column in [column for column in mutant if column.startswith(prefix)]:
        partner = mutant_column.split("-", 1)[1]
        wt_column = f"shortest_{wt_residue}-{partner}"
        mutant_values = pd.to_numeric(mutant[mutant_column], errors="coerce").dropna()
        wt_values = (
            pd.to_numeric(wt[wt_column], errors="coerce").dropna()
            if wt_column in wt
            else pd.Series(dtype=float)
        )
        rows.append(
            {
                "Protocol": protocol,
                "Partner": partner,
                "WT median (Å)": wt_values.median() if len(wt_values) else np.nan,
                "Mutant median (Å)": mutant_values.median(),
                "Mutant − WT (Å)": (
                    mutant_values.median() - wt_values.median()
                    if len(wt_values)
                    else np.nan
                ),
                "WT contact ≤4 Å": (wt_values <= 4).mean() if len(wt_values) else np.nan,
                "Mutant contact ≤4 Å": (mutant_values <= 4).mean(),
                "Mutant overlap <2 Å": (mutant_values < 2).mean(),
                "n": len(mutant_values),
            }
        )
    return pd.DataFrame(rows)


def mutation_protocol_contact_table(
    comparisons: Mapping[str, pd.DataFrame],
    mutant_residue: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize every shortest-distance column involving a mutant residue.

    Distances from 2–4 Å are counted as geometrically plausible contacts.
    Values below 2 Å are reported separately because they are more consistent
    with atomic overlap than with a stabilizing side-chain interaction.
    """
    rows: list[dict[str, object]] = []
    provenance: list[dict[str, object]] = []
    for protocol, frame in comparisons.items():
        partner_columns: dict[str, list[str]] = {}
        for column in frame.columns:
            if not column.startswith("shortest_") or "-" not in column:
                continue
            first, second = column.removeprefix("shortest_").split("-", 1)
            if first == mutant_residue:
                partner_columns.setdefault(second, []).append(column)
            elif second == mutant_residue:
                partner_columns.setdefault(first, []).append(column)

        for partner, columns in partner_columns.items():
            numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
            values = numeric.min(axis=1, skipna=True).dropna()
            provenance.append(
                {
                    "Protocol": protocol,
                    "Mutant residue": mutant_residue,
                    "Partner": partner,
                    "Source columns": "; ".join(columns),
                    "Number of source columns": len(columns),
                }
            )
            rows.append(
                {
                    "Protocol": protocol,
                    "Mutant residue": mutant_residue,
                    "Partner": partner,
                    "n": len(values),
                    "Median (Å)": values.median(),
                    "5th percentile (Å)": values.quantile(0.05),
                    "95th percentile (Å)": values.quantile(0.95),
                    "Plausible contact 2–4 Å": values.between(
                        2, 4, inclusive="both"
                    ).mean(),
                    "Atomic overlap <2 Å": values.lt(2).mean(),
                    "Shortest distance ≤4 Å": values.le(4).mean(),
                }
            )

    table = pd.DataFrame(rows)
    if table.empty:
        return table, pd.DataFrame(provenance)

    wide = table.pivot(index="Partner", columns="Protocol")
    for metric in (
        "Median (Å)",
        "Plausible contact 2–4 Å",
        "Atomic overlap <2 Å",
    ):
        if (metric, "vanilla") in wide and (metric, "masked") in wide:
            delta_name = {
                "Median (Å)": "Masked − vanilla median (Å)",
                "Plausible contact 2–4 Å":
                    "Masked − vanilla plausible-contact fraction",
                "Atomic overlap <2 Å":
                    "Masked − vanilla overlap fraction",
            }[metric]
            deltas = (
                wide[(metric, "masked")] - wide[(metric, "vanilla")]
            ).rename(delta_name)
            table = table.merge(deltas, left_on="Partner", right_index=True)

    max_contact = table.groupby("Partner")["Plausible contact 2–4 Å"].transform("max")
    max_overlap = table.groupby("Partner")["Atomic overlap <2 Å"].transform("max")
    table["Contact interpretation"] = np.select(
        [max_overlap.ge(0.01), max_contact.ge(0.05), max_contact.gt(0)],
        [
            "overlap-sensitive; inspect structures",
            "recurrent plausible contact",
            "rare plausible contact",
        ],
        default="no sampled contact",
    )
    return table.sort_values(
        ["Atomic overlap <2 Å", "Plausible contact 2–4 Å", "Median (Å)"],
        ascending=[False, False, True],
    ), pd.DataFrame(provenance)


def cav12_experimental_shortest_distances(
    repo_root: str | Path,
    raw_site: int,
    partners: list[str],
) -> pd.DataFrame:
    """Calculate WT experimental shortest distances using validated raw mapping.

    CaV1.2 experimental structures contain glycine at the G402/G406 sites.
    These values are therefore WT structural baselines, not mutation-matched
    serine- or arginine-side-chain measurements.
    """
    root = Path(repo_root)
    mapping_path = (
        root / "cav12" / "dataRMSF" / "references"
        / "cav12_aligned_references.npz"
    )
    mapping = np.load(mapping_path, allow_pickle=False)
    reference_ids = [str(item) for item in mapping["reference_ids"]]
    raw_numbers = mapping["raw_residue_numbers"].astype(int)
    pdb_numbers = mapping["pdb_residue_numbers"].astype(int)
    identities = mapping["identities"]
    raw_index = {int(raw): index for index, raw in enumerate(raw_numbers)}
    chains = {"8WE6": "A", "8HLP": "A", "8FD7": "K"}
    parser = PDBParser(QUIET=True)
    rows: list[dict[str, object]] = []
    for reference_index, reference_id in enumerate(reference_ids):
        chain_id = chains[reference_id]
        pdb_path = root / "cav12" / "experimental" / f"{reference_id}.pdb"
        chain = parser.get_structure(reference_id, pdb_path)[0][chain_id]
        site_index = raw_index[raw_site]
        site_pdb_number = int(pdb_numbers[reference_index, site_index])
        if site_pdb_number <= 0 or site_pdb_number not in chain:
            continue
        site_residue = chain[site_pdb_number]
        site_identity = str(identities[reference_index, site_index])
        for partner in partners:
            partner_raw = int("".join(character for character in partner if character.isdigit()))
            partner_index = raw_index[partner_raw]
            partner_pdb_number = int(pdb_numbers[reference_index, partner_index])
            if partner_pdb_number <= 0 or partner_pdb_number not in chain:
                continue
            partner_residue = chain[partner_pdb_number]
            atoms_a = [
                atom for atom in site_residue.get_atoms()
                if str(atom.element).upper() != "H"
            ]
            atoms_b = [
                atom for atom in partner_residue.get_atoms()
                if str(atom.element).upper() != "H"
            ]
            distance = min(
                np.linalg.norm(atom_a.coord - atom_b.coord)
                for atom_a in atoms_a for atom_b in atoms_b
            )
            rows.append(
                {
                    "Structure": reference_id,
                    "State": "WT experimental",
                    "Chain": chain_id,
                    "Raw site": raw_site,
                    "Site identity": site_identity,
                    "Site PDB residue": site_pdb_number,
                    "Partner": partner,
                    "Partner PDB residue": partner_pdb_number,
                    "Shortest distance (Å)": float(distance),
                }
            )
    return pd.DataFrame(rows)


def kv21_l403a_experimental_shortest_distances(
    repo_root: str | Path,
    partners: list[str],
) -> pd.DataFrame:
    """Calculate chain-matched L403/A403 shortest distances in 8SD3/8SDA.

    Prediction CSV residue numbers are two positions above the rat
    experimental/paper numbering. The mutation site is fixed to chain A,
    matching the mutation-contact panels.
    """
    root = Path(repo_root)
    parser = PDBParser(QUIET=True)
    references = {
        "8SD3": ("LEU", 403, "WT L403"),
        "8SDA": ("ALA", 403, "L403A A403"),
    }
    rows: list[dict[str, object]] = []
    for reference, (expected_name, site_number, site_label) in references.items():
        model = parser.get_structure(
            reference, root / "kv21" / "experimental" / f"{reference}.pdb"
        )[0]
        site = model["A"][(" ", site_number, " ")]
        if site.resname != expected_name:
            raise ValueError(
                f"{reference} chain A residue {site_number} is {site.resname}; "
                f"expected {expected_name}."
            )
        for partner in partners:
            match = re.fullmatch(r"([A-Z]{3})(\d+)([A-D])", str(partner))
            if not match:
                continue
            expected_partner, model_number, chain_id = match.groups()
            paper_number = int(model_number) - 2
            partner_residue = model[chain_id][(" ", paper_number, " ")]
            if partner_residue.resname != expected_partner:
                raise ValueError(
                    f"{reference} {chain_id}{paper_number} is "
                    f"{partner_residue.resname}; expected {expected_partner}."
                )
            atoms_a = [
                atom for atom in site.get_atoms()
                if str(atom.element).upper() != "H"
            ]
            atoms_b = [
                atom for atom in partner_residue.get_atoms()
                if str(atom.element).upper() != "H"
            ]
            distance = min(
                np.linalg.norm(atom_a.coord - atom_b.coord)
                for atom_a in atoms_a for atom_b in atoms_b
            )
            rows.append(
                {
                    "Structure": reference,
                    "Site": site_label,
                    "Site chain": "A",
                    "Partner": partner,
                    "Partner paper label": _display_residue(partner, "Kv2.1"),
                    "Shortest distance (Å)": float(distance),
                }
            )
    return pd.DataFrame(rows)


def plot_mutation_protocol_contacts(
    comparisons: Mapping[str, pd.DataFrame],
    mutant_residue: str,
    mutant_label: str,
    colors: Mapping[str, str],
    channel: str = "Cav1.2",
    top_n: int = 8,
):
    """Compare mutant-site contact and overlap frequencies by protocol."""
    table, provenance = mutation_protocol_contact_table(
        comparisons, mutant_residue
    )
    if table.empty:
        raise ValueError(f"No shortest-distance columns contain {mutant_residue}.")

    rank = (
        table.groupby("Partner")
        .agg(
            plausible=("Plausible contact 2–4 Å", "max"),
            overlap=("Atomic overlap <2 Å", "max"),
            median=("Median (Å)", "min"),
        )
        .sort_values(["plausible", "overlap", "median"], ascending=[False, False, True])
        .head(top_n)
        .index
    )
    plot_df = table[table["Partner"].isin(rank)].copy()
    plot_df["Display partner"] = plot_df["Partner"].map(
        lambda value: _display_residue(value, channel)
    )
    order = [_display_residue(value, channel) for value in rank]
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4), sharey=True)
    for ax, metric, title in (
        (axes[0], "Plausible contact 2–4 Å", "Plausible side-chain contact"),
        (axes[1], "Atomic overlap <2 Å", "Probable atomic overlap"),
    ):
        sns.barplot(
            data=plot_df,
            y="Display partner",
            x=metric,
            hue="Protocol",
            order=order,
            hue_order=["vanilla", "masked"],
            palette=colors,
            edgecolor="#45504A",
            linewidth=0.55,
            ax=ax,
        )
        ax.set_title(title)
        ax.set_ylabel("Interaction partner" if ax is axes[0] else "")
        ax.set_xlabel("Fraction of models")
        ax.grid(axis="x", color="#E9ECEF", linewidth=0.45, linestyle="--")
        if metric == "Atomic overlap <2 Å" and plot_df[metric].max() == 0:
            ax.set_xlim(0, 0.05)
            ax.text(
                0.5, 0.5, "No distances below 2 Å",
                transform=ax.transAxes, ha="center", va="center",
                color="#665A70",
            )
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        sns.despine(ax=ax)
    handles = [
        Patch(
            facecolor=colors[protocol], edgecolor="#45504A",
            linewidth=.65, label=protocol,
        )
        for protocol in ("vanilla", "masked")
    ]
    fig.legend(
        handles=handles, title="Prediction protocol",
        loc="upper center", bbox_to_anchor=(.5, .91),
        ncol=2, frameon=False,
    )
    fig.suptitle(
        format_channel_title(
            f"{channel} | {mutant_label} | mutation-site contacts"
        ),
        fontweight="bold",
    )
    fig.text(
        .5, .01,
        "Contacts are counted only from 2–4 Å; distances <2 Å are shown "
        "separately as probable atomic overlap.",
        ha="center", fontsize=9, color="#665A70",
    )
    fig.tight_layout(rect=(0, .05, 1, .84))
    return table, provenance, fig


def plot_mutant_contact_distributions(
    comparisons: Mapping[str, pd.DataFrame],
    mutant_residue: str,
    mutant_label: str,
    colors: Mapping[str, str],
    partners: list[str],
    channel: str = "Cav1.2",
    experimental_distances: pd.DataFrame | None = None,
):
    """Show vanilla-versus-masked mutant-site shortest-distance distributions."""
    records: list[dict[str, object]] = []
    for protocol, frame in comparisons.items():
        for partner in partners:
            candidates = [
                f"shortest_{mutant_residue}-{partner}",
                f"shortest_{partner}-{mutant_residue}",
            ]
            columns = [column for column in candidates if column in frame]
            if not columns:
                continue
            values = frame[columns].apply(pd.to_numeric, errors="coerce").min(axis=1)
            records.extend(
                {
                    "Protocol": protocol,
                    "Partner": partner,
                    "Shortest distance (Å)": value,
                }
                for value in values.dropna()
            )
    plot_df = pd.DataFrame(records)
    if plot_df.empty:
        raise ValueError("None of the requested mutation-site partners were found.")
    plot_df["Display partner"] = plot_df["Partner"].map(
        lambda value: _display_residue(value, channel)
    )
    order = [_display_residue(value, channel) for value in partners]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.violinplot(
        data=plot_df,
        x="Display partner",
        y="Shortest distance (Å)",
        hue="Protocol",
        order=order,
        hue_order=["vanilla", "masked"],
        split=True,
        inner="quartile",
        cut=0,
        linewidth=.65,
        palette=colors,
        ax=ax,
    )
    ax.axhspan(0, 2, color="#F4D6D7", alpha=.35, zorder=0)
    ax.axhline(2, color="#C44E52", linewidth=.9, linestyle=":")
    ax.axhline(4, color="#7B6D86", linewidth=.8, linestyle="--")
    experimental_handles: list[Line2D] = []
    if experimental_distances is not None and not experimental_distances.empty:
        structure_order = [
            item for item in ("8WE6", "8HLP", "8FD7")
            if item in set(experimental_distances["Structure"])
        ]
        offsets = np.linspace(-0.12, 0.12, max(len(structure_order), 1))
        for structure, offset in zip(structure_order, offsets):
            style = RMSD_REFERENCE_STYLES[structure]
            subset = experimental_distances[
                experimental_distances["Structure"].eq(structure)
            ]
            for _, row in subset.iterrows():
                display_partner = _display_residue(row["Partner"], channel)
                if display_partner not in order:
                    continue
                ax.scatter(
                    order.index(display_partner) + offset,
                    row["Shortest distance (Å)"],
                    s=34,
                    marker=style["marker"],
                    facecolor="white",
                    edgecolor=style["color"],
                    linewidth=1.15,
                    zorder=6,
                )
            experimental_handles.append(
                Line2D(
                    [0], [0], linestyle="", marker=style["marker"],
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.15, markersize=6,
                    label=f"{structure} | WT experimental",
                )
            )
    ax.set_title(
        format_channel_title(
            f"{channel} | {mutant_label} | mutation-site shortest distances"
        ),
        fontweight="bold",
    )
    ax.set_xlabel(f"Partner of {_display_residue(mutant_residue, channel)}")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="y", color="#E9ECEF", linewidth=.45, linestyle="--")
    sns.despine(ax=ax)
    protocol_handles = [
        Patch(
            facecolor=colors[protocol], edgecolor="#45504A",
            linewidth=.65, label=protocol,
        )
        for protocol in ("vanilla", "masked")
    ]
    legend = ax.legend(
        handles=protocol_handles + experimental_handles,
        title="Prediction ensembles and WT references",
        frameon=False, ncol=3,
        loc="upper center", bbox_to_anchor=(0.5, 1.0),
    )
    legend.set_zorder(10)
    fig.text(
        .5, .01,
        "Dashed line: 4 Å contact boundary. Red region: probable atomic "
        "overlap (<2 Å).",
        ha="center", fontsize=9, color="#665A70",
    )
    fig.tight_layout(rect=(0, .04, 1, .95))
    return plot_df, fig


def plot_mutation_contacts(
    comparisons: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    wt_residue: str,
    mutant_residue: str,
    mutant_label: str,
    colors: Mapping[str, str],
    top_n: int = 10,
    channel: str = "Cav1.2",
):
    tables = [
        mutation_contact_table(wt, mutant, wt_residue, mutant_residue, protocol)
        for protocol, (wt, mutant) in comparisons.items()
    ]
    table = pd.concat(tables, ignore_index=True)
    ranking = (
        table.groupby("Partner")["Mutant contact ≤4 Å"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    plot_df = table[table["Partner"].isin(ranking)].copy()
    plot_df["Display partner"] = plot_df["Partner"].map(
        lambda value: _display_residue(value, channel)
    )
    display_order = [_display_residue(value, channel) for value in ranking]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    sns.barplot(
        data=plot_df,
        x="Display partner",
        y="Mutant contact ≤4 Å",
        hue="Protocol",
        order=display_order,
        palette=colors,
        ax=ax,
    )
    ax.set_ylabel("Fraction of models with shortest distance ≤4 Å")
    ax.set_xlabel(f"Partner of {_display_residue(mutant_residue, channel)}")
    ax.set_title(
        format_channel_title(
            f"{channel} | {mutant_label} | mutation-side-chain contacts"
        )
    )
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", color="#E9ECEF", linewidth=0.45, linestyle="--")
    sns.despine(ax=ax)
    fig.tight_layout()
    return table.sort_values(
        ["Mutant contact ≤4 Å", "Mutant median (Å)"], ascending=[False, True]
    ), fig


def plot_selected_contact_distributions(
    comparisons: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]],
    wt_residue: str,
    mutant_residue: str,
    mutant_label: str,
    partners: list[str],
    colors: Mapping[str, tuple[str, str]],
    channel: str = "Cav1.2",
    show_overlap_region: bool = True,
    experimental_distances: pd.DataFrame | None = None,
):
    """Show WT and mutant shortest-distance distributions for selected partners."""
    records = []
    for protocol, (wt, mutant) in comparisons.items():
        for partner in partners:
            columns = {
                "WT": f"shortest_{wt_residue}-{partner}",
                mutant_label: f"shortest_{mutant_residue}-{partner}",
            }
            for ensemble, column in columns.items():
                frame = wt if ensemble == "WT" else mutant
                if column not in frame:
                    continue
                records.extend(
                    {
                        "Protocol": protocol,
                        "Partner": partner,
                        "Ensemble": ensemble,
                        "Shortest distance (Å)": value,
                    }
                    for value in pd.to_numeric(frame[column], errors="coerce").dropna()
                )
    plot_df = pd.DataFrame(records)
    plot_df["Display partner"] = plot_df["Partner"].map(
        lambda value: _display_residue(value, channel)
    )
    display_partners = [_display_residue(value, channel) for value in partners]
    protocols = list(comparisons)
    fig, axes = plt.subplots(
        1, len(protocols), figsize=(5.4 * len(protocols), 5.5),
        sharey=True, squeeze=False,
    )
    for index, protocol in enumerate(protocols):
        ax = axes[0, index]
        part = plot_df[plot_df["Protocol"].eq(protocol)]
        sns.violinplot(
            data=part,
            x="Display partner",
            y="Shortest distance (Å)",
            hue="Ensemble",
            order=display_partners,
            hue_order=["WT", mutant_label],
            split=True,
            inner="quartile",
            cut=0,
            linewidth=0.6,
            palette={"WT": colors[protocol][0], mutant_label: colors[protocol][1]},
            ax=ax,
        )
        ax.axhline(4, color="#7B6D86", linewidth=0.8, linestyle="--")
        if show_overlap_region:
            ax.axhline(2, color="#C44E52", linewidth=0.9, linestyle=":")
            ax.axhspan(0, 2, color="#F4D6D7", alpha=0.35, zorder=0)
        if experimental_distances is not None and not experimental_distances.empty:
            structure_order = [
                structure for structure in ("8SD3", "8SDA")
                if structure in set(experimental_distances["Structure"])
            ]
            offsets = np.linspace(-.11, .11, max(len(structure_order), 1))
            for structure, offset in zip(structure_order, offsets):
                style = RMSD_REFERENCE_STYLES[structure]
                subset = experimental_distances[
                    experimental_distances["Structure"].eq(structure)
                ]
                for _, row in subset.iterrows():
                    display_partner = _display_residue(row["Partner"], channel)
                    if display_partner not in display_partners:
                        continue
                    ax.scatter(
                        display_partners.index(display_partner) + offset,
                        row["Shortest distance (Å)"],
                        s=32, marker=style["marker"],
                        facecolor="white", edgecolor=style["color"],
                        linewidth=1.15, zorder=6,
                    )
        ax.set_title(protocol)
        ax.set_xlabel("Interaction partner")
        ax.tick_params(axis="x", rotation=35)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.grid(axis="y", color="#E9ECEF", linewidth=0.45, linestyle="--")
        sns.despine(ax=ax)
    fig.suptitle(
        format_channel_title(
            f"{channel} | {mutant_label} | mutation-site shortest distances"
        ),
        fontweight="bold",
    )
    legend_handles = []
    for protocol in protocols:
        wt_color, mutant_color = colors[protocol]
        legend_handles.extend([
            Patch(
                facecolor=wt_color, edgecolor="#4E5B52", linewidth=.65,
                label=f"WT | {protocol}",
            ),
            Patch(
                facecolor=mutant_color, edgecolor="#4E5B52", linewidth=.65,
                label=f"{mutant_label} | {protocol}",
            ),
        ])
    if experimental_distances is not None and not experimental_distances.empty:
        for structure, description in (
            ("8SD3", "WT L403 experimental"),
            ("8SDA", "L403A A403 experimental"),
        ):
            if structure not in set(experimental_distances["Structure"]):
                continue
            style = RMSD_REFERENCE_STYLES[structure]
            legend_handles.append(
                Line2D(
                    [0], [0], linestyle="", marker=style["marker"],
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.15, markersize=6,
                    label=f"{structure} | {description}",
                )
            )
    fig.legend(
        handles=legend_handles,
        title="Prediction ensembles and experimental structures",
        loc="upper center", bbox_to_anchor=(.5, .925),
        ncol=min(4, len(legend_handles)), frameon=False,
    )
    wt_display = _display_residue_with_chain(wt_residue, channel)
    mutant_display = _display_residue_with_chain(mutant_residue, channel)
    note = (
        f"Left violin half: WT {wt_display}→partner; right half: "
        f"{mutant_label} {mutant_display}→partner. "
        "Dashed line: 4 Å proximity guide."
    )
    if show_overlap_region:
        note += " Red region: probable atomic overlap (<2 Å)."
    fig.text(0.5, 0.01, note, ha="center", fontsize=9, color="#665A70")
    fig.tight_layout(rect=(0, 0.04, 1, 0.84))
    return plot_df, fig
