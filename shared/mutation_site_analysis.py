"""Reusable nearby-mutation distance analysis for channel ensemble notebooks."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from Bio.PDB import PDBParser


# Explicit distances calculated from the downloaded Cav1.2 α-subunits. Values are Å.
CAV12_MUTANT_EXPERIMENTAL_DISTANCES = {
    "G402S": {
        "8WE6 | chain A": {"CA_LEU401_CA-ILE1186_CA":[10.070],"CA_LEU401_CA-VAL753_CA":[11.718],"GLY406-MET1524":[7.404],"LEU401-ALA1521":[9.341],"LEU401-ASN1526":[9.824],"LEU401-ASP1525":[10.976],"LEU401-ASP1528":[11.996],"LEU401-MET1524":[8.529],"LEU401-TYR1529":[10.879],"LEU401-VAL1522":[8.829],"LEU404-MET1524":[9.680],"SER405-ASP1525":[9.391],"SER405-ASP1533":[11.521],"SER405-MET1524":[6.077],"SER405-TYR1529":[9.355],"VAL400-ASP1528":[13.981],"VAL400-MET1524":[11.185],"VAL403-ASP1528":[10.703],"VAL403-MET1524":[9.090]},
        "8HLP | chain A": {"CA_LEU401_CA-ILE1186_CA":[9.986],"CA_LEU401_CA-VAL753_CA":[11.751],"GLY406-MET1524":[8.961],"LEU401-ALA1521":[9.471],"LEU401-ASN1526":[9.889],"LEU401-ASP1525":[11.505],"LEU401-ASP1528":[11.921],"LEU401-MET1524":[9.147],"LEU401-TYR1529":[10.943],"LEU401-VAL1522":[8.975],"LEU404-MET1524":[11.400],"SER405-ASP1525":[9.994],"SER405-ASP1533":[11.415],"SER405-MET1524":[7.551],"SER405-TYR1529":[9.544],"VAL400-ASP1528":[14.009],"VAL400-MET1524":[11.875],"VAL403-ASP1528":[10.726],"VAL403-MET1524":[9.957]},
        "8FD7 | chain K": {"CA_LEU401_CA-ILE1186_CA":[13.653],"CA_LEU401_CA-VAL753_CA":[15.182],"GLY406-MET1524":[13.416],"LEU401-ALA1521":[7.614],"LEU401-ASN1526":[3.233],"LEU401-ASP1525":[7.669],"LEU401-ASP1528":[8.778],"LEU401-MET1524":[6.187],"LEU401-TYR1529":[8.138],"LEU401-VAL1522":[5.311],"LEU404-MET1524":[8.234],"SER405-ASP1525":[11.258],"SER405-ASP1533":[7.884],"SER405-MET1524":[10.802],"SER405-TYR1529":[6.804],"VAL400-ASP1528":[13.343],"VAL400-MET1524":[11.054],"VAL403-ASP1528":[13.659],"VAL403-MET1524":[13.091]},
    },
    "G406R": {
        "8WE6 | chain A": {"CA_LEU401_CA-ILE1186_CA":[10.070],"CA_LEU401_CA-VAL753_CA":[11.718],"CA_GLU407_CA-ARG1532_CA":[9.716],"CA_GLU407_CA-LEU1530_CA":[7.270],"CA_GLU407_CA-THR1531_CA":[6.724],"CA_PHE408_CA-ARG1532_CA":[12.668],"CA_PHE408_CA-THR1531_CA":[9.273],"GLU407-ARG1532":[8.011],"GLU407-ASP1533":[9.304],"GLU407-TYR1529":[8.236],"LEU404-ARG1532":[11.220],"PHE408-ARG1532":[10.269],"PHE408-TRP1534":[13.153],"SER405-ARG1532":[9.744]},
        "8HLP | chain A": {"CA_LEU401_CA-ILE1186_CA":[9.986],"CA_LEU401_CA-VAL753_CA":[11.751],"CA_GLU407_CA-ARG1532_CA":[8.687],"CA_GLU407_CA-LEU1530_CA":[7.352],"CA_GLU407_CA-THR1531_CA":[6.392],"CA_PHE408_CA-ARG1532_CA":[11.595],"CA_PHE408_CA-THR1531_CA":[8.828],"GLU407-ARG1532":[7.179],"GLU407-ASP1533":[8.362],"GLU407-TYR1529":[7.913],"LEU404-ARG1532":[10.248],"PHE408-ARG1532":[9.330],"PHE408-TRP1534":[12.814],"SER405-ARG1532":[9.122]},
        "8FD7 | chain K": {"CA_LEU401_CA-ILE1186_CA":[13.653],"CA_LEU401_CA-VAL753_CA":[15.182],"CA_GLU407_CA-ARG1532_CA":[11.739],"CA_GLU407_CA-LEU1530_CA":[10.872],"CA_GLU407_CA-THR1531_CA":[8.756],"CA_PHE408_CA-ARG1532_CA":[8.953],"CA_PHE408_CA-THR1531_CA":[6.103],"GLU407-ARG1532":[9.949],"GLU407-ASP1533":[10.982],"GLU407-TYR1529":[11.953],"LEU404-ARG1532":[8.867],"PHE408-ARG1532":[5.747],"PHE408-TRP1534":[10.327],"SER405-ARG1532":[6.412]},
    },
}


def explicit_experimental_subset(mutant_label: str, aliases: Mapping[str, str]):
    """Return only explicit experimental values matching the currently plotted aliases."""
    return {
        structure: {alias: values for alias, values in distances.items() if alias in aliases}
        for structure, distances in CAV12_MUTANT_EXPERIMENTAL_DISTANCES[mutant_label].items()
    }


def load_ensemble(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def nearby_distance_columns(df: pd.DataFrame, residue: int, radius: int = 4) -> list[str]:
    """Return numeric distance columns mentioning a residue within the requested window."""
    window = set(range(residue - radius, residue + radius + 1))
    columns = []
    for column in df.columns:
        residue_numbers = {int(value) for value in re.findall(r"\d+", column)}
        if residue_numbers & window and column != "pdb_file":
            columns.append(column)
    return columns


def rank_nearby_shifts(wt: pd.DataFrame, mutant: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    common = [column for column in columns if column in wt.columns and column in mutant.columns]
    wt_values = wt[common].apply(pd.to_numeric, errors="coerce")
    mutant_values = mutant[common].apply(pd.to_numeric, errors="coerce")
    result = pd.DataFrame({
        "distance": common,
        "wt_median_A": wt_values.median().reindex(common).values,
        "mutant_median_A": mutant_values.median().reindex(common).values,
    })
    result["median_shift_A"] = result["mutant_median_A"] - result["wt_median_A"]
    result["abs_shift_A"] = result["median_shift_A"].abs()
    # A split violin requires observations from both ensembles. Dropping
    # incomplete pairs prevents a WT-only half violin from being presented as
    # a WT-versus-mutant comparison when a CSV block is entirely empty.
    return result.dropna(subset=["wt_median_A", "mutant_median_A"]).sort_values(
        "abs_shift_A", ascending=False
    )


def top_aliases(
    shift_table: pd.DataFrame, n: int = 12, residue_number_offset: int = 0
) -> Mapping[str, str]:
    """Return display aliases mapped to untouched CSV columns.

    ``residue_number_offset`` changes only visible residue labels; it is -2 for
    Kv2.1 model/CSV columns displayed in rat experimental/paper numbering.
    """
    selected = shift_table.head(n)["distance"].tolist()
    aliases = {}
    for column in selected:
        label = column.replace("CA_CA_", "").replace("shortest_", "")
        if residue_number_offset:
            label = re.sub(
                r"(?<=[A-Z]{3})(\d+)",
                lambda match: str(int(match.group(1)) + residue_number_offset),
                label,
            )
        aliases[label] = column
    return aliases


def _residue_tokens(column: str) -> list[tuple[str, int]]:
    return [(name, int(number)) for name, number in re.findall(r"([A-Z]{3})(\d+)", column)]


def cav12_experimental_distances(
    aliases: Mapping[str, str],
    structures: Mapping[str, tuple[str | Path, str]],
) -> dict[str, dict[str, list[float]]]:
    """Calculate Cav1.2 CSV-matched distances in explicitly selected α-subunit chains."""
    parser = PDBParser(QUIET=True)
    result = {}
    for structure_label, (path, chain_id) in structures.items():
        chain = parser.get_structure(structure_label, path)[0][chain_id]
        distances = {}
        for alias, column in aliases.items():
            tokens = _residue_tokens(column)
            if len(tokens) != 2:
                continue
            (_, residue_a), (_, residue_b) = tokens
            if residue_a not in chain or residue_b not in chain:
                continue
            first, second = chain[residue_a], chain[residue_b]
            if column.startswith("CA_"):
                if "CA" not in first or "CA" not in second:
                    continue
                value = np.linalg.norm(first["CA"].coord - second["CA"].coord)
            elif column.startswith("shortest_"):
                atoms_a = [atom for atom in first.get_atoms() if atom.element != "H"]
                atoms_b = [atom for atom in second.get_atoms() if atom.element != "H"]
                value = min(np.linalg.norm(atom_a.coord - atom_b.coord) for atom_a in atoms_a for atom_b in atoms_b)
            else:
                continue
            distances[alias] = [round(float(value), 3)]
        result[structure_label] = distances
    return result


def plot_nearby_overlay(
    wt: pd.DataFrame,
    mutant: pd.DataFrame,
    aliases: Mapping[str, str],
    channel: str,
    mutant_label: str,
    protocol: str,
    colors: tuple[str, str],
    experimental_distances: Mapping[str, Mapping[str, list[float]]] | None = None,
    experimental_colors: Mapping[str, str] | None = None,
):
    records = []
    for ensemble, frame in (("WT", wt), (mutant_label, mutant)):
        for alias, column in aliases.items():
            records.extend(
                {"Distance alias": alias, "Distance (Å)": value, "Ensemble": ensemble}
                for value in pd.to_numeric(frame[column], errors="coerce").dropna()
            )
    fig, ax = plt.subplots(figsize=(10.5, 8.0))
    sns.violinplot(
        data=pd.DataFrame(records), x="Distance alias", y="Distance (Å)", hue="Ensemble",
        order=list(aliases), hue_order=["WT", mutant_label], split=True,
        palette={"WT": colors[0], mutant_label: colors[1]}, inner="quartile",
        cut=0, linewidth=0.6, saturation=0.82, ax=ax,
    )
    if experimental_distances:
        marker_colors = experimental_colors or {}
        for structure_index, (structure, distances) in enumerate(experimental_distances.items()):
            used_label = False
            offset = (structure_index - (len(experimental_distances) - 1) / 2) * 0.08
            for alias, values in distances.items():
                if alias not in aliases:
                    continue
                xpos = list(aliases).index(alias) + offset
                for value in values:
                    ax.scatter(
                        xpos, value, marker=("o", "s", "D", "^", "v", "P", "X", "*")[structure_index % 8],
                        s=34, facecolors="white",
                        edgecolors=marker_colors.get(structure, "#E57373"), linewidths=0.8,
                        zorder=6, label=f"Experimental | {structure}" if not used_label else None,
                    )
                    used_label = True
    experimental_suffix = " | experimental distances" if experimental_distances else ""
    ax.set_title(f"{channel} | WT vs {mutant_label} | {protocol} | nearby mutation site | split violin{experimental_suffix}")
    ax.tick_params(axis="x", rotation=55)
    ax.legend(title="Ensembles and experimental structures", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", color="#EDF5EF", linestyle="--", linewidth=0.4)
    sns.despine(ax=ax)
    fig.tight_layout()
    plt.show()
    return ax
