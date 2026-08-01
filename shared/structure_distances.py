"""Coordinate and distance utilities shared by channel-analysis notebooks."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd


def read_ca_atoms(path: str | Path, chain: str | None = None):
    """Return Cα coordinates keyed by ``(chain, residue_number)``."""
    atoms = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[12:16].strip() != "CA":
                continue
            atom_chain = line[21].strip()
            if chain is not None and atom_chain != chain:
                continue
            atoms[(atom_chain, int(line[22:26]))] = tuple(
                float(line[index:index + 8]) for index in (30, 38, 46)
            )
    return atoms


def read_ca_residues(path: str | Path, chain: str = "A"):
    """Return Cα coordinates keyed by ``(residue_name, residue_number)``."""
    atoms = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            if (
                line.startswith("ATOM")
                and line[21].strip() == chain
                and line[12:16].strip() == "CA"
            ):
                atoms[(line[17:20].strip(), int(line[22:26]))] = np.array(
                    [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                )
    return atoms


def ca_distance_map(
    path: str | Path,
    mapping: Mapping[str, tuple[str, int, str, int]],
):
    """Calculate named chain/residue Cα distances from one PDB."""
    atoms = read_ca_atoms(path)
    distances, missing = {}, []
    for label, (chain_a, residue_a, chain_b, residue_b) in mapping.items():
        first = atoms.get((chain_a, residue_a))
        second = atoms.get((chain_b, residue_b))
        if first is None or second is None:
            missing.append(label)
        else:
            distances[label] = round(math.dist(first, second), 3)
    return distances, missing


def experimental_distances_for_aliases(
    aliases: Mapping[str, str] | Sequence[str],
    pdb_path: str | Path,
    *,
    alias_number_offset: int = 0,
):
    """Calculate Cα distances for common experimental-numbered alias formats.

    Supported labels include ``K427A-N179A`` and
    ``A_PHE412-B_LEU401``. ``alias_number_offset`` is added to numbers parsed
    from the visible alias before looking them up in the PDB.
    """
    atoms = read_ca_atoms(pdb_path)
    distances, missing = {}, []
    for alias in aliases:
        residues = re.findall(r"([A-D])_([A-Z]{3})(\d+)", alias)
        if len(residues) == 2:
            first_key = (residues[0][0], int(residues[0][2]) + alias_number_offset)
            second_key = (residues[1][0], int(residues[1][2]) + alias_number_offset)
        else:
            compact = re.findall(r"([A-Z])(\d+)([A-D])", alias)
            if len(compact) != 2:
                missing.append(alias)
                continue
            first_key = (compact[0][2], int(compact[0][1]) + alias_number_offset)
            second_key = (compact[1][2], int(compact[1][1]) + alias_number_offset)
        first, second = atoms.get(first_key), atoms.get(second_key)
        if first is None or second is None:
            missing.append(alias)
        else:
            distances[alias] = [round(math.dist(first, second), 3)]
    return distances, missing


def read_residue_heavy_atoms(path: str | Path, chain: str = "A"):
    """Return non-hydrogen coordinates grouped by residue identity."""
    residues = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[21].strip() != chain:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() or atom_name[0]
            if element.upper() == "H" or line[16] not in (" ", "A"):
                continue
            key = (line[17:20].strip(), int(line[22:26]))
            residues.setdefault(key, []).append(
                np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            )
    return residues


def read_named_heavy_atoms(path: str | Path, chain: str = "A"):
    """Return non-hydrogen coordinates keyed by residue identity and atom name."""
    atoms = {}
    with open(path, errors="ignore") as handle:
        for line in handle:
            if not line.startswith("ATOM") or line[21].strip() != chain:
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() or atom_name[0]
            if element.upper() == "H" or line[16] not in (" ", "A"):
                continue
            key = (
                line[17:20].strip(),
                int(line[22:26]),
                atom_name,
            )
            atoms[key] = np.array(
                [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            )
    return atoms


def named_atom_distance(atoms, key_a, key_b) -> float:
    """Return the distance between two explicitly named residue atoms."""
    first, second = atoms.get(key_a), atoms.get(key_b)
    if first is None or second is None:
        return np.nan
    return float(np.linalg.norm(first - second))


def minimum_residue_distance(residues, keys_a, key_b) -> float:
    """Return the minimum heavy-atom distance between residue set A and B."""
    atoms_a = [coordinate for key in keys_a for coordinate in residues.get(key, [])]
    atoms_b = residues.get(key_b, [])
    if not atoms_a or not atoms_b:
        return np.nan
    first, second = np.vstack(atoms_a), np.vstack(atoms_b)
    return float(np.linalg.norm(first[:, None, :] - second[None, :, :], axis=2).min())


def distance_shift_table(
    wt: pd.DataFrame,
    mutant: pd.DataFrame,
    aliases: Mapping[str, str],
) -> pd.DataFrame:
    """Rank median mutant-minus-WT shifts for shared distance columns."""
    rows = []
    for alias, column in aliases.items():
        if column not in wt or column not in mutant:
            continue
        wt_median = pd.to_numeric(wt[column], errors="coerce").median()
        mutant_median = pd.to_numeric(mutant[column], errors="coerce").median()
        rows.append(
            {
                "distance": alias,
                "WT_median_A": wt_median,
                "mutant_median_A": mutant_median,
                "median_shift_A": mutant_median - wt_median,
            }
        )
    result = pd.DataFrame(rows).dropna()
    return result.assign(abs_shift_A=result["median_shift_A"].abs()).sort_values(
        "abs_shift_A", ascending=False
    )
