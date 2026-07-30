"""Experimental-coordinate calculations used by Kv2.1 notebooks."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
import tempfile
from urllib.request import urlretrieve

from Bio.PDB import MMCIFParser, PDBParser
import numpy as np
import pandas as pd


def experimental_overlay(
    region,
    *,
    experimental_ca,
    pdb_order,
    colors,
    alias_maps,
):
    """Build experimental marker dictionaries for one structural region."""
    alias_map = alias_maps[region]
    distances = [
        {
            alias_map[alias]: values
            for alias, values in experimental_ca[pdb_id][region].items()
            if alias in alias_map
        }
        for pdb_id in pdb_order
    ]
    labels = [
        f"Experimental | {pdb_id}: {experimental_ca[pdb_id]['state']}"
        for pdb_id in pdb_order
    ]
    return distances, labels, colors


def experimental_coordinate_path(pdb_id, *, repo_root):
    """Use a local experimental PDB, otherwise cache the official RCSB mmCIF."""
    local_pdb = Path(repo_root) / "kv21" / "experimental" / f"{pdb_id}.pdb"
    if local_pdb.exists():
        return local_pdb
    cache_dir = Path(tempfile.gettempdir()) / "kv21_experimental_coordinates"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_cif = cache_dir / f"{pdb_id}.cif"
    if not cached_cif.exists():
        urlretrieve(f"https://files.rcsb.org/download/{pdb_id}.cif", cached_cif)
    return cached_cif


def coordinate_model(pdb_id, *, repo_root):
    path = experimental_coordinate_path(pdb_id, repo_root=repo_root)
    parser = (
        PDBParser(QUIET=True)
        if path.suffix == ".pdb"
        else MMCIFParser(QUIET=True, auth_chains=True, auth_residues=True)
    )
    return parser.get_structure(pdb_id, path)[0]


def ca_coordinate(model, chain_id, residue_number):
    return model[chain_id][(" ", residue_number, " ")]["CA"].coord


def ca_distance(coordinate_a, coordinate_b):
    return [round(float(np.linalg.norm(coordinate_a - coordinate_b)), 3)]


def calculate_a404_ring_distances(
    pdb_id, *, repo_root, experimental_ca
):
    model = coordinate_model(pdb_id, repo_root=repo_root)
    residue_number = experimental_ca[pdb_id]["mapping"]["CSV_A404"]
    coordinates = {
        chain: ca_coordinate(model, chain, residue_number) for chain in "ABCD"
    }
    return {
        f"{chain_a}_A404-{chain_b}_A404": ca_distance(
            coordinates[chain_a], coordinates[chain_b]
        )
        for chain_a, chain_b in combinations("ABCD", 2)
    }


def calculate_all_plotted_experimental_distances(
    pdb_id, *, repo_root, experimental_ca
):
    """Calculate filter, gate, and VSD distances from one experimental model."""
    model = coordinate_model(pdb_id, repo_root=repo_root)
    mapping = experimental_ca[pdb_id]["mapping"]

    filter_coordinates = {
        chain: ca_coordinate(model, chain, mapping["CSV_G377"])
        for chain in "ABCD"
    }
    selectivity_filter = {}
    for chain_a, chain_b in combinations("ABCD", 2):
        left_alias = f"G377{chain_a}" if chain_a == "C" else f"G377_{chain_a}"
        selectivity_filter[f"{left_alias}-G377{chain_b}"] = ca_distance(
            filter_coordinates[chain_a], filter_coordinates[chain_b]
        )

    gate_coordinates = {
        chain: ca_coordinate(model, chain, mapping["CSV_A404"])
        for chain in "ABCD"
    }
    intracellular_gate = {
        f"{chain_a}_A404-{chain_b}_A404": ca_distance(
            gate_coordinates[chain_a], gate_coordinates[chain_b]
        )
        for chain_a, chain_b in combinations("ABCD", 2)
    }

    voltage_sensor = {}
    for chain in "ABCD":
        phe = ca_coordinate(model, chain, mapping["CSV_F238"])
        voltage_sensor[f"{chain}_F238-{chain}_R291"] = ca_distance(
            phe, ca_coordinate(model, chain, mapping["CSV_R291"])
        )
        voltage_sensor[f"{chain}_F238-{chain}_R310"] = ca_distance(
            phe, ca_coordinate(model, chain, mapping["CSV_R310"])
        )
    return {
        "selectivity_filter": selectivity_filter,
        "intracellular_gate": intracellular_gate,
        "voltage_sensor": voltage_sensor,
    }


def calculate_s6_experimental_profile(
    pdb_id, *, repo_root, s6_ring_residues
):
    """Calculate maximum four-chain Cα ring spans along experimental S6."""
    model = coordinate_model(pdb_id, repo_root=repo_root)
    residue_offset = -2 if pdb_id in {"8SD3", "8SDA"} else 2
    profile = {}
    for label, (_, csv_residue_number) in s6_ring_residues.items():
        pdb_residue_number = csv_residue_number + residue_offset
        coordinates = [
            ca_coordinate(model, chain, pdb_residue_number) for chain in "ABCD"
        ]
        distances = [
            float(np.linalg.norm(coordinates[first] - coordinates[second]))
            for first, second in combinations(range(4), 2)
        ]
        profile[label] = [round(max(distances), 3)]
    return profile


def distance_shift_table(wt, mutant, mutant_name, protocol):
    """Rank all shared ensemble columns by robust median displacement."""
    common = sorted((set(wt.columns) & set(mutant.columns)) - {"pdb_file"})
    wt_numeric = wt[common].apply(pd.to_numeric, errors="coerce")
    mutant_numeric = mutant[common].apply(pd.to_numeric, errors="coerce")
    wt_median, mutant_median = wt_numeric.median(), mutant_numeric.median()
    wt_iqr = wt_numeric.quantile(0.75) - wt_numeric.quantile(0.25)
    mutant_iqr = mutant_numeric.quantile(0.75) - mutant_numeric.quantile(0.25)
    pooled_iqr = (wt_iqr + mutant_iqr) / 2
    result = pd.DataFrame(
        {
            "distance": common,
            "wt_median_A": wt_median.values,
            "mutant_median_A": mutant_median.values,
            "median_shift_A": (mutant_median - wt_median).values,
            "wt_iqr_A": wt_iqr.values,
            "mutant_iqr_A": mutant_iqr.values,
            "robust_shift_iqr": (
                (mutant_median - wt_median) / pooled_iqr.replace(0, pd.NA)
            ).values,
            "mutant": mutant_name,
            "protocol": protocol,
        }
    )
    return result.sort_values(
        "median_shift_A", key=lambda values: values.abs(), ascending=False
    )
