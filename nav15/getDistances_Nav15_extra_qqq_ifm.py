#!/usr/bin/env python3
"""Calculate sequence-verified Nav1.5 IFM-to-QQQ latching distances."""

import argparse
import os
import sys

import pandas as pd


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--folder-path", required=True, help="Folder containing QQQ PDB files.")
parser.add_argument("--output-csv", required=True, help="Output CSV path.")
parser.add_argument("--chain", default="A", help="Target chain ID; use '' for all chains.")
args = parser.parse_args()

folder_path = args.folder_path
output_csv = args.output_csv
target_chain_id = args.chain if args.chain != "" else None

repo_root_path = "/quobyte/yarovoygrp/ahgz/scripts/channelStatesAnalysis_AF2"
sys.path.insert(0, repo_root_path)

from stateAnalysis_tools.stateAnalysis_tools import (  # noqa: E402
    check_residue_atom_existence,
    measure_ca_distances,
    measure_shortest_distance,
)


# Sequence-verified construct mapping:
# canonical Q1486/N1659/N1765 -> model Q1170/N1343/N1449.
residue_pairs_shortest = [
    ("GLN", 1170, "ASN", 1343),  # IFM1
    ("GLN", 1170, "ASN", 1449),  # IFM2
]
residue_pairs_ca = residue_pairs_shortest

print(f"--- Starting QQQ PDB analysis from: {folder_path} ---")
if not os.path.isdir(folder_path):
    print(f"Error: folder does not exist: {folder_path}")
    sys.exit(1)

pdb_files = sorted(name for name in os.listdir(folder_path) if name.endswith(".pdb"))
if not pdb_files:
    print(f"Error: no PDB files found in: {folder_path}")
    sys.exit(1)

representative = os.path.join(folder_path, pdb_files[0])
all_pairs = residue_pairs_shortest + residue_pairs_ca
if not check_residue_atom_existence(representative, all_pairs, chain=target_chain_id):
    print("Critical error: residue/atom existence check failed.")
    sys.exit(1)

data = []
for pdb_file_name in pdb_files:
    pdb_path = os.path.join(folder_path, pdb_file_name)
    print(f"Calculating QQQ IFM distances: {pdb_file_name}")
    row = {"pdb_file": pdb_file_name}

    for residue1_type, residue1_number, residue2_type, residue2_number in residue_pairs_shortest:
        result = measure_shortest_distance(
            pdb_path,
            residue1_type,
            residue1_number,
            residue2_type,
            residue2_number,
            chain=target_chain_id,
        )
        column = (
            f"shortest_{residue1_type}{residue1_number}-"
            f"{residue2_type}{residue2_number}"
        )
        try:
            value = float(result.split(": ")[1].replace(" Å", "")) if result else None
        except (IndexError, ValueError):
            value = None
        row[column] = round(value, 2) if value is not None else None

    ca_distances = measure_ca_distances(
        pdb_path, residue_pairs_ca, chain=target_chain_id
    )
    if ca_distances:
        for pair_label, value in ca_distances.items():
            row[f"CA_{pair_label}"] = round(value, 2)
    data.append(row)

os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
pd.DataFrame(data).to_csv(output_csv, index=False)
print(f"Results saved to {output_csv}")
