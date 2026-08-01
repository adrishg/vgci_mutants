#!/usr/bin/env python3
"""Add the missing Kv2.1 F236–R308 chain-resolved Cα distances.

The historical WT-vanilla distance table contains the chain-A measurement but
not the equivalent B–D columns. This utility reads only the PDBs represented
in a final-QC CSV and appends the genuine chain-resolved measurements without
changing the retained model set.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RESIDUES = {
    "first_number": 238,
    "first_name": "PHE",
    "second_number": 310,
    "second_name": "ARG",
}
CHAINS = ("A", "B", "C", "D")


def column_name(chain: str) -> str:
    return f"CA_CA_{chain}_ARG310_CA-{chain}_PHE238_CA"


def build_pdb_index(pdb_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in pdb_root.rglob("*.pdb"):
        if path.name in index:
            duplicates.add(path.name)
        else:
            index[path.name] = path
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise RuntimeError(f"Duplicate PDB basenames under {pdb_root}: {preview}")
    return index


def ca_distances(pdb_path: Path) -> dict[str, float]:
    targets = {
        (chain, RESIDUES["first_number"]): RESIDUES["first_name"]
        for chain in CHAINS
    }
    targets.update({
        (chain, RESIDUES["second_number"]): RESIDUES["second_name"]
        for chain in CHAINS
    })
    coordinates: dict[tuple[str, int], np.ndarray] = {}
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM  ") or line[12:16].strip() != "CA":
                continue
            if line[16] not in (" ", "A"):
                continue
            chain = line[21]
            try:
                residue_number = int(line[22:26])
            except ValueError:
                continue
            key = (chain, residue_number)
            if key not in targets:
                continue
            residue_name = line[17:20].strip()
            if residue_name != targets[key]:
                raise ValueError(
                    f"{pdb_path.name} chain {chain}: residue "
                    f"{residue_number} is {residue_name}, "
                    f"expected {targets[key]}"
                )
            coordinates[key] = np.array([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54]),
            ])
            if len(coordinates) == len(targets):
                break

    missing = sorted(set(targets) - set(coordinates))
    if missing:
        raise KeyError(f"{pdb_path.name}: missing Cα coordinates {missing}")

    return {
        chain: float(np.linalg.norm(
            coordinates[(chain, RESIDUES["first_number"])]
            - coordinates[(chain, RESIDUES["second_number"])]
        ))
        for chain in CHAINS
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--pdb-root",
        required=True,
        type=Path,
        help="Root containing the Kv2.1 WT-vanilla AlphaFold PDB ensemble.",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    table = pd.read_csv(args.input_csv)
    if "pdb_file" not in table:
        raise KeyError(f"{args.input_csv} has no pdb_file column")

    pdb_index = build_pdb_index(args.pdb_root)
    missing = sorted({
        Path(value).name
        for value in table["pdb_file"].astype(str)
        if Path(value).name not in pdb_index
    })
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"{len(missing)} final-QC PDBs were not found under "
            f"{args.pdb_root}. First missing files:\n{preview}"
        )

    measurements = [
        ca_distances(pdb_index[Path(value).name])
        for value in table["pdb_file"].astype(str)
    ]
    for chain in CHAINS:
        table[column_name(chain)] = np.round(
            [item[chain] for item in measurements], 3
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(table):,} final-QC rows to {args.output_csv}")
    for chain in CHAINS:
        series = table[column_name(chain)]
        print(
            f"{chain}: n={series.notna().sum():,}, "
            f"range={series.min():.3f}–{series.max():.3f} Å"
        )


if __name__ == "__main__":
    main()
