#!/usr/bin/env python3
"""Carry the established Nav1.5 3 Å selections into the expanded distance tables."""

from __future__ import annotations

import csv
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "nav15" / "dataDistances"

DATASETS = {
    "WT vanilla": (
        "26-07-27_Nav15_wt_vanillaAF2_distances_all_all.csv",
        "25-12-02_Nav15_wt_vanillaAF2_distances_all_ok_rmsd_3A.csv",
    ),
    "WT masked v2": (
        "26-07-27_Nav15_wt_maskedv2_AF2_distances_all_all.csv",
        "26-07-25_Nav15_wt_maskedv2_AF2_distances_all_ok_rmsd_3A.csv",
    ),
    "WT masked v2 no-IFM": (
        "26-07-27_Nav15_wt_maskedv2_noIFM_AF2_distances_all_all.csv",
        "26-07-25_Nav15_wt_maskedv2_noIFM_AF2_distances_all_ok_rmsd_3A.csv",
    ),
    "QQQ vanilla": (
        "26-07-27_Nav15_qqq_vanilla_AF2_distances_all_all.csv",
        "26-02-01_Nav15_qqq_vanilla_distances_all_ok_rmsd_3A.csv",
    ),
    "QQQ masked": (
        "26-07-27_Nav15_qqq_masked_AF2_distances_all_all.csv",
        "26-07-24_Nav15_qqq_maskedAF2_distances_all_ok_rmsd_3A.csv",
    ),
    "QQQ masked v2": (
        "26-07-27_Nav15_qqq_maskedv2_AF2_distances_all_all.csv",
        "26-07-25_Nav15_qqq_maskedAF2v2_distances_all_ok_rmsd_3A.csv",
    ),
}


def pdb_basename(row: dict[str, str]) -> str:
    return Path(row["pdb_file"].strip()).name


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        if "pdb_file" not in fields:
            raise KeyError(f"{path.name} has no pdb_file column")
        return fields, list(reader)


def output_path(source: Path) -> Path:
    stem = source.stem
    while stem.endswith("_all"):
        stem = stem[:-4]
    return source.with_name(f"{stem}_all_ok_rmsd_3A.csv")


def main() -> None:
    for label, (expanded_name, prior_selection_name) in DATASETS.items():
        expanded_path = DATA_DIR / expanded_name
        prior_path = DATA_DIR / prior_selection_name
        fields, expanded_rows = read_rows(expanded_path)
        _, prior_rows = read_rows(prior_path)

        selected_keys = {pdb_basename(row) for row in prior_rows}
        if len(selected_keys) != len(prior_rows):
            raise RuntimeError(f"{label}: duplicate pdb_file keys in prior 3 Å selection")

        expanded_keys = [pdb_basename(row) for row in expanded_rows]
        if len(set(expanded_keys)) != len(expanded_keys):
            raise RuntimeError(f"{label}: duplicate pdb_file keys in expanded distance table")

        missing = selected_keys.difference(expanded_keys)
        if missing:
            raise RuntimeError(f"{label}: {len(missing)} selected models absent from expanded table")

        filtered_rows = [
            row for key, row in zip(expanded_keys, expanded_rows) if key in selected_keys
        ]
        if len(filtered_rows) != len(selected_keys):
            raise RuntimeError(f"{label}: filtered row count does not match 3 Å selection")

        destination = output_path(expanded_path)
        with destination.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(filtered_rows)

        print(
            f"{label}: retained {len(filtered_rows)}/{len(expanded_rows)} rows "
            f"with {len(fields)} columns -> {destination.name}"
        )


if __name__ == "__main__":
    main()
