#!/usr/bin/env python3
"""QC-adjusted focal target yields over nominal five-model trajectories."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.statistics_revision.scripts.run_paired_seed_v2 import paired_scalar_rows
from analysis.statistics_revision.scripts.run_seed_block_revision import (
    FINAL_DISTANCE_PATHS, L403_COLUMNS, add_metadata, derive_l403a_threshold, read_csv,
)


BASE_SEED = 20260824


def nominal_outcomes(frame: pd.DataFrame, target: str) -> dict[str, pd.DataFrame]:
    ordered = frame.sort_values(["seed", "model_number", "recycle_number", "pdb_file"])
    grouped = ordered.groupby(["seed", "model_number"])[target]
    summaries = {
        "fraction_retained_snapshots_target": grouped.mean(),
        "any_retained_snapshot_target": grouped.max(),
        "latest_retained_snapshot_target": grouped.last(),
    }
    seeds = sorted(frame.seed.unique())
    index = pd.MultiIndex.from_product([seeds, [1, 2, 3, 4, 5]], names=["seed", "model_number"])
    return {
        name: values.reindex(index, fill_value=0).rename("target").reset_index()
        for name, values in summaries.items()
    }


def append_contrasts(rows, a, b, prefix, bootstrap, seed):
    for index, definition in enumerate(a):
        rows.extend(paired_scalar_rows(
            a[definition], b[definition], "target", contrast=f"{prefix} masked - vanilla",
            outcome=f"QC_adjusted_{definition}", bootstrap=bootstrap, seed=seed + index * 10,
        ))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    parser.add_argument("--mode", choices=("exploratory", "publication"), default="publication")
    args = parser.parse_args()
    bootstrap = 250 if args.mode == "exploratory" else 2000
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    rows = []

    threshold, _, _ = derive_l403a_threshold(output)
    l403 = add_metadata(read_csv(ROOT / "kv21/dataDistances/analysis/L403A_E423_N179_all_structure_distances.csv"))
    l403[L403_COLUMNS] = l403[L403_COLUMNS].apply(pd.to_numeric, errors="coerce")
    l403["target"] = l403[L403_COLUMNS].ge(threshold).any(axis=1).astype(float)
    append_contrasts(
        rows, nominal_outcomes(l403[l403.condition.eq("vanilla")], "target"),
        nominal_outcomes(l403[l403.condition.eq("masked")], "target"),
        "Kv2.1 L403A experiment-anchored shifted interface", bootstrap, BASE_SEED + 10000,
    )

    contact_path = ROOT / "kv21/dataRMSD/analysis/comparison_v5/f412l_pocket_D_paper_nexus_shortest_contacts_long_v5.csv"
    contact = read_csv(contact_path)
    value = "Shortest heavy-atom distance (Å)"
    wide = contact.pivot_table(index=["Protocol", "pdb_file"], columns="Contact", values=value, aggfunc="first").reset_index()
    wide = add_metadata(wide)
    for contact_name in [column for column in wide if "L412" in str(column)]:
        wide["target"] = pd.to_numeric(wide[contact_name], errors="coerce").le(4).astype(float)
        append_contrasts(
            rows, nominal_outcomes(wide[wide.Protocol.eq("Vanilla")], "target"),
            nominal_outcomes(wide[wide.Protocol.eq("Masked")], "target"),
            f"Kv2.1 F412L {str(contact_name).replace(chr(10), ' ')} within-4-A proximity",
            bootstrap, BASE_SEED + 11000 + len(rows),
        )

    g406 = {
        protocol: add_metadata(read_csv(ROOT / FINAL_DISTANCE_PATHS[f"cav12|G406R|{protocol}"]))
        for protocol in ("vanilla", "masked")
    }
    centered = [column for column in g406["vanilla"] if column.startswith("shortest_ARG406-")]
    for protocol, frame in g406.items():
        distances = frame[centered].apply(pd.to_numeric, errors="coerce")
        frame["overlap_pass"] = (~distances.lt(2).any(axis=1)).astype(float)
    for target_name in ("overlap_pass", "ASP1528", "ASP1533"):
        for protocol, frame in g406.items():
            if target_name == "overlap_pass":
                frame["target"] = frame.overlap_pass
            else:
                proximity = pd.to_numeric(frame[f"shortest_ARG406-{target_name}"], errors="coerce").le(4)
                frame["target"] = (frame.overlap_pass.eq(1) & proximity).astype(float)
        append_contrasts(
            rows, nominal_outcomes(g406["vanilla"], "target"),
            nominal_outcomes(g406["masked"], "target"),
            f"Cav1.2 G406R {target_name}", bootstrap, BASE_SEED + 12000 + len(rows),
        )

    table = pd.DataFrame(rows)
    table["nominal_denominator"] = "100 recorded seed labels x five AF2 model strata per condition; missing/failed trajectories contribute zero"
    table["frequency_interpretation"] = "QC-adjusted protocol sampling yield, not thermodynamic occupancy"
    table.to_csv(output / "usable_target_geometry_yields.csv", index=False)


if __name__ == "__main__":
    main()
