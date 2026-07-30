#!/usr/bin/env python3
"""Inventory RMSF inputs and record schema classifications."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from .io import discover_rmsf_inputs, load_primary_profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    for channel in ("kv21", "nav15", "cav12"):
        tables = repo / channel / "dataRMSF" / "analysis" / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        inventory = discover_rmsf_inputs(repo, channel)
        inventory.to_csv(tables / f"{channel}_input_inventory.csv", index=False)
        profile, schema, path = load_primary_profile(repo, channel)
        datasets = (
            profile.groupby(["dataset", "sequence_condition", "protocol"])
            .agg(rows=("raw_residue_number", "size"),
                 minimum_raw_residue=("raw_residue_number", "min"),
                 maximum_raw_residue=("raw_residue_number", "max"))
            .reset_index()
        )
        datasets["selected_profile"] = str(path.relative_to(repo))
        datasets["rmsf_column"] = schema["rmsf"]
        datasets["coverage_column"] = schema["coverage"]
        datasets.to_csv(tables / f"{channel}_profile_schema.csv", index=False)
        print(channel, path.relative_to(repo), schema)


if __name__ == "__main__":
    main()
