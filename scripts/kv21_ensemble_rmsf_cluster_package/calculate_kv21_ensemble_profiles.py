#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kv21_ensemble_rmsf.core import load_config
from kv21_ensemble_rmsf.workflow import calculate_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate full-protein Kv2.1 ensemble RMSF and deviations to 8SD3/8SDA.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--merged-dir", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subset-name", default="all_models")
    parser.add_argument("--subset-query", default=None, help="Optional pandas query applied to alignment metadata.")
    parser.add_argument("--subset-manifest", default=None, help="Optional CSV containing model-selection flags.")
    parser.add_argument("--subset-column", default=None, help="Boolean selection column in --subset-manifest.")
    parser.add_argument("--subset-key", default="pdb_file", help="Join key shared by metadata and subset manifest.")
    parser.add_argument("--minimum-residue-coverage", type=float, default=None)
    args = parser.parse_args()
    outputs = calculate_profiles(
        load_config(args.config),
        args.merged_dir,
        args.references,
        args.annotations,
        args.output_dir,
        subset_name=args.subset_name,
        subset_query=args.subset_query,
        subset_manifest=args.subset_manifest,
        subset_column=args.subset_column,
        subset_key=args.subset_key,
        minimum_residue_coverage=args.minimum_residue_coverage,
    )
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
