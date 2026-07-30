#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from nav15_ensemble_rmsf.core import load_config
from nav15_ensemble_rmsf.workflow import align_manifest_shard

def main() -> None:
    parser = argparse.ArgumentParser(description="Align one deterministic shard of single-chain Nav1.5 AlphaFold structures to the common 6UZ3 frame.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--task-count", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.task_id < args.task_count:
        parser.error("--task-id must satisfy 0 <= task-id < task-count")
    outputs = align_manifest_shard(load_config(args.config), args.manifest, args.references, args.output_dir, args.task_id, args.task_count, args.overwrite)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")

if __name__ == "__main__":
    main()
