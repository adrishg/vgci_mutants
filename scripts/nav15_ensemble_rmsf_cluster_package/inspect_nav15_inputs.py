#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from nav15_ensemble_rmsf.core import load_config
from nav15_ensemble_rmsf.workflow import inspect_inputs

def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect Nav1.5 AlphaFold ensemble inputs and build the complete model manifest.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-models-per-dataset", type=int, default=3)
    args = parser.parse_args()
    outputs = inspect_inputs(load_config(args.config), args.output_dir, args.max_models_per_dataset)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")

if __name__ == "__main__":
    main()
