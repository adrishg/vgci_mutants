#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from cav12_ensemble_rmsf.core import load_config
from cav12_ensemble_rmsf.workflow import prepare_references

def main() -> None:
    parser = argparse.ArgumentParser(description="Map and align CaV1.2 experimental references into the common 8WE6 coordinate frame.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    outputs = prepare_references(load_config(args.config), args.output_dir)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")

if __name__ == "__main__":
    main()
