#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from kv21_ensemble_rmsf.workflow import merge_shards


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge all Kv2.1 aligned-coordinate shards into memory-mapped arrays.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parts-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    outputs = merge_shards(args.manifest, args.parts_dir, args.output_dir, args.allow_missing, args.overwrite)
    for name, path in outputs.items():
        print(f"{name}: {Path(path).resolve()}")


if __name__ == "__main__":
    main()
