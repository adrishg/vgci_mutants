#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser(description="Select a small balanced CaV1.2 pilot manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models-per-dataset", type=int, default=2)
    args = parser.parse_args()
    frame = pd.read_csv(args.input)
    pilot = frame.groupby("dataset", sort=True, group_keys=False).head(args.models_per_dataset).copy()
    pilot.insert(0, "source_manifest_index", pilot["manifest_index"].astype(int))
    pilot["manifest_index"] = range(len(pilot))
    pilot.to_csv(args.output, index=False)
    print(f"Wrote {len(pilot)} pilot rows to {args.output}")

if __name__ == "__main__":
    main()
