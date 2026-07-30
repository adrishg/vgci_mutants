#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small, reindexed pilot manifest from the full Kv2.1 model manifest.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--models-per-dataset", type=int, default=2)
    args = parser.parse_args()
    df = pd.read_csv(args.input)
    pilot = df.groupby("dataset", group_keys=False, sort=False).head(args.models_per_dataset).copy()
    pilot.insert(1, "source_manifest_index", pilot["manifest_index"])
    pilot["manifest_index"] = range(len(pilot))
    pilot.to_csv(args.output, index=False)
    print(f"Wrote {len(pilot)} pilot rows to {args.output}")


if __name__ == "__main__":
    main()
