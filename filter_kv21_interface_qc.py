#!/usr/bin/env python3
"""Persist the trajectory-level Kv2.1 pore–VSD interface quality check."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from shared.dataset_selection import apply_kv21_interface_qc


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "kv21" / "dataDistances"
SOURCE_SUFFIX = "_all_ok_rmsd_3A_structural_qc.csv"
OUTPUT_SUFFIX = "_all_ok_rmsd_3A_structural_interface_qc.csv"


def main() -> None:
    sources = sorted(DATA_DIR.glob(f"26-02-11_Kv2.1_*{SOURCE_SUFFIX}"))
    if len(sources) != 6:
        raise RuntimeError(f"Expected six Kv2.1 structural-QC inputs; found {len(sources)}")

    for source in sources:
        frame = pd.read_csv(source)
        frame.attrs["dataset_label"] = source.stem
        filtered = apply_kv21_interface_qc(frame, threshold_A=27.0, trajectory_level=True)
        output = source.with_name(source.name.replace(SOURCE_SUFFIX, OUTPUT_SUFFIX))
        filtered.to_csv(output, index=False)
        print(f"Wrote {len(filtered)} rows: {output}")


if __name__ == "__main__":
    main()
