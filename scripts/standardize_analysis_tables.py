"""Add explicit channel, condition, and protocol metadata to analysis tables.

The analysis notebooks often encode these fields in filenames or in columns
such as ``sequence_condition``.  This utility makes the identity explicit in
every generated CSV so tables remain interpretable when copied out of their
repository folders.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


CHANNEL_LABELS = {"kv21": "Kv2.1", "nav15": "Nav1.5", "cav12": "CaV1.2"}
CONDITIONS = {
    "kv21": ("f412l", "l403a", "wt"),
    "nav15": ("qqq", "wt"),
    "cav12": ("g490r", "g406r", "g402s", "wt"),
}


def infer_condition(path: Path, frame: pd.DataFrame, channel: str) -> str:
    for column in ("condition", "sequence_condition", "sequence", "dataset"):
        if column in frame and frame[column].notna().any():
            values = frame[column].dropna().astype(str)
            if values.nunique() == 1:
                return values.iloc[0]
            return "multiple conditions"
    name = path.stem.lower()
    for condition in CONDITIONS[channel]:
        if re.search(rf"(?:^|_){re.escape(condition)}(?:_|$)", name):
            return condition.upper() if condition == "wt" else condition.upper()
    return "all conditions"


def infer_protocol(path: Path, frame: pd.DataFrame) -> str:
    for column in ("protocol", "comparison_protocol", "ensemble"):
        if column in frame and frame[column].notna().any():
            values = frame[column].dropna().astype(str)
            if values.nunique() == 1:
                return values.iloc[0]
            return "multiple protocols"
    name = path.stem.lower()
    if "masked_v2_noifm" in name:
        return "masked v2 no-IFM"
    if "masked_v2" in name:
        return "masked v2"
    if "masked" in name:
        return "masked versus vanilla"
    return "not protocol-specific"


def standardize_table(path: Path, channel: str) -> bool:
    frame = pd.read_csv(path)
    changed = False
    additions = {
        "channel": CHANNEL_LABELS[channel],
        "condition": infer_condition(path, frame, channel),
        "protocol": infer_protocol(path, frame),
    }
    for column, value in reversed(list(additions.items())):
        if column not in frame.columns:
            frame.insert(0, column, value)
            changed = True
    if changed:
        frame.to_csv(path, index=False)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    changed = 0
    total = 0
    for channel in CHANNEL_LABELS:
        for path in sorted((args.repo_root / channel).glob("data*/analysis/**/tables/*.csv")):
            total += 1
            changed += standardize_table(path, channel)
    print(f"Standardized {changed} of {total} analysis tables.")


if __name__ == "__main__":
    main()
