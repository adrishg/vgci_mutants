"""Experimental-distance column discovery and paired deltas."""

from __future__ import annotations
import re
import pandas as pd


def experimental_distance_columns(frame: pd.DataFrame) -> list[str]:
    return [
        c for c in frame if re.search(r"(mean.*distance_to_|rms_deviation_to_).+_A$", c)
        and not c.startswith(("delta_", "ratio_"))
    ]


def paired_experimental_differences(
    profiles: pd.DataFrame, condition: str, masked_protocol: str
) -> pd.DataFrame:
    columns = experimental_distance_columns(profiles)
    part = profiles.loc[profiles.sequence_condition.eq(condition)]
    vanilla = part.loc[part.protocol.eq("vanilla"), ["raw_residue_number"] + columns]
    masked = part.loc[part.protocol.eq(masked_protocol), ["raw_residue_number"] + columns]
    merged = vanilla.merge(masked, on="raw_residue_number", suffixes=("_vanilla", "_masked"))
    for column in columns:
        merged[f"masked_minus_vanilla__{column}"] = (
            merged[f"{column}_masked"] - merged[f"{column}_vanilla"]
        )
    return merged
