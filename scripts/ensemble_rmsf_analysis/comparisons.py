"""Paired protocol comparisons and mask-distance classes."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annotate_mask_classes(frame: pd.DataFrame, mask_positions: set[int]) -> pd.DataFrame:
    result = frame.copy()
    raw = result["raw_residue_number"].astype(int).to_numpy()
    mask = np.array(sorted(mask_positions), dtype=int)
    if not len(mask):
        result["directly_masked"] = False
        result["sequence_distance_to_nearest_mask"] = np.nan
        result["mask_sequence_class"] = "mask_unavailable"
        return result
    distances = np.min(np.abs(raw[:, None] - mask[None, :]), axis=1)
    result["directly_masked"] = distances == 0
    result["sequence_distance_to_nearest_mask"] = distances
    result["mask_sequence_class"] = np.select(
        [distances == 0, (distances >= 1) & (distances <= 5), (distances >= 6) & (distances <= 10)],
        ["directly_masked", "adjacent_to_mask_1_to_5", "adjacent_to_mask_6_to_10"],
        default="unmasked",
    )
    return result


def paired_rmsf_comparison(
    profiles: pd.DataFrame, condition: str, masked_protocol: str,
    rmsf_column: str, mask_positions: set[int], epsilon: float = 1e-6,
) -> pd.DataFrame:
    part = profiles.loc[profiles.sequence_condition.astype(str).str.lower().eq(condition.lower())]
    vanilla = part.loc[part.protocol.eq("vanilla")]
    masked = part.loc[part.protocol.eq(masked_protocol)]
    if vanilla.empty or masked.empty:
        raise ValueError(f"Missing paired profiles for {condition}: vanilla versus {masked_protocol}")
    id_columns = ["raw_residue_number"]
    carry = [c for c in ("residue_identity", "coverage_fraction", "mean_chain_coverage_fraction",
                         "annotation_regions", "alignment_core") if c in part]
    left = vanilla[id_columns + [rmsf_column] + carry].rename(
        columns={rmsf_column: "vanilla_rmsf_A", **{c: f"{c}_vanilla" for c in carry}}
    )
    right = masked[id_columns + [rmsf_column] + carry].rename(
        columns={rmsf_column: "masked_rmsf_A", **{c: f"{c}_masked" for c in carry}}
    )
    result = left.merge(right, on=id_columns, validate="one_to_one")
    result["sequence_condition"] = condition
    result["comparison_protocol"] = masked_protocol
    result["masked_minus_vanilla_rmsf_A"] = result.masked_rmsf_A - result.vanilla_rmsf_A
    result["masked_divided_by_vanilla_rmsf"] = result.masked_rmsf_A / result.vanilla_rmsf_A
    result["log2_masked_over_vanilla_rmsf"] = np.log2(
        (result.masked_rmsf_A + epsilon) / (result.vanilla_rmsf_A + epsilon)
    )
    return annotate_mask_classes(result, mask_positions)


def summarize_comparison(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    delta = frame.masked_minus_vanilla_rmsf_A.abs()
    total = delta.sum()
    inside = delta[frame.directly_masked].sum()
    allocation = pd.DataFrame([{
        "total_absolute_rmsf_change_A": total,
        "inside_mask_absolute_change_A": inside,
        "outside_mask_absolute_change_A": total - inside,
        "fraction_inside_mask": inside / total if total else np.nan,
        "fraction_outside_mask": 1 - inside / total if total else np.nan,
    }])
    classes = frame.groupby("mask_sequence_class").agg(
        number_of_residues=("raw_residue_number", "size"),
        median_delta_rmsf_A=("masked_minus_vanilla_rmsf_A", "median"),
        mean_delta_rmsf_A=("masked_minus_vanilla_rmsf_A", "mean"),
        fraction_increased=("masked_minus_vanilla_rmsf_A", lambda x: x.gt(0).mean()),
    ).reset_index()
    return allocation, classes
