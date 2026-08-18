"""Select QC-passing L403A structures with the most 8SDA-like S6 geometry.

No coordinates or structural metrics are recalculated. Selection uses the
existing chain-resolved metrics and the exact distance-QC manifest join.
"""
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kv21.run_l403a_conformational_validation import load_inputs, TAB

METRICS = [
    "kink_angle_deg",
    "whole_s6_rotation_vs_8SD3_deg",
    "I401_azimuth_deg",
    "I405_azimuth_deg",
]
ANGULAR = set(METRICS[1:])


def angular_error(values, target):
    return np.abs((values - target + 180.0) % 360.0 - 180.0)


def rank_candidates():
    exp, _, selected = load_inputs()
    candidates = selected[
        selected.source_type.eq("prediction") & selected.condition.eq("l403a")
    ].copy()
    targets = exp.set_index("canonical_subunit")

    component_columns = []
    for metric in METRICS:
        target_map = targets[f"8SDA__{metric}"]
        target = candidates.canonical_subunit.map(target_map)
        error = (
            angular_error(candidates[metric], target)
            if metric in ANGULAR
            else (candidates[metric] - target).abs()
        )
        candidates[f"error__{metric}"] = error

    index = ["structure_id"]
    metadata_columns = ["structure_id", "protocol", "source_path", "seed", "model_number",
                        "recycle", "recycle_label", "rank", "is_final_model",
                        "frame_orientation_score"]
    metadata = candidates[metadata_columns].drop_duplicates("structure_id")
    pieces = []
    for metric in METRICS:
        values = candidates.pivot_table(
            index=index, columns="canonical_subunit", values=metric, aggfunc="first"
        )
        values.columns = [f"value__{metric}__subunit_{c}" for c in values.columns]
        p = candidates.pivot_table(
            index=index, columns="canonical_subunit", values=f"error__{metric}",
            aggfunc="first",
        )
        p.columns = [f"abs_error__{metric}__subunit_{c}" for c in p.columns]
        pieces.extend([values, p])
        component_columns.extend(p.columns)
    wide = pd.concat(pieces, axis=1).reset_index().merge(
        metadata, on="structure_id", how="left", validate="one_to_one"
    )

    # Percentile ranks make Å and degree errors comparable without hiding the
    # original physical-unit errors. Lower is more 8SDA-like.
    rank_columns = []
    for col in component_columns:
        rcol = col.replace("abs_error__", "percentile_error__")
        wide[rcol] = wide[col].rank(method="average", pct=True)
        rank_columns.append(rcol)
    bd = [c for c in rank_columns if c.endswith("subunit_B") or c.endswith("subunit_D")]
    wide["remodeled_BD_percentile_score"] = wide[bd].mean(axis=1)
    wide["all_subunit_percentile_score"] = wide[rank_columns].mean(axis=1)
    wide["source_basename"] = wide.source_path.map(lambda x: os.path.basename(str(x)))
    wide["source_exists_locally"] = wide.source_path.map(lambda x: Path(str(x)).is_file())
    for metric in METRICS:
        for subunit in "ABCD":
            wide[f"target_8SDA__{metric}__subunit_{subunit}"] = targets.loc[subunit, f"8SDA__{metric}"]
    wide = wide.sort_values(
        ["remodeled_BD_percentile_score", "all_subunit_percentile_score", "protocol", "seed"]
    ).reset_index(drop=True)
    wide["overall_rank"] = np.arange(1, len(wide) + 1)

    # Primary list: best B/D experimental-like S6 candidates, at most one per seed.
    primary = wide.drop_duplicates("seed").head(10).copy()
    primary.insert(0, "selection_rank", np.arange(1, len(primary) + 1))
    primary["selection_reason"] = (
        "Top QC-passing L403A structure by mean percentile error for kink, "
        "whole-S6 rotation, I401 azimuth, and I405 azimuth in experimentally "
        "remodeled subunits B/D; one structure per seed"
    )

    # Protocol-control set: five best unique seeds from each protocol.
    balanced = []
    for protocol in ["vanilla", "masked"]:
        x = wide[wide.protocol.eq(protocol)].drop_duplicates("seed").head(5).copy()
        x.insert(0, "within_protocol_rank", np.arange(1, len(x) + 1))
        balanced.append(x)
    balanced = pd.concat(balanced, ignore_index=True)

    TAB.mkdir(parents=True, exist_ok=True)
    wide.to_csv(TAB / "l403a_s6_relaxation_candidate_ranking.csv", index=False)
    primary.to_csv(TAB / "l403a_s6_relaxation_top10.csv", index=False)
    balanced.to_csv(TAB / "l403a_s6_relaxation_balanced_5plus5.csv", index=False)
    return primary, balanced, wide


if __name__ == "__main__":
    primary, balanced, _ = rank_candidates()
    show = ["selection_rank", "protocol", "seed", "model_number", "recycle_label",
            "rank", "source_basename", "remodeled_BD_percentile_score",
            "all_subunit_percentile_score", "source_exists_locally"]
    print(primary[show].to_string(index=False))
