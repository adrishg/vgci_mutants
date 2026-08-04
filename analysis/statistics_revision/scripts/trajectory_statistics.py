#!/usr/bin/env python3
"""Shared trajectory parsing and cluster-aware statistical utilities.

The independent unit is an AlphaFold model-parameterization/random-seed
trajectory. Rank and recycle identify outputs within that hierarchy but are not
independent sampling units.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Callable, Iterable

import numpy as np
import pandas as pd


FILENAME_RE = re.compile(
    r"(?:rank_(?P<rank>\d+)_)?alphafold2(?:_multimer_v3|_ptm)?_model_"
    r"(?P<model>\d+)_seed_(?P<seed>\d+)"
    r"(?:\.r(?P<recycle>\d+))?(?:\.pdb)?$",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedModelName:
    rank: int | None
    model: int
    seed: int
    recycle: int | None

    @property
    def trajectory_key(self) -> str:
        return f"model_{self.model}_seed_{self.seed:03d}"


def parse_model_name(value: object) -> ParsedModelName:
    """Parse rank/model/seed/recycle from every repository naming pattern."""
    name = Path(str(value)).name
    match = FILENAME_RE.search(name)
    if not match:
        raise ValueError(f"Unrecognized AlphaFold filename: {value}")
    groups = match.groupdict()
    return ParsedModelName(
        rank=int(groups["rank"]) if groups["rank"] else None,
        model=int(groups["model"]),
        seed=int(groups["seed"]),
        recycle=int(groups["recycle"]) if groups["recycle"] else None,
    )


def add_trajectory_columns(
    frame: pd.DataFrame,
    *,
    filename_col: str = "pdb_file",
    dataset: str,
) -> pd.DataFrame:
    if filename_col not in frame:
        raise KeyError(f"Missing filename column {filename_col!r}")
    parsed = frame[filename_col].map(parse_model_name)
    result = frame.copy()
    result["af2_model"] = parsed.map(lambda item: item.model)
    result["seed"] = parsed.map(lambda item: item.seed)
    result["rank"] = parsed.map(lambda item: item.rank)
    result["recycle_number"] = parsed.map(lambda item: item.recycle)
    result["trajectory_id"] = parsed.map(
        lambda item: f"{dataset}|{item.trajectory_key}"
    )
    return result


def select_one_snapshot(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Select earliest or latest retained numbered recycle per trajectory."""
    if rule not in {"earliest", "latest"}:
        raise ValueError("rule must be 'earliest' or 'latest'")
    if "trajectory_id" not in frame or "recycle_number" not in frame:
        raise KeyError("trajectory_id and recycle_number are required")
    numbered = frame.dropna(subset=["recycle_number"]).copy()
    ordered = numbered.sort_values(
        ["trajectory_id", "recycle_number", "pdb_file"]
    )
    return ordered.groupby("trajectory_id", as_index=False).nth(
        0 if rule == "earliest" else -1
    ).reset_index(drop=True)


def equal_trajectory_estimate(
    frame: pd.DataFrame,
    value_col: str,
    within: Callable[[pd.Series], float],
    across: Callable[[pd.Series], float] = np.mean,
) -> float:
    per_trajectory = frame.groupby("trajectory_id")[value_col].apply(within)
    return float(across(per_trajectory.dropna()))


def cluster_bootstrap(
    frame: pd.DataFrame,
    statistic: Callable[[pd.DataFrame], float],
    *,
    replicates: int,
    seed: int,
) -> tuple[float, float, float]:
    """Percentile bootstrap sampling whole trajectories with replacement."""
    codes, keys = pd.factorize(frame["trajectory_id"], sort=True)
    cluster_count = len(keys)
    if not cluster_count:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    estimates = np.empty(replicates, dtype=float)
    for index in range(replicates):
        # Multinomial cluster counts are exactly equivalent to drawing the same
        # number of trajectories with replacement, but avoid concatenating
        # hundreds of small DataFrames in every replicate.
        counts = rng.multinomial(
            cluster_count, np.full(cluster_count, 1.0 / cluster_count)
        )
        row_repeats = counts[codes]
        draw = frame.iloc[np.repeat(np.arange(len(frame)), row_repeats)]
        estimates[index] = statistic(draw)
    finite = estimates[np.isfinite(estimates)]
    if not len(finite):
        return np.nan, np.nan, np.nan
    return (
        float(statistic(frame)),
        float(np.quantile(finite, 0.025)),
        float(np.quantile(finite, 0.975)),
    )


def test_filename_parser() -> None:
    examples: Iterable[tuple[str, tuple[int | None, int, int, int | None]]] = [
        (
            "kv21_l403a_masked_unrelaxed_rank_062_alphafold2_multimer_v3_model_5_seed_046.r10.pdb",
            (62, 5, 46, 10),
        ),
        (
            "kv21_l403a_unrelaxed_alphafold2_multimer_v3_model_1_seed_055.r7.pdb",
            (None, 1, 55, 7),
        ),
        (
            "nav15_qqq_7fbs_masked_unrelaxed_rank_222_alphafold2_ptm_model_1_seed_033.r0.pdb",
            (222, 1, 33, 0),
        ),
        (
            "cav12_g402s_short_masked_unrelaxed_rank_349_alphafold2_ptm_model_1_seed_091.r1.pdb",
            (349, 1, 91, 1),
        ),
    ]
    for filename, expected in examples:
        parsed = parse_model_name(filename)
        observed = (parsed.rank, parsed.model, parsed.seed, parsed.recycle)
        if observed != expected:
            raise AssertionError(f"{filename}: expected {expected}, observed {observed}")
