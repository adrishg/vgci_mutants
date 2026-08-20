"""Trajectory-block statistics for precomputed structural-distance tables.

This module never reads coordinates or calculates structural distances.  Its
sampling unit is an AlphaFold trajectory (seed, model), while retained recycle
snapshots remain observations within that block.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from statsmodels.stats.multitest import multipletests

from shared.seed_block_statistics import weighted_quantile


_SEED = re.compile(r"_seed_(\d+)", re.I)
_MODEL = re.compile(r"_model_(\d+)", re.I)
_RECYCLE = re.compile(r"\.r(\d+)(?=\.pdb$|$)", re.I)
DISTANCE_PREFIXES = ("CA_", "shortest_")
IQR_EPSILON = 1e-12


def parse_trajectory_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with audited seed/model/recycle trajectory metadata."""
    if "pdb_file" not in frame:
        raise KeyError("Distance table requires a 'pdb_file' column")
    result = frame.copy()
    names = result["pdb_file"].astype(str)
    result["seed"] = pd.to_numeric(names.str.extract(_SEED, expand=False), errors="coerce")
    result["model_number"] = pd.to_numeric(
        names.str.extract(_MODEL, expand=False), errors="coerce"
    )
    result["recycle_number"] = pd.to_numeric(
        names.str.extract(_RECYCLE, expand=False), errors="coerce"
    )
    failed = result[["seed", "model_number"]].isna().any(axis=1)
    if failed.any():
        examples = result.loc[failed, "pdb_file"].head(8).tolist()
        raise ValueError(
            f"Could not parse seed/model trajectory identity for {int(failed.sum())} rows: "
            f"{examples}"
        )
    result["seed"] = result["seed"].astype(int)
    result["model_number"] = result["model_number"].astype(int)
    # Missing .rN is a valid final/base file; preserve it as nullable metadata.
    result["recycle_number"] = result["recycle_number"].astype("Int64")
    result["trajectory_id"] = list(zip(result["seed"], result["model_number"]))
    result.attrs.update(frame.attrs)
    return result


def trajectory_weights(frame: pd.DataFrame, distance: str) -> pd.Series:
    """Give every trajectory equal mass and divide it over its valid rows."""
    valid = pd.to_numeric(frame[distance], errors="coerce").notna()
    work = frame.loc[valid]
    if work.empty:
        return pd.Series(dtype=float, name="weight")
    counts = work.groupby("trajectory_id")[distance].transform("size")
    number = work["trajectory_id"].nunique()
    return pd.Series(1.0 / (number * counts), index=work.index, name="weight")


def candidate_distance_columns(frame: pd.DataFrame) -> list[str]:
    """Conservatively identify numeric precomputed structural-distance columns."""
    columns = []
    for column in frame.columns:
        if not str(column).startswith(DISTANCE_PREFIXES):
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().any():
            columns.append(column)
    return columns


def exact_common_distance_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    """Return exact shared names only; residue identities are never stripped."""
    return sorted(set(candidate_distance_columns(a)) & set(candidate_distance_columns(b)))


def unmatched_distance_columns(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    ca, cb = set(candidate_distance_columns(a)), set(candidate_distance_columns(b))
    return pd.DataFrame({
        "only_in_A": pd.Series(sorted(ca - cb), dtype="string"),
        "only_in_B": pd.Series(sorted(cb - ca), dtype="string"),
    })


def _blocks(frame: pd.DataFrame, distance: str) -> dict[tuple[int, int], np.ndarray]:
    values = pd.to_numeric(frame[distance], errors="coerce")
    work = pd.DataFrame({"trajectory_id": frame["trajectory_id"], "value": values}).dropna()
    return {
        key: group["value"].to_numpy(float)
        for key, group in work.groupby("trajectory_id", sort=True)
    }


def _weighted_samples(blocks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    blocks = [np.asarray(block, dtype=float) for block in blocks if len(block)]
    if not blocks:
        return np.array([], dtype=float), np.array([], dtype=float)
    total = len(blocks)
    return (
        np.concatenate(blocks),
        np.concatenate([np.full(len(block), 1.0 / (total * len(block))) for block in blocks]),
    )


def trajectory_balanced_w1(
    blocks_a: Sequence[np.ndarray], blocks_b: Sequence[np.ndarray]
) -> float:
    a, wa = _weighted_samples(blocks_a)
    b, wb = _weighted_samples(blocks_b)
    if not len(a) or not len(b):
        return np.nan
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


def _paired_transport_basis(
    blocks_a: Sequence[np.ndarray], blocks_b: Sequence[np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-pair CDF differences and pooled support interval widths.

    In one dimension W1 is the integral of the absolute CDF difference.  The
    support is fixed across block swaps/resamples, so calculating each block's
    empirical CDF once avoids re-sorting thousands of rows in every Monte
    Carlo replicate.  Float64 is retained so this fast path is numerically
    equivalent to repeated ``scipy.stats.wasserstein_distance`` calls.
    """
    a = [np.sort(np.asarray(block, dtype=float)) for block in blocks_a]
    b = [np.sort(np.asarray(block, dtype=float)) for block in blocks_b]
    support = np.unique(np.concatenate([*a, *b]))
    if len(support) < 2:
        return np.zeros((len(a), 0), dtype=float), np.zeros(0, dtype=float)
    points = support[:-1]
    differences = np.vstack([
        np.searchsorted(left, points, side="right") / len(left)
        - np.searchsorted(right, points, side="right") / len(right)
        for left, right in zip(a, b)
    ])
    return differences, np.diff(support)


def _transport_from_coefficients(
    coefficients: np.ndarray, differences: np.ndarray, widths: np.ndarray,
    *, batch_size: int = 64,
) -> np.ndarray:
    """Evaluate many CDF integrals in bounded-memory matrix batches."""
    coefficients = np.atleast_2d(np.asarray(coefficients, dtype=float))
    result = np.empty(len(coefficients), dtype=float)
    for start in range(0, len(coefficients), batch_size):
        stop = min(start + batch_size, len(coefficients))
        cdf_difference = coefficients[start:stop] @ differences
        result[start:stop] = np.abs(cdf_difference) @ widths
    return result


def _replicated_block_quantiles(
    blocks: Sequence[np.ndarray], counts: np.ndarray,
    quantiles: Sequence[float] = (.25, .5, .75),
) -> np.ndarray:
    """Equal-block weighted inverse-ECDF quantiles after block resampling."""
    values = np.concatenate([np.asarray(block, dtype=float) for block in blocks])
    block_ids = np.concatenate([
        np.full(len(block), index, dtype=int) for index, block in enumerate(blocks)
    ])
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    block_ids = block_ids[order]
    within_block = np.concatenate([
        np.full(len(block), 1.0 / len(block)) for block in blocks
    ])[order]
    row_multiplicity = counts[:, block_ids] * within_block
    cumulative = np.cumsum(row_multiplicity, axis=1)
    totals = cumulative[:, -1]
    output = np.empty((len(counts), len(quantiles)), dtype=float)
    for column, quantile in enumerate(quantiles):
        indices = (cumulative >= (totals * quantile)[:, None]).argmax(axis=1)
        output[:, column] = values[indices]
    return output


def _flatten(blocks: Sequence[np.ndarray]) -> np.ndarray:
    kept = [np.asarray(x, dtype=float) for x in blocks if len(x)]
    return np.concatenate(kept) if kept else np.array([], dtype=float)


def distribution_metrics(
    blocks_a: Sequence[np.ndarray], blocks_b: Sequence[np.ndarray], *, epsilon: float = IQR_EPSILON
) -> dict[str, float | bool]:
    """Effect sizes and descriptive summaries for two block collections."""
    a, wa = _weighted_samples(blocks_a)
    b, wb = _weighted_samples(blocks_b)
    if not len(a) or not len(b):
        raise ValueError("Both ensembles require at least one valid observation")
    qa = weighted_quantile(a, [.25, .5, .75], wa)
    qb = weighted_quantile(b, [.25, .5, .75], wb)
    median_a, median_b = float(qa[1]), float(qb[1])
    iqr_a = float(qa[2] - qa[0])
    iqr_b = float(qb[2] - qb[0])
    pooled = np.concatenate([a, b])
    pooled_weights = np.concatenate([wa * .5, wb * .5])
    pooled_q = weighted_quantile(pooled, [.25, .75], pooled_weights)
    pooled_iqr = float(pooled_q[1] - pooled_q[0])
    zero_a, zero_b = iqr_a <= epsilon, iqr_b <= epsilon
    ratio = float((iqr_b + epsilon) / (iqr_a + epsilon))
    w1 = trajectory_balanced_w1(blocks_a, blocks_b)
    return {
        "median_A_A": median_a,
        "median_B_A": median_b,
        "delta_median_A": median_b - median_a,
        "IQR_A_A": iqr_a,
        "IQR_B_A": iqr_b,
        "IQR_ratio_B_over_A": ratio,
        "log2_IQR_ratio": float(np.log2(ratio)),
        "IQR_A_effectively_zero": zero_a,
        "IQR_B_effectively_zero": zero_b,
        "W1_A": w1,
        "pooled_IQR_A": pooled_iqr,
        "W1_normalized_by_pooled_IQR": w1 / pooled_iqr if pooled_iqr > epsilon else np.nan,
        "KS_D": _weighted_ks(a, wa, b, wb),
    }


def _weighted_ks(a: np.ndarray, wa: np.ndarray, b: np.ndarray, wb: np.ndarray) -> float:
    """Maximum difference between the same weighted ECDFs used by W1."""
    support = np.unique(np.concatenate([a, b]))
    order_a, order_b = np.argsort(a), np.argsort(b)
    a, wa, b, wb = a[order_a], wa[order_a], b[order_b], wb[order_b]
    cdf_a, cdf_b = np.cumsum(wa) / wa.sum(), np.cumsum(wb) / wb.sum()
    ia = np.searchsorted(a, support, side="right") - 1
    ib = np.searchsorted(b, support, side="right") - 1
    fa = np.where(ia >= 0, cdf_a[np.maximum(ia, 0)], 0.0)
    fb = np.where(ib >= 0, cdf_b[np.maximum(ib, 0)], 0.0)
    return float(np.max(np.abs(fa - fb)))


def pairing_audit(a: Mapping, b: Mapping) -> dict[str, float | int]:
    ka, kb = set(a), set(b)
    matched = ka & kb
    return {
        "n_trajectories_A": len(ka), "n_trajectories_B": len(kb),
        "n_matched_trajectories": len(matched),
        "fraction_A_matched": len(matched) / len(ka) if ka else np.nan,
        "fraction_B_matched": len(matched) / len(kb) if kb else np.nan,
    }


def paired_block_permutation(
    blocks_a: Mapping[tuple[int, int], np.ndarray],
    blocks_b: Mapping[tuple[int, int], np.ndarray],
    *, n_permutations: int, random_seed: int, return_null: bool = False,
):
    """Swap complete matched trajectory blocks; never swap recycle rows."""
    keys = sorted(set(blocks_a) & set(blocks_b))
    if not keys:
        raise ValueError("Paired permutation requires matched trajectories")
    a = [blocks_a[k] for k in keys]
    b = [blocks_b[k] for k in keys]
    observed = trajectory_balanced_w1(a, b)
    rng = np.random.default_rng(random_seed)
    differences, widths = _paired_transport_basis(a, b)
    # +1 retains A/B orientation and -1 swaps the complete pair.
    signs = 1.0 - 2.0 * rng.integers(
        0, 2, size=(n_permutations, len(keys)), dtype=np.int8
    )
    null = _transport_from_coefficients(
        signs / len(keys), differences, widths
    )
    pvalue = (1 + np.count_nonzero(null >= observed - 1e-15)) / (1 + n_permutations)
    return (float(pvalue), null) if return_null else float(pvalue)


def unpaired_block_permutation(
    blocks_a: Sequence[np.ndarray], blocks_b: Sequence[np.ndarray],
    *, n_permutations: int, random_seed: int,
) -> float:
    """Sensitivity test that permutes whole unpaired trajectories between groups."""
    a, b = list(blocks_a), list(blocks_b)
    observed = trajectory_balanced_w1(a, b)
    combined, n_a = a + b, len(a)
    rng = np.random.default_rng(random_seed)
    exceed = 0
    for _ in range(n_permutations):
        order = rng.permutation(len(combined))
        value = trajectory_balanced_w1(
            [combined[i] for i in order[:n_a]], [combined[i] for i in order[n_a:]]
        )
        exceed += value >= observed - 1e-15
    return float((1 + exceed) / (1 + n_permutations))


def paired_block_bootstrap(
    blocks_a: Mapping[tuple[int, int], np.ndarray],
    blocks_b: Mapping[tuple[int, int], np.ndarray],
    *, n_bootstrap: int, random_seed: int, return_samples: bool = False,
):
    """Resample matched trajectory pairs, carrying every recycle row with them."""
    keys = sorted(set(blocks_a) & set(blocks_b))
    if not keys:
        raise ValueError("Paired bootstrap requires matched trajectories")
    rng = np.random.default_rng(random_seed)
    a = [blocks_a[key] for key in keys]
    b = [blocks_b[key] for key in keys]
    differences, widths = _paired_transport_basis(a, b)
    draws = rng.integers(0, len(keys), size=(n_bootstrap, len(keys)))
    counts = np.zeros((n_bootstrap, len(keys)), dtype=float)
    for index, draw in enumerate(draws):
        counts[index] = np.bincount(draw, minlength=len(keys))
    bootstrap_w1 = _transport_from_coefficients(
        counts / len(keys), differences, widths
    )
    quantiles_a = _replicated_block_quantiles(a, counts)
    quantiles_b = _replicated_block_quantiles(b, counts)
    median_shift = quantiles_b[:, 1] - quantiles_a[:, 1]
    iqr_a = quantiles_a[:, 2] - quantiles_a[:, 0]
    iqr_b = quantiles_b[:, 2] - quantiles_b[:, 0]
    log_ratio = np.log2((iqr_b + IQR_EPSILON) / (iqr_a + IQR_EPSILON))
    samples = np.column_stack([bootstrap_w1, median_shift, log_ratio])
    ci = np.quantile(samples, [.025, .975], axis=0)
    result = {
        "W1_CI_low_A": ci[0, 0], "W1_CI_high_A": ci[1, 0],
        "delta_median_CI_low_A": ci[0, 1], "delta_median_CI_high_A": ci[1, 1],
        "log2_IQR_ratio_CI_low": ci[0, 2], "log2_IQR_ratio_CI_high": ci[1, 2],
    }
    return (result, samples) if return_samples else result


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a reproducible per-analysis seed without Python's randomized hash."""
    payload = "|".join([str(base_seed), *(str(part) for part in parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def analyze_distance(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame, distance: str, *,
    analysis_level: str = "trajectory_balanced_snapshots",
    paired: bool = True, n_permutations: int = 999, n_bootstrap: int = 500,
    random_seed: int = 20260816,
) -> dict[str, object]:
    """Analyze one exact shared distance using trajectory-block inference."""
    ba, bb = _blocks(frame_a, distance), _blocks(frame_b, distance)
    audit = pairing_audit(ba, bb)
    if analysis_level == "trajectory_median":
        ba = {key: np.array([np.median(value)]) for key, value in ba.items()}
        bb = {key: np.array([np.median(value)]) for key, value in bb.items()}
    elif analysis_level != "trajectory_balanced_snapshots":
        raise ValueError(f"Unknown analysis_level: {analysis_level}")
    if paired:
        keys = sorted(set(ba) & set(bb))
        if not keys:
            raise ValueError(f"{distance}: no matched trajectory pairs")
        metrics = distribution_metrics([ba[k] for k in keys], [bb[k] for k in keys])
        pvalue = paired_block_permutation(
            ba, bb, n_permutations=n_permutations, random_seed=random_seed
        )
        ci = paired_block_bootstrap(
            ba, bb, n_bootstrap=n_bootstrap, random_seed=random_seed + 1
        )
        test_mode = "paired_trajectory_block"
    else:
        metrics = distribution_metrics(list(ba.values()), list(bb.values()))
        pvalue = unpaired_block_permutation(
            list(ba.values()), list(bb.values()),
            n_permutations=n_permutations, random_seed=random_seed,
        )
        ci = {key: np.nan for key in (
            "W1_CI_low_A", "W1_CI_high_A", "delta_median_CI_low_A",
            "delta_median_CI_high_A", "log2_IQR_ratio_CI_low", "log2_IQR_ratio_CI_high"
        )}
        test_mode = "unpaired_trajectory_block"
    return {
        "distance": distance, "analysis_level": analysis_level,
        "rows_A": int(pd.to_numeric(frame_a[distance], errors="coerce").notna().sum()),
        "rows_B": int(pd.to_numeric(frame_b[distance], errors="coerce").notna().sum()),
        **audit, **metrics, "p_W1_permutation": pvalue, **ci,
        "test_mode": test_mode, "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap, "random_seed": random_seed,
    }


def adjust_fdr(results: pd.DataFrame) -> pd.DataFrame:
    """Add within-comparison and global Benjamini-Hochberg q-values."""
    result = results.copy()
    result["q_within_comparison"] = np.nan
    for _, index in result.groupby(["comparison_id", "analysis_level"], sort=False).groups.items():
        valid = result.loc[index, "p_W1_permutation"].notna()
        selected = result.loc[index].index[valid]
        if len(selected):
            result.loc[selected, "q_within_comparison"] = multipletests(
                result.loc[selected, "p_W1_permutation"], method="fdr_bh"
            )[1]
    valid = result["p_W1_permutation"].notna()
    if valid.any():
        result.loc[valid, "q_global"] = multipletests(
            result.loc[valid, "p_W1_permutation"], method="fdr_bh"
        )[1]
    else:
        result["q_global"] = np.nan
    return result


def dataset_audit(
    frame: pd.DataFrame, *, channel: str, condition: str, protocol: str,
    requested_dataset: str,
) -> dict[str, object]:
    parsed = parse_trajectory_metadata(frame)
    counts = parsed.groupby("trajectory_id").size()
    return {
        "channel": channel, "condition": condition, "protocol": protocol,
        "requested_dataset": requested_dataset,
        "actual_dataset": frame.attrs.get("actual_selection"),
        "source_path": frame.attrs.get("source_path"), "n_rows": len(frame),
        "n_distance_columns": len(candidate_distance_columns(frame)),
        "n_trajectories": len(counts), "min_rows_per_trajectory": int(counts.min()),
        "median_rows_per_trajectory": float(counts.median()),
        "max_rows_per_trajectory": int(counts.max()), "trajectory_parse_failures": 0,
        "duplicate_pdb_file_count": int(frame["pdb_file"].duplicated().sum()),
    }


def cache_key(
    comparison_id: str, source_paths: Iterable[str | Path], run_mode: str,
    n_permutations: int, n_bootstrap: int, random_seed: int,
) -> str:
    payload = "|".join(map(str, (
        comparison_id, *source_paths, run_mode, n_permutations, n_bootstrap, random_seed
    )))
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
