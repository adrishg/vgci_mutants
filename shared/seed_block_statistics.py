"""Seed-block inference for AlphaFold ensembles.

The random seed is the independent resampling unit.  Within each seed, every
available AlphaFold model parameterization receives equal mass, and the mass
of a model is divided equally over its retained recycle snapshots.  This keeps
model number as a fixed stratum instead of treating five parameterizations
sharing a seed as five independent replicates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


IQR_EPSILON = 1e-12


SeedBlock = Mapping[int, Mapping[int, np.ndarray]]


def reduce_trajectory_values(
    frame: pd.DataFrame, value_col: str, *, reduction: str = "median"
) -> pd.DataFrame:
    """Reduce recycles before inference while retaining seed/model strata."""
    required = {"seed", "model_number", value_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns required for trajectory reduction: {missing}")
    values = pd.to_numeric(frame[value_col], errors="coerce")
    work = frame[["seed", "model_number"]].assign(_value=values).dropna(subset=["_value"])
    grouped = work.groupby(["seed", "model_number"], as_index=False)["_value"]
    if reduction == "median":
        result = grouped.median()
    elif reduction == "latest":
        if "recycle_number" not in frame:
            raise KeyError("latest reduction requires recycle_number")
        numbered = frame.assign(_value=values).dropna(subset=["_value"]).sort_values(
            ["seed", "model_number", "recycle_number"]
        )
        result = numbered.groupby(["seed", "model_number"], as_index=False).tail(1)[
            ["seed", "model_number", "_value"]
        ]
    else:
        raise ValueError(f"Unsupported trajectory reduction: {reduction}")
    return result.rename(columns={"_value": value_col}).reset_index(drop=True)


def seed_model_stratum_weights(frame: pd.DataFrame) -> np.ndarray:
    """Weights for a one-row-per-seed-model trajectory-reduced table."""
    required = {"seed", "model_number"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns required for seed/model weights: {missing}")
    if frame.duplicated(["seed", "model_number"]).any():
        raise ValueError("Trajectory-reduced input must contain one row per seed/model stratum")
    if frame.empty:
        return np.array([], dtype=float)
    number_seeds = frame["seed"].nunique()
    models_per_seed = frame.groupby("seed")["model_number"].transform("nunique")
    return (1.0 / (number_seeds * models_per_seed)).to_numpy(float)


def _weighted_quantiles_matrix(
    values: np.ndarray, weights: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    """Column-wise inverse-ECDF quantiles for complete finite matrices."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if values.ndim != 2 or weights.shape != (values.shape[0],):
        raise ValueError("values must be rows x coordinates and weights one per row")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Matrix statistics require finite values and positive finite weights")
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = np.take_along_axis(weights[:, None], order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    totals = cumulative[-1]
    result = np.empty((len(quantiles), values.shape[1]), dtype=float)
    for position, quantile in enumerate(quantiles):
        indices = np.argmax(cumulative >= quantile * totals, axis=0)
        result[position] = np.take_along_axis(
            sorted_values, indices[None, :], axis=0
        )[0]
    return result


def seed_distribution_metrics_matrix(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_cols: Sequence[str],
    *,
    epsilon: float = IQR_EPSILON,
) -> pd.DataFrame:
    """Exact point metrics for many trajectory-reduced coordinates at once.

    This is the scalable full-panel companion to :func:`seed_distribution_metrics`.
    It intentionally calculates point estimates only; confirmatory uncertainty is
    produced separately for prespecified coordinates by whole-seed resampling.
    """
    columns = list(value_cols)
    if not columns:
        return pd.DataFrame(index=pd.Index([], name="distance"))
    missing_a = sorted(set(columns) - set(frame_a.columns))
    missing_b = sorted(set(columns) - set(frame_b.columns))
    if missing_a or missing_b:
        raise KeyError(f"Missing matrix columns: A={missing_a}, B={missing_b}")
    a = frame_a[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    b = frame_b[columns].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Full-panel matrix contains non-finite trajectory summaries")
    wa, wb = seed_model_stratum_weights(frame_a), seed_model_stratum_weights(frame_b)
    qa = _weighted_quantiles_matrix(a, wa, (.25, .5, .75))
    qb = _weighted_quantiles_matrix(b, wb, (.25, .5, .75))

    pooled_values = np.vstack([a, b])
    pooled_weights = np.concatenate([wa * .5, wb * .5])
    qp = _weighted_quantiles_matrix(pooled_values, pooled_weights, (.25, .75))

    # In one dimension, W1 is the integral of the absolute weighted-CDF
    # difference. Column-wise sorting makes the calculation exact.
    signed_weights = np.concatenate([wa, -wb])
    order = np.argsort(pooled_values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(pooled_values, order, axis=0)
    sorted_signed_weights = np.take_along_axis(signed_weights[:, None], order, axis=0)
    cdf_difference = np.cumsum(sorted_signed_weights, axis=0)
    w1 = np.sum(np.abs(cdf_difference[:-1]) * np.diff(sorted_values, axis=0), axis=0)

    iqr_a, iqr_b, pooled_iqr = qa[2] - qa[0], qb[2] - qb[0], qp[1] - qp[0]
    ratio = (iqr_b + epsilon) / (iqr_a + epsilon)
    return pd.DataFrame({
        "distance": columns,
        "weighted_median_A_A": qa[1],
        "weighted_median_B_A": qb[1],
        "delta_weighted_median_A": qb[1] - qa[1],
        "weighted_IQR_A_A": iqr_a,
        "weighted_IQR_B_A": iqr_b,
        "weighted_IQR_ratio_B_over_A": ratio,
        "weighted_log2_IQR_ratio": np.log2(ratio),
        "weighted_pooled_IQR_A": pooled_iqr,
        "seed_balanced_W1_A": w1,
        "seed_balanced_W1_normalized_by_weighted_pooled_IQR": np.divide(
            w1, pooled_iqr, out=np.full_like(w1, np.nan), where=pooled_iqr > epsilon
        ),
        "IQR_A_effectively_zero": iqr_a <= epsilon,
        "IQR_B_effectively_zero": iqr_b <= epsilon,
    })


def seed_model_weights(frame: pd.DataFrame, value_col: str) -> pd.Series:
    """Return equal-seed/equal-model/equal-recycle weights for valid rows."""
    required = {"seed", "model_number", value_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing columns required for seed weights: {missing}")
    valid = pd.to_numeric(frame[value_col], errors="coerce").notna()
    work = frame.loc[valid, ["seed", "model_number"]]
    if work.empty:
        return pd.Series(dtype=float, name="weight")
    n_seeds = work["seed"].nunique()
    models_per_seed = work.groupby("seed")["model_number"].transform("nunique")
    rows_per_model = work.groupby(["seed", "model_number"])["model_number"].transform("size")
    weights = 1.0 / (n_seeds * models_per_seed * rows_per_model)
    return pd.Series(weights.to_numpy(float), index=work.index, name="weight")


def weighted_quantile(
    values: Sequence[float] | np.ndarray,
    quantiles: float | Sequence[float] | np.ndarray,
    weights: Sequence[float] | np.ndarray,
) -> float | np.ndarray:
    """Weighted inverse-ECDF quantiles using a single explicit population."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    requested = np.atleast_1d(np.asarray(quantiles, dtype=float))
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        result = np.full(requested.shape, np.nan)
    else:
        values, weights = values[valid], weights[valid]
        order = np.argsort(values, kind="mergesort")
        values, weights = values[order], weights[order]
        cumulative = np.cumsum(weights)
        targets = np.clip(requested, 0, 1) * cumulative[-1]
        indices = np.searchsorted(cumulative, targets, side="left")
        result = values[np.minimum(indices, len(values) - 1)]
    return float(result[0]) if np.ndim(quantiles) == 0 else result


def make_seed_blocks(frame: pd.DataFrame, value_col: str) -> dict[int, dict[int, np.ndarray]]:
    """Build seed -> model -> retained values without flattening the hierarchy."""
    values = pd.to_numeric(frame[value_col], errors="coerce")
    work = frame.assign(_seed_block_value=values).dropna(subset=["_seed_block_value"])
    blocks: dict[int, dict[int, np.ndarray]] = {}
    for (seed, model), group in work.groupby(["seed", "model_number"], sort=True):
        blocks.setdefault(int(seed), {})[int(model)] = group["_seed_block_value"].to_numpy(float)
    return blocks


def _seed_components(blocks: SeedBlock) -> tuple[list[int], list[np.ndarray], list[np.ndarray]]:
    seeds, values, weights = [], [], []
    for seed in sorted(blocks):
        models = blocks[seed]
        kept = [np.asarray(models[model], dtype=float) for model in sorted(models) if len(models[model])]
        if not kept:
            continue
        seeds.append(seed)
        values.append(np.concatenate(kept))
        weights.append(np.concatenate([
            np.full(len(model_values), 1.0 / (len(kept) * len(model_values)))
            for model_values in kept
        ]))
    return seeds, values, weights


def seed_weighted_samples(blocks: SeedBlock) -> tuple[np.ndarray, np.ndarray]:
    """Flatten for calculation while preserving equal seed/model mass."""
    _, values, within_seed = _seed_components(blocks)
    if not values:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.concatenate(values), np.concatenate([weight / len(values) for weight in within_seed])


def seed_balanced_w1(blocks_a: SeedBlock, blocks_b: SeedBlock) -> float:
    a, wa = seed_weighted_samples(blocks_a)
    b, wb = seed_weighted_samples(blocks_b)
    if not len(a) or not len(b):
        return np.nan
    return float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))


def seed_distribution_metrics(
    blocks_a: SeedBlock,
    blocks_b: SeedBlock,
    *,
    epsilon: float = IQR_EPSILON,
) -> dict[str, float | bool]:
    """W1 and quantile summaries using the same hierarchical weights."""
    a, wa = seed_weighted_samples(blocks_a)
    b, wb = seed_weighted_samples(blocks_b)
    if not len(a) or not len(b):
        raise ValueError("Both ensembles require at least one retained seed")
    qa = weighted_quantile(a, [.25, .5, .75], wa)
    qb = weighted_quantile(b, [.25, .5, .75], wb)
    pooled_values = np.concatenate([a, b])
    pooled_weights = np.concatenate([wa * .5, wb * .5])
    qp = weighted_quantile(pooled_values, [.25, .75], pooled_weights)
    iqr_a, iqr_b, pooled_iqr = qa[2] - qa[0], qb[2] - qb[0], qp[1] - qp[0]
    ratio = (iqr_b + epsilon) / (iqr_a + epsilon)
    w1 = float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))
    return {
        "weighted_median_A_A": float(qa[1]),
        "weighted_median_B_A": float(qb[1]),
        "delta_weighted_median_A": float(qb[1] - qa[1]),
        "weighted_IQR_A_A": float(iqr_a),
        "weighted_IQR_B_A": float(iqr_b),
        "weighted_IQR_ratio_B_over_A": float(ratio),
        "weighted_log2_IQR_ratio": float(np.log2(ratio)),
        "weighted_pooled_IQR_A": float(pooled_iqr),
        "seed_balanced_W1_A": w1,
        "seed_balanced_W1_normalized_by_weighted_pooled_IQR": (
            w1 / pooled_iqr if pooled_iqr > epsilon else np.nan
        ),
        "IQR_A_effectively_zero": bool(iqr_a <= epsilon),
        "IQR_B_effectively_zero": bool(iqr_b <= epsilon),
    }


def _cdf_basis(
    values: list[np.ndarray], within_seed_weights: list[np.ndarray], support: np.ndarray
) -> np.ndarray:
    points = support[:-1]
    rows = []
    for seed_values, seed_weights in zip(values, within_seed_weights):
        order = np.argsort(seed_values, kind="mergesort")
        ordered_values = seed_values[order]
        cumulative = np.cumsum(seed_weights[order])
        positions = np.searchsorted(ordered_values, points, side="right") - 1
        rows.append(np.where(positions >= 0, cumulative[np.maximum(positions, 0)], 0.0))
    return np.vstack(rows)


def _quantiles_from_seed_counts(
    values: list[np.ndarray],
    within_seed_weights: list[np.ndarray],
    counts: np.ndarray,
    quantiles: Sequence[float] = (.25, .5, .75),
    *,
    batch_size: int = 64,
) -> np.ndarray:
    flat_values = np.concatenate(values)
    seed_ids = np.concatenate([
        np.full(len(seed_values), index, dtype=int)
        for index, seed_values in enumerate(values)
    ])
    base_weights = np.concatenate(within_seed_weights)
    order = np.argsort(flat_values, kind="mergesort")
    flat_values, seed_ids, base_weights = (
        flat_values[order], seed_ids[order], base_weights[order]
    )
    result = np.empty((len(counts), len(quantiles)), dtype=float)
    for start in range(0, len(counts), batch_size):
        stop = min(start + batch_size, len(counts))
        row_weights = counts[start:stop, seed_ids] * base_weights
        cumulative = np.cumsum(row_weights, axis=1)
        totals = cumulative[:, -1]
        for column, quantile in enumerate(quantiles):
            targets = quantile * totals
            indices = (cumulative >= targets[:, None]).argmax(axis=1)
            result[start:stop, column] = flat_values[indices]
    return result


def _pooled_quantiles_from_seed_counts(
    values_a: list[np.ndarray], weights_a: list[np.ndarray], counts_a: np.ndarray,
    values_b: list[np.ndarray], weights_b: list[np.ndarray], counts_b: np.ndarray,
    quantiles: Sequence[float] = (.25, .75), *, batch_size: int = 64,
) -> np.ndarray:
    va, vb = np.concatenate(values_a), np.concatenate(values_b)
    ia = np.concatenate([np.full(len(v), i, dtype=int) for i, v in enumerate(values_a)])
    ib = np.concatenate([np.full(len(v), i, dtype=int) for i, v in enumerate(values_b)])
    wa, wb = np.concatenate(weights_a), np.concatenate(weights_b)
    values = np.concatenate([va, vb])
    groups = np.concatenate([np.zeros(len(va), dtype=int), np.ones(len(vb), dtype=int)])
    seed_ids = np.concatenate([ia, ib])
    base = np.concatenate([wa / (2 * counts_a.shape[1]), wb / (2 * counts_b.shape[1])])
    order = np.argsort(values, kind="mergesort")
    values, groups, seed_ids, base = values[order], groups[order], seed_ids[order], base[order]
    result = np.empty((len(counts_a), len(quantiles)), dtype=float)
    for start in range(0, len(counts_a), batch_size):
        stop = min(start + batch_size, len(counts_a))
        row_weights = np.empty((stop - start, len(values)), dtype=float)
        a_rows = groups == 0
        row_weights[:, a_rows] = counts_a[start:stop, seed_ids[a_rows]] * base[a_rows]
        row_weights[:, ~a_rows] = counts_b[start:stop, seed_ids[~a_rows]] * base[~a_rows]
        cumulative = np.cumsum(row_weights, axis=1)
        totals = cumulative[:, -1]
        for column, quantile in enumerate(quantiles):
            indices = (cumulative >= (quantile * totals)[:, None]).argmax(axis=1)
            result[start:stop, column] = values[indices]
    return result


def seed_block_bootstrap(
    blocks_a: SeedBlock,
    blocks_b: SeedBlock,
    *,
    n_bootstrap: int,
    random_seed: int,
    return_samples: bool = False,
) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
    """Independently resample whole seeds in two all-survivor cohorts."""
    seeds_a, values_a, weights_a = _seed_components(blocks_a)
    seeds_b, values_b, weights_b = _seed_components(blocks_b)
    if not seeds_a or not seeds_b:
        raise ValueError("Seed bootstrap requires retained seeds in both groups")
    support = np.unique(np.concatenate([*values_a, *values_b]))
    widths = np.diff(support)
    cdf_a = _cdf_basis(values_a, weights_a, support)
    cdf_b = _cdf_basis(values_b, weights_b, support)
    rng = np.random.default_rng(random_seed)
    draws_a = rng.integers(0, len(seeds_a), size=(n_bootstrap, len(seeds_a)))
    draws_b = rng.integers(0, len(seeds_b), size=(n_bootstrap, len(seeds_b)))
    counts_a = np.stack([np.bincount(row, minlength=len(seeds_a)) for row in draws_a])
    counts_b = np.stack([np.bincount(row, minlength=len(seeds_b)) for row in draws_b])
    cdf_difference = counts_a / len(seeds_a) @ cdf_a - counts_b / len(seeds_b) @ cdf_b
    w1 = np.abs(cdf_difference) @ widths
    qa = _quantiles_from_seed_counts(values_a, weights_a, counts_a)
    qb = _quantiles_from_seed_counts(values_b, weights_b, counts_b)
    qp = _pooled_quantiles_from_seed_counts(
        values_a, weights_a, counts_a, values_b, weights_b, counts_b
    )
    iqr_a, iqr_b, pooled_iqr = qa[:, 2] - qa[:, 0], qb[:, 2] - qb[:, 0], qp[:, 1] - qp[:, 0]
    samples = np.column_stack([
        w1,
        qb[:, 1] - qa[:, 1],
        np.log2((iqr_b + IQR_EPSILON) / (iqr_a + IQR_EPSILON)),
        np.divide(w1, pooled_iqr, out=np.full_like(w1, np.nan), where=pooled_iqr > IQR_EPSILON),
    ])
    ci = np.nanquantile(samples, [.025, .975], axis=0)
    result = {
        "seed_balanced_W1_CI_low_A": float(ci[0, 0]),
        "seed_balanced_W1_CI_high_A": float(ci[1, 0]),
        "delta_weighted_median_CI_low_A": float(ci[0, 1]),
        "delta_weighted_median_CI_high_A": float(ci[1, 1]),
        "weighted_log2_IQR_ratio_CI_low": float(ci[0, 2]),
        "weighted_log2_IQR_ratio_CI_high": float(ci[1, 2]),
        "normalized_W1_CI_low": float(ci[0, 3]),
        "normalized_W1_CI_high": float(ci[1, 3]),
    }
    return (result, samples) if return_samples else result


def seed_block_permutation(
    blocks_a: SeedBlock,
    blocks_b: SeedBlock,
    *,
    n_permutations: int,
    random_seed: int,
) -> float:
    """Permute complete seeds between two unpaired all-survivor cohorts."""
    _, values_a, weights_a = _seed_components(blocks_a)
    _, values_b, weights_b = _seed_components(blocks_b)
    if not values_a or not values_b:
        raise ValueError("Seed permutation requires retained seeds in both groups")
    support = np.unique(np.concatenate([*values_a, *values_b]))
    widths = np.diff(support)
    cdf = np.vstack([
        _cdf_basis(values_a, weights_a, support),
        _cdf_basis(values_b, weights_b, support),
    ])
    n_a, total = len(values_a), len(values_a) + len(values_b)
    observed = seed_balanced_w1(blocks_a, blocks_b)
    rng = np.random.default_rng(random_seed)
    exceed = 0
    for _ in range(n_permutations):
        order = rng.permutation(total)
        difference = cdf[order[:n_a]].mean(axis=0) - cdf[order[n_a:]].mean(axis=0)
        exceed += float(np.abs(difference) @ widths) >= observed - 1e-15
    return float((1 + exceed) / (1 + n_permutations))


def analyze_seed_distance(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_col: str,
    *,
    n_permutations: int = 999,
    n_bootstrap: int = 500,
    random_seed: int = 20260816,
    within_trajectory_reduction: str | None = None,
) -> dict[str, object]:
    """Primary all-survivor geometry analysis with seed-block inference."""
    original_a, original_b = frame_a, frame_b
    if within_trajectory_reduction is not None:
        frame_a = reduce_trajectory_values(
            frame_a, value_col, reduction=within_trajectory_reduction
        )
        frame_b = reduce_trajectory_values(
            frame_b, value_col, reduction=within_trajectory_reduction
        )
    blocks_a, blocks_b = make_seed_blocks(frame_a, value_col), make_seed_blocks(frame_b, value_col)
    metrics = seed_distribution_metrics(blocks_a, blocks_b)
    ci = seed_block_bootstrap(
        blocks_a, blocks_b, n_bootstrap=n_bootstrap, random_seed=random_seed + 1
    )
    return {
        "distance": value_col,
        "analysis_level": (
            f"seed_balanced_{within_trajectory_reduction}_per_trajectory_all_survivors"
            if within_trajectory_reduction is not None
            else "seed_balanced_all_survivors"
        ),
        "within_trajectory_reduction": within_trajectory_reduction or "equal-weight snapshots",
        "n_seeds_A": len(blocks_a),
        "n_seeds_B": len(blocks_b),
        "n_trajectories_A": int(original_a.loc[pd.to_numeric(original_a[value_col], errors="coerce").notna(), ["seed", "model_number"]].drop_duplicates().shape[0]),
        "n_trajectories_B": int(original_b.loc[pd.to_numeric(original_b[value_col], errors="coerce").notna(), ["seed", "model_number"]].drop_duplicates().shape[0]),
        **metrics,
        **ci,
        "p_W1_seed_permutation": seed_block_permutation(
            blocks_a, blocks_b, n_permutations=n_permutations, random_seed=random_seed
        ),
        "resampling_unit": "seed; AF2 model is a fixed within-seed stratum",
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
    }


def leave_one_model_out(
    frame_a: pd.DataFrame, frame_b: pd.DataFrame, value_col: str,
    *, within_trajectory_reduction: str | None = None,
) -> pd.DataFrame:
    """Report seed-balanced point estimates after omitting each AF2 model."""
    if within_trajectory_reduction is not None:
        frame_a = reduce_trajectory_values(
            frame_a, value_col, reduction=within_trajectory_reduction
        )
        frame_b = reduce_trajectory_values(
            frame_b, value_col, reduction=within_trajectory_reduction
        )
    models = sorted(set(frame_a["model_number"]).union(frame_b["model_number"]))
    rows = []
    for omitted in models:
        a = frame_a[frame_a["model_number"].ne(omitted)]
        b = frame_b[frame_b["model_number"].ne(omitted)]
        metrics = seed_distribution_metrics(make_seed_blocks(a, value_col), make_seed_blocks(b, value_col))
        rows.append({"omitted_model_number": omitted, **metrics})
    return pd.DataFrame(rows)
