"""Joint nominal-seed inference for paired AlphaFold ensemble designs.

The functions in this module keep the legacy seed-block implementation intact
while making the revised estimands explicit:

``primary_joint_nominal_seed``
    Resample the union of nominal seed labels once per replicate.  Each
    condition keeps its own QC survivors, including condition-specific missing
    seeds and model strata.
``common_seed``
    Restrict to seed labels represented in both conditions, then resample the
    common labels jointly.
``common_model_seed``
    Restrict to seed/model strata represented in both conditions, then
    resample their common seed labels jointly.

Numeric seed labels are pairing keys, not proof that the underlying random
seed values were recorded.  Callers must report that provenance separately.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from shared.seed_block_statistics import (
    IQR_EPSILON,
    make_seed_blocks,
    seed_distribution_metrics,
    seed_weighted_samples,
    weighted_quantile,
)


ESTIMANDS = (
    "primary_joint_nominal_seed",
    "common_seed",
    "common_model_seed",
)


def low_pooled_iqr_flags(pooled_iqr: float) -> dict[str, bool]:
    """Flag normalized-W1 denominators below prespecified ångström cutoffs."""
    return {
        "pooled_IQR_below_0.05A": bool(pooled_iqr < 0.05),
        "pooled_IQR_below_0.10A": bool(pooled_iqr < 0.10),
        "pooled_IQR_below_0.25A": bool(pooled_iqr < 0.25),
    }


def _seed_scalar(frame: pd.DataFrame, value_col: str) -> pd.Series:
    """Equal-model mean within seed for one-row-per-trajectory input."""
    work = _validate_frame(frame, value_col)
    if work.duplicated(["seed", "model_number"]).any():
        raise ValueError("Scalar paired inference requires one summary per seed/model trajectory")
    return work.groupby("seed")[value_col].mean().sort_index()


def _joint_scalar_samples(
    values: Mapping[str, pd.Series], *, n_bootstrap: int, random_seed: int
) -> tuple[dict[str, float], np.ndarray, list[int]]:
    labels = sorted(set().union(*(set(series.index) for series in values.values())))
    if not labels or any(series.empty for series in values.values()):
        raise ValueError("Every condition needs at least one contributing nominal seed")
    rng = np.random.default_rng(random_seed)
    samples = np.empty((n_bootstrap, len(values)), dtype=float)
    names = list(values)
    for index in range(n_bootstrap):
        draw = rng.choice(labels, size=len(labels), replace=True)
        for column, name in enumerate(names):
            series = values[name]
            retained = series.reindex(draw).dropna()
            samples[index, column] = retained.mean() if len(retained) else np.nan
    return {name: float(values[name].mean()) for name in names}, samples, labels


def joint_seed_bootstrap_contrast(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_col: str,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, object]:
    """Primary B-A contrast with one joint draw of nominal seed labels."""
    a, b = _seed_scalar(frame_a, value_col), _seed_scalar(frame_b, value_col)
    points, samples, labels = _joint_scalar_samples(
        {"A": a, "B": b}, n_bootstrap=n_bootstrap, random_seed=random_seed
    )
    delta = samples[:, 1] - samples[:, 0]
    low, high = np.nanquantile(delta, [.025, .975])
    return {
        "estimate_A": points["A"],
        "estimate_B": points["B"],
        "estimate_B_minus_A": points["B"] - points["A"],
        "CI_low_B_minus_A": float(low),
        "CI_high_B_minus_A": float(high),
        "n_nominal_seed_labels_union": len(labels),
        "n_contributing_seeds_A": len(a),
        "n_contributing_seeds_B": len(b),
        "n_common_contributing_seeds": len(set(a.index) & set(b.index)),
        "n_nonfinite_bootstrap_replicates": int((~np.isfinite(delta)).sum()),
        "bootstrap_replicates": n_bootstrap,
        "random_seed": random_seed,
        "estimand": "primary marginal QC-qualified contrast with joint nominal-seed resampling",
    }


def paired_common_seed_bootstrap(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_col: str,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, object]:
    """Bootstrap paired B-A seed differences over common contributing seeds."""
    a, b = _seed_scalar(frame_a, value_col), _seed_scalar(frame_b, value_col)
    common = sorted(set(a.index) & set(b.index))
    if not common:
        raise ValueError("No common contributing seeds")
    differences = b.loc[common].to_numpy(float) - a.loc[common].to_numpy(float)
    rng = np.random.default_rng(random_seed)
    samples = differences[
        rng.integers(0, len(differences), size=(n_bootstrap, len(differences)))
    ].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return {
        "estimate_B_minus_A": float(differences.mean()),
        "CI_low_B_minus_A": float(low),
        "CI_high_B_minus_A": float(high),
        "n_common_contributing_seeds": len(common),
        "bootstrap_replicates": n_bootstrap,
        "random_seed": random_seed,
        "estimand": "common-contributing-seed paired contrast",
    }


def paired_seed_ratio_of_means(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_col: str,
    *,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, object]:
    """Ratio of seed-balanced protocol means without pseudocounts."""
    a, b = _seed_scalar(frame_a, value_col), _seed_scalar(frame_b, value_col)
    points, samples, labels = _joint_scalar_samples(
        {"A": a, "B": b}, n_bootstrap=n_bootstrap, random_seed=random_seed
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = samples[:, 1] / samples[:, 0]
        point = np.divide(np.float64(points["B"]), np.float64(points["A"]))
    finite = ratios[np.isfinite(ratios)]
    low, high = (np.quantile(finite, [.025, .975]) if len(finite) else (np.nan, np.nan))
    return {
        "ratio_of_seed_balanced_protocol_sampling_fractions": float(point),
        "ratio_CI_low": float(low),
        "ratio_CI_high": float(high),
        "absolute_percentage_point_difference": 100 * (points["B"] - points["A"]),
        "n_nominal_seed_labels_union": len(labels),
        "nonfinite_bootstrap_replicates": int((~np.isfinite(ratios)).sum()),
        "nonfinite_bootstrap_fraction": float((~np.isfinite(ratios)).mean()),
        "bootstrap_replicates": n_bootstrap,
        "random_seed": random_seed,
        "pseudocount": "none",
    }


def paired_seed_retention_contrast(
    passed_a: pd.DataFrame,
    passed_b: pd.DataFrame,
    *,
    pass_col: str,
    nominal_seeds_a: Sequence[int],
    nominal_seeds_b: Sequence[int] | None = None,
    nominal_models: Sequence[int] = (1, 2, 3, 4, 5),
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, object]:
    """Joint-seed retention contrast with failed/missing nominal strata set to zero."""
    def complete(frame: pd.DataFrame, nominal_seeds: Sequence[int]) -> pd.DataFrame:
        required = {"seed", "model_number", pass_col}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Missing retention columns: {missing}")
        index = pd.MultiIndex.from_product(
            [list(nominal_seeds), list(nominal_models)], names=["seed", "model_number"]
        )
        values = frame.groupby(["seed", "model_number"])[pass_col].max()
        return values.reindex(index, fill_value=False).astype(float).rename(pass_col).reset_index()

    nominal_seeds_b = nominal_seeds_a if nominal_seeds_b is None else nominal_seeds_b
    a, b = complete(passed_a, nominal_seeds_a), complete(passed_b, nominal_seeds_b)
    result = joint_seed_bootstrap_contrast(
        a, b, pass_col, n_bootstrap=n_bootstrap, random_seed=random_seed
    )
    result.update({
        "nominal_models_per_seed": len(nominal_models),
        "nominal_seeds_A": len(set(nominal_seeds_a)),
        "nominal_seeds_B": len(set(nominal_seeds_b)),
        "missing_or_failed_policy": "zero in the nominal denominator",
        "estimand": "paired nominal-trajectory retention contrast",
    })
    return result


def paired_seed_categorical_vector_contrast(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    *,
    category_col: str,
    categories: Sequence[object],
    n_bootstrap: int,
    random_seed: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Paired category-vector comparison with TV and Jensen-Shannon divergence."""
    required = {"seed", "model_number", category_col}
    for frame in (frame_a, frame_b):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise KeyError(f"Missing categorical columns: {missing}")

    def seed_vectors(frame: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (seed, model), part in frame.groupby(["seed", "model_number"]):
            probabilities = part[category_col].value_counts(normalize=True)
            rows.append({"seed": seed, "model_number": model, **{
                str(category): float(probabilities.get(category, 0)) for category in categories
            }})
        return pd.DataFrame(rows).groupby("seed")[[str(x) for x in categories]].mean()

    a, b = seed_vectors(frame_a), seed_vectors(frame_b)
    labels = sorted(set(a.index) | set(b.index))
    point_a, point_b = a.mean().to_numpy(float), b.mean().to_numpy(float)

    def divergences(pa: np.ndarray, pb: np.ndarray) -> tuple[float, float]:
        tv = 0.5 * np.abs(pb - pa).sum()
        midpoint = 0.5 * (pa + pb)
        with np.errstate(divide="ignore", invalid="ignore"):
            kl_a = np.where(pa > 0, pa * np.log2(pa / midpoint), 0).sum()
            kl_b = np.where(pb > 0, pb * np.log2(pb / midpoint), 0).sum()
        return float(tv), float(0.5 * (kl_a + kl_b))

    rng = np.random.default_rng(random_seed)
    samples = np.empty((n_bootstrap, 2 + len(categories)), dtype=float)
    for index in range(n_bootstrap):
        draw = rng.choice(labels, len(labels), replace=True)
        pa = a.reindex(draw).dropna().mean().to_numpy(float)
        pb = b.reindex(draw).dropna().mean().to_numpy(float)
        samples[index, :2] = divergences(pa, pb)
        samples[index, 2:] = pb - pa
    tv, js = divergences(point_a, point_b)
    intervals = np.nanquantile(samples, [.025, .975], axis=0)
    summary = {
        "total_variation_distance": tv,
        "total_variation_CI_low": float(intervals[0, 0]),
        "total_variation_CI_high": float(intervals[1, 0]),
        "jensen_shannon_divergence_bits": js,
        "jensen_shannon_CI_low": float(intervals[0, 1]),
        "jensen_shannon_CI_high": float(intervals[1, 1]),
        "n_nominal_seed_labels_union": len(labels),
        "bootstrap_replicates": n_bootstrap,
        "random_seed": random_seed,
    }
    category_table = pd.DataFrame({
        "category": list(categories),
        "probability_A": point_a,
        "probability_B": point_b,
        "difference_B_minus_A": point_b - point_a,
        "difference_CI_low": intervals[0, 2:],
        "difference_CI_high": intervals[1, 2:],
    })
    return summary, category_table


def _validate_frame(frame: pd.DataFrame, value_col: str) -> pd.DataFrame:
    required = {"seed", "model_number", value_col}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing paired-seed columns: {missing}")
    work = frame.copy()
    work["seed"] = pd.to_numeric(work["seed"], errors="raise").astype(int)
    work["model_number"] = pd.to_numeric(
        work["model_number"], errors="raise"
    ).astype(int)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    return work.dropna(subset=[value_col])


def select_paired_estimand(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_col: str,
    *,
    estimand: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int | str]]:
    """Apply one of the three prespecified paired-survivor estimands."""
    if estimand not in ESTIMANDS:
        raise ValueError(f"estimand must be one of {ESTIMANDS}; received {estimand!r}")
    a, b = _validate_frame(frame_a, value_col), _validate_frame(frame_b, value_col)
    original_seeds_a, original_seeds_b = set(a.seed), set(b.seed)
    original_pairs_a = set(zip(a.seed, a.model_number))
    original_pairs_b = set(zip(b.seed, b.model_number))

    if estimand == "common_seed":
        common = original_seeds_a & original_seeds_b
        a, b = a[a.seed.isin(common)], b[b.seed.isin(common)]
    elif estimand == "common_model_seed":
        common_pairs = original_pairs_a & original_pairs_b
        a = a[[pair in common_pairs for pair in zip(a.seed, a.model_number)]]
        b = b[[pair in common_pairs for pair in zip(b.seed, b.model_number)]]

    selected_seeds_a, selected_seeds_b = set(a.seed), set(b.seed)
    selected_pairs_a = set(zip(a.seed, a.model_number))
    selected_pairs_b = set(zip(b.seed, b.model_number))
    audit: dict[str, int | str] = {
        "estimand": estimand,
        "n_seed_labels_A_before": len(original_seeds_a),
        "n_seed_labels_B_before": len(original_seeds_b),
        "n_seed_labels_union_before": len(original_seeds_a | original_seeds_b),
        "n_seed_labels_common_before": len(original_seeds_a & original_seeds_b),
        "n_model_seed_pairs_A_before": len(original_pairs_a),
        "n_model_seed_pairs_B_before": len(original_pairs_b),
        "n_model_seed_pairs_common_before": len(original_pairs_a & original_pairs_b),
        "n_seed_labels_A_analyzed": len(selected_seeds_a),
        "n_seed_labels_B_analyzed": len(selected_seeds_b),
        "n_seed_labels_union_analyzed": len(selected_seeds_a | selected_seeds_b),
        "n_seed_labels_common_analyzed": len(selected_seeds_a & selected_seeds_b),
        "n_model_seed_pairs_A_analyzed": len(selected_pairs_a),
        "n_model_seed_pairs_B_analyzed": len(selected_pairs_b),
        "n_model_seed_pairs_common_analyzed": len(selected_pairs_a & selected_pairs_b),
    }
    return a.reset_index(drop=True), b.reset_index(drop=True), audit


def _weighted_sample_for_counts(
    blocks: Mapping[int, Mapping[int, np.ndarray]],
    counts: Mapping[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    values: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    total = sum(counts.get(seed, 0) for seed in blocks)
    if total <= 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    for seed in sorted(blocks):
        multiplicity = counts.get(seed, 0)
        if multiplicity <= 0:
            continue
        models = [
            np.asarray(blocks[seed][model], dtype=float)
            for model in sorted(blocks[seed])
            if len(blocks[seed][model])
        ]
        for model_values in models:
            values.append(model_values)
            weights.append(
                np.full(
                    len(model_values),
                    multiplicity / (total * len(models) * len(model_values)),
                )
            )
    return np.concatenate(values), np.concatenate(weights)


def _metrics_for_counts(
    blocks_a: Mapping[int, Mapping[int, np.ndarray]],
    blocks_b: Mapping[int, Mapping[int, np.ndarray]],
    counts: Mapping[int, int],
) -> np.ndarray:
    a, wa = _weighted_sample_for_counts(blocks_a, counts)
    b, wb = _weighted_sample_for_counts(blocks_b, counts)
    if not len(a) or not len(b):
        return np.full(4, np.nan)
    qa = weighted_quantile(a, [.25, .5, .75], wa)
    qb = weighted_quantile(b, [.25, .5, .75], wb)
    qp = weighted_quantile(
        np.concatenate([a, b]), [.25, .75], np.concatenate([wa * .5, wb * .5])
    )
    w1 = float(wasserstein_distance(a, b, u_weights=wa, v_weights=wb))
    pooled_iqr = qp[1] - qp[0]
    return np.array([
        w1,
        qb[1] - qa[1],
        np.log2(((qb[2] - qb[0]) + IQR_EPSILON) / ((qa[2] - qa[0]) + IQR_EPSILON)),
        w1 / pooled_iqr if pooled_iqr > IQR_EPSILON else np.nan,
    ])


def joint_seed_bootstrap(
    blocks_a: Mapping[int, Mapping[int, np.ndarray]],
    blocks_b: Mapping[int, Mapping[int, np.ndarray]],
    *,
    n_bootstrap: int,
    random_seed: int,
    return_samples: bool = False,
) -> dict[str, float | int] | tuple[dict[str, float | int], np.ndarray]:
    """Jointly resample nominal labels while retaining each cohort's survivors."""
    labels = sorted(set(blocks_a) | set(blocks_b))
    if not labels or not blocks_a or not blocks_b:
        raise ValueError("Joint bootstrap requires retained observations in both groups")
    rng = np.random.default_rng(random_seed)
    samples = np.empty((n_bootstrap, 4), dtype=float)
    invalid = 0
    for index in range(n_bootstrap):
        draw = rng.choice(labels, size=len(labels), replace=True)
        unique, multiplicities = np.unique(draw, return_counts=True)
        counts = dict(zip(unique.tolist(), multiplicities.tolist()))
        samples[index] = _metrics_for_counts(blocks_a, blocks_b, counts)
        invalid += int(not np.isfinite(samples[index, :3]).all())
    ci = np.nanquantile(samples, [.025, .975], axis=0)
    result: dict[str, float | int] = {
        "seed_balanced_W1_CI_low_A": float(ci[0, 0]),
        "seed_balanced_W1_CI_high_A": float(ci[1, 0]),
        "delta_weighted_median_CI_low_A": float(ci[0, 1]),
        "delta_weighted_median_CI_high_A": float(ci[1, 1]),
        "weighted_log2_IQR_ratio_CI_low": float(ci[0, 2]),
        "weighted_log2_IQR_ratio_CI_high": float(ci[1, 2]),
        "normalized_W1_CI_low": float(ci[0, 3]),
        "normalized_W1_CI_high": float(ci[1, 3]),
        "n_invalid_bootstrap_replicates": invalid,
    }
    return (result, samples) if return_samples else result


def paired_seed_permutation(
    blocks_a: Mapping[int, Mapping[int, np.ndarray]],
    blocks_b: Mapping[int, Mapping[int, np.ndarray]],
    *,
    n_permutations: int,
    random_seed: int,
) -> float:
    """Swap complete condition blocks within nominal labels for a paired W1 test."""
    labels = sorted(set(blocks_a) | set(blocks_b))
    if not labels or not blocks_a or not blocks_b:
        raise ValueError("Paired permutation requires retained observations in both groups")
    observed = seed_distribution_metrics(blocks_a, blocks_b)["seed_balanced_W1_A"]
    rng = np.random.default_rng(random_seed)
    exceed = 0
    valid = 0
    for _ in range(n_permutations):
        permuted_a: dict[int, Mapping[int, np.ndarray]] = {}
        permuted_b: dict[int, Mapping[int, np.ndarray]] = {}
        for label, swap in zip(labels, rng.integers(0, 2, size=len(labels))):
            source_a = blocks_b if swap else blocks_a
            source_b = blocks_a if swap else blocks_b
            if label in source_a:
                permuted_a[label] = source_a[label]
            if label in source_b:
                permuted_b[label] = source_b[label]
        if not permuted_a or not permuted_b:
            continue
        valid += 1
        statistic = seed_distribution_metrics(permuted_a, permuted_b)["seed_balanced_W1_A"]
        exceed += statistic >= float(observed) - 1e-15
    if not valid:
        return np.nan
    return float((1 + exceed) / (1 + valid))


def analyze_paired_seed_distance(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    value_col: str,
    *,
    estimand: str = "primary_joint_nominal_seed",
    n_permutations: int = 999,
    n_bootstrap: int = 500,
    random_seed: int = 20260824,
) -> dict[str, object]:
    """Analyze one distance using a declared paired-survivor estimand."""
    a, b, audit = select_paired_estimand(frame_a, frame_b, value_col, estimand=estimand)
    blocks_a, blocks_b = make_seed_blocks(a, value_col), make_seed_blocks(b, value_col)
    if not blocks_a or not blocks_b:
        raise ValueError("Selected estimand has no retained values in one or both groups")
    return {
        "distance": value_col,
        "analysis_level": estimand,
        **audit,
        **seed_distribution_metrics(blocks_a, blocks_b),
        **joint_seed_bootstrap(
            blocks_a,
            blocks_b,
            n_bootstrap=n_bootstrap,
            random_seed=random_seed + 1,
        ),
        "p_W1_paired_seed_permutation": paired_seed_permutation(
            blocks_a,
            blocks_b,
            n_permutations=n_permutations,
            random_seed=random_seed,
        ),
        "resampling_unit": "joint nominal seed label; AF2 model is a fixed within-seed stratum",
        "actual_random_seed_pairing_status": "not verified; numeric nominal seed labels only",
        "n_permutations": n_permutations,
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
    }


def _condition_statistic(
    frame: pd.DataFrame, value_col: str, counts: Mapping[int, int], statistic: str
) -> float:
    values, weights = _weighted_sample_for_counts(make_seed_blocks(frame, value_col), counts)
    if not len(values):
        return np.nan
    if statistic == "mean":
        return float(np.average(values, weights=weights))
    if statistic == "median":
        return float(weighted_quantile(values, .5, weights))
    raise ValueError("statistic must be 'mean' or 'median'")


def paired_factorial_interaction(
    conditions: Mapping[str, pd.DataFrame],
    value_col: str,
    *,
    order: Sequence[str] = ("wt_vanilla", "wt_masked", "mutant_vanilla", "mutant_masked"),
    statistic: str = "mean",
    n_bootstrap: int = 500,
    random_seed: int = 20260824,
) -> dict[str, object]:
    """Bootstrap complete-seed (mutant mask effect) - (WT mask effect)."""
    if set(order) - set(conditions):
        raise KeyError(f"Missing factorial conditions: {sorted(set(order) - set(conditions))}")
    frames = {name: _validate_frame(conditions[name], value_col) for name in order}
    seed_values = {name: _seed_scalar(frame, value_col) for name, frame in frames.items()}
    nominal_union = sorted(set().union(*(set(series.index) for series in seed_values.values())))
    complete = sorted(set.intersection(*(set(series.index) for series in seed_values.values())))
    if not complete:
        raise ValueError("Factorial interaction requires at least one complete four-cell seed")
    matrix = np.column_stack([seed_values[name].loc[complete].to_numpy(float) for name in order])
    per_seed = (matrix[:, 3] - matrix[:, 2]) - (matrix[:, 1] - matrix[:, 0])
    values = matrix.mean(axis=0).tolist()
    point = float(per_seed.mean())
    rng = np.random.default_rng(random_seed)
    samples = per_seed[
        rng.integers(0, len(per_seed), size=(n_bootstrap, len(per_seed)))
    ].mean(axis=1)
    low, high = np.nanquantile(samples, [.025, .975])
    return {
        "statistic": statistic,
        "wt_vanilla": values[0],
        "wt_masked": values[1],
        "mutant_vanilla": values[2],
        "mutant_masked": values[3],
        "wt_mask_effect": values[1] - values[0],
        "mutant_mask_effect": values[3] - values[2],
        "masking_by_sequence_interaction": point,
        "interaction_CI_low": float(low),
        "interaction_CI_high": float(high),
        "n_nominal_seed_labels_union": len(nominal_union),
        "n_complete_four_cell_seeds": len(complete),
        "model_seed_trajectories_wt_vanilla": len(frames[order[0]]),
        "model_seed_trajectories_wt_masked": len(frames[order[1]]),
        "model_seed_trajectories_mutant_vanilla": len(frames[order[2]]),
        "model_seed_trajectories_mutant_masked": len(frames[order[3]]),
        "n_bootstrap": n_bootstrap,
        "random_seed": random_seed,
        "resampling_unit": "complete four-cell nominal seed; bootstrap of within-seed interactions",
        "actual_random_seed_pairing_status": "not verified; complete overlap of recorded numeric seed labels only",
    }
