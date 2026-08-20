import numpy as np
import pandas as pd
import pytest

from shared.seed_block_statistics import (
    analyze_seed_distance,
    leave_one_model_out,
    make_seed_blocks,
    reduce_trajectory_values,
    seed_block_bootstrap,
    seed_distribution_metrics,
    seed_distribution_metrics_matrix,
    seed_model_stratum_weights,
    seed_model_weights,
    weighted_quantile,
)


def synthetic_frame(offset=0.0):
    rows = []
    for seed in range(1, 9):
        for model in (1, 2):
            for recycle in range(1, 1 + (3 if model == 1 else 1)):
                rows.append({
                    "seed": seed,
                    "model_number": model,
                    "value": offset + seed * .1 + model + recycle * .01,
                })
    return pd.DataFrame(rows)


def test_seed_model_weights_balance_all_three_levels():
    frame = synthetic_frame()
    weights = seed_model_weights(frame, "value")
    assert weights.sum() == pytest.approx(1.0)
    by_seed = weights.groupby(frame.loc[weights.index, "seed"]).sum()
    assert by_seed.tolist() == pytest.approx([1 / 8] * 8)
    by_model = weights.groupby([
        frame.loc[weights.index, "seed"], frame.loc[weights.index, "model_number"]
    ]).sum()
    assert by_model.tolist() == pytest.approx([1 / 16] * 16)


def test_weighted_quantile_uses_weights():
    assert weighted_quantile([0, 10], .5, [.9, .1]) == 0
    assert weighted_quantile([0, 10], [.25, .75], [.1, .9]).tolist() == [10, 10]


def test_seed_metrics_use_consistent_weighted_quantiles():
    a, b = synthetic_frame(), synthetic_frame(3)
    metrics = seed_distribution_metrics(make_seed_blocks(a, "value"), make_seed_blocks(b, "value"))
    assert metrics["seed_balanced_W1_A"] == pytest.approx(3.0)
    assert metrics["delta_weighted_median_A"] == pytest.approx(3.0)
    assert metrics["weighted_IQR_A_A"] == pytest.approx(metrics["weighted_IQR_B_A"])


def test_seed_bootstrap_is_deterministic_and_resamples_seeds():
    a, b = synthetic_frame(), synthetic_frame(2)
    blocks_a, blocks_b = make_seed_blocks(a, "value"), make_seed_blocks(b, "value")
    first, samples = seed_block_bootstrap(
        blocks_a, blocks_b, n_bootstrap=40, random_seed=9, return_samples=True
    )
    second = seed_block_bootstrap(blocks_a, blocks_b, n_bootstrap=40, random_seed=9)
    assert first == second
    assert samples.shape == (40, 4)
    assert np.isfinite(samples).all()


def test_primary_analysis_and_leave_one_model_out():
    a, b = synthetic_frame(), synthetic_frame(4)
    result = analyze_seed_distance(
        a, b, "value", n_permutations=99, n_bootstrap=30, random_seed=4
    )
    assert result["analysis_level"] == "seed_balanced_all_survivors"
    assert result["n_seeds_A"] == 8
    assert result["seed_balanced_W1_A"] == pytest.approx(4.0)
    sensitivity = leave_one_model_out(a, b, "value")
    assert sensitivity.omitted_model_number.tolist() == [1, 2]
    assert sensitivity.seed_balanced_W1_A.tolist() == pytest.approx([4.0, 4.0])


def test_trajectory_median_reduction_and_matrix_metrics_match_scalar_metrics():
    a = reduce_trajectory_values(synthetic_frame(), "value", reduction="median")
    b = reduce_trajectory_values(synthetic_frame(2.5), "value", reduction="median")
    scalar = seed_distribution_metrics(make_seed_blocks(a, "value"), make_seed_blocks(b, "value"))
    matrix = seed_distribution_metrics_matrix(a, b, ["value"]).iloc[0]
    for key in (
        "weighted_median_A_A", "weighted_median_B_A", "delta_weighted_median_A",
        "weighted_IQR_A_A", "weighted_IQR_B_A", "weighted_IQR_ratio_B_over_A",
        "weighted_log2_IQR_ratio", "weighted_pooled_IQR_A", "seed_balanced_W1_A",
        "seed_balanced_W1_normalized_by_weighted_pooled_IQR",
    ):
        assert matrix[key] == pytest.approx(scalar[key])


def test_matrix_metrics_preserve_equal_seed_mass_with_missing_model_strata():
    a = pd.DataFrame({
        "seed": [1, 1, 2], "model_number": [1, 2, 1], "value": [0.0, 2.0, 4.0]
    })
    b = a.assign(value=a["value"] + 1.0)
    assert seed_model_stratum_weights(a).tolist() == pytest.approx([.25, .25, .5])
    scalar = seed_distribution_metrics(make_seed_blocks(a, "value"), make_seed_blocks(b, "value"))
    matrix = seed_distribution_metrics_matrix(a, b, ["value"]).iloc[0]
    assert matrix["seed_balanced_W1_A"] == pytest.approx(scalar["seed_balanced_W1_A"])
    assert matrix["delta_weighted_median_A"] == pytest.approx(scalar["delta_weighted_median_A"])


def test_primary_analysis_can_use_trajectory_median_as_single_estimand():
    result = analyze_seed_distance(
        synthetic_frame(), synthetic_frame(1.5), "value",
        n_permutations=19, n_bootstrap=20, random_seed=7,
        within_trajectory_reduction="median",
    )
    assert result["analysis_level"] == "seed_balanced_median_per_trajectory_all_survivors"
    assert result["within_trajectory_reduction"] == "median"
    assert result["seed_balanced_W1_A"] == pytest.approx(1.5)
