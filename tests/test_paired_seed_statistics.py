import numpy as np
import pandas as pd
import pytest

from shared.paired_seed_statistics import (
    analyze_paired_seed_distance,
    joint_seed_bootstrap_contrast,
    joint_seed_bootstrap,
    low_pooled_iqr_flags,
    paired_common_seed_bootstrap,
    paired_factorial_interaction,
    paired_seed_categorical_vector_contrast,
    paired_seed_permutation,
    paired_seed_ratio_of_means,
    paired_seed_retention_contrast,
    select_paired_estimand,
)
from shared.seed_block_statistics import make_seed_blocks, seed_block_bootstrap
from analysis.statistics_revision.scripts.run_paired_seed_v2 import g402_common_ca_columns


def frame(offset=0.0, seeds=range(1, 9), models=(1, 2)):
    return pd.DataFrame([
        {"seed": seed, "model_number": model, "value": seed + model / 10 + offset}
        for seed in seeds for model in models
    ])


def test_estimand_selection_preserves_or_intersects_survivors():
    a = frame(seeds=(1, 2, 3), models=(1, 2))
    b = frame(seeds=(2, 3, 4), models=(2, 3))
    primary_a, primary_b, audit = select_paired_estimand(
        a, b, "value", estimand="primary_joint_nominal_seed"
    )
    assert set(primary_a.seed) == {1, 2, 3}
    assert set(primary_b.seed) == {2, 3, 4}
    assert audit["n_seed_labels_union_before"] == 4

    common_a, common_b, _ = select_paired_estimand(a, b, "value", estimand="common_seed")
    assert set(common_a.seed) == set(common_b.seed) == {2, 3}

    model_a, model_b, model_audit = select_paired_estimand(
        a, b, "value", estimand="common_model_seed"
    )
    assert set(zip(model_a.seed, model_a.model_number)) == {(2, 2), (3, 2)}
    assert set(zip(model_b.seed, model_b.model_number)) == {(2, 2), (3, 2)}
    assert model_audit["n_model_seed_pairs_common_analyzed"] == 2


def test_joint_bootstrap_uses_one_nominal_seed_draw_for_both_conditions():
    a, b = frame(), frame(offset=2.0)
    result, samples = joint_seed_bootstrap(
        make_seed_blocks(a, "value"),
        make_seed_blocks(b, "value"),
        n_bootstrap=40,
        random_seed=11,
        return_samples=True,
    )
    assert samples[:, 0].tolist() == pytest.approx([2.0] * 40)
    assert samples[:, 1].tolist() == pytest.approx([2.0] * 40)
    assert result["n_invalid_bootstrap_replicates"] == 0


def test_primary_joint_bootstrap_preserves_condition_specific_missing_seeds():
    a = frame(seeds=(1, 2, 3))
    b = frame(offset=1, seeds=(2, 3, 4))
    first, samples = joint_seed_bootstrap(
        make_seed_blocks(a, "value"), make_seed_blocks(b, "value"),
        n_bootstrap=50, random_seed=3, return_samples=True,
    )
    second = joint_seed_bootstrap(
        make_seed_blocks(a, "value"), make_seed_blocks(b, "value"),
        n_bootstrap=50, random_seed=3,
    )
    assert first == second
    assert np.isfinite(samples[:, :3]).any(axis=1).all()
    assert first["n_invalid_bootstrap_replicates"] >= 0


def test_paired_permutation_is_deterministic_and_detects_large_shift():
    a, b = frame(), frame(offset=20)
    blocks_a, blocks_b = make_seed_blocks(a, "value"), make_seed_blocks(b, "value")
    first = paired_seed_permutation(
        blocks_a, blocks_b, n_permutations=199, random_seed=14
    )
    second = paired_seed_permutation(
        blocks_a, blocks_b, n_permutations=199, random_seed=14
    )
    assert first == second
    assert first <= 0.02


def test_analysis_reports_pairing_provenance_caveat():
    result = analyze_paired_seed_distance(
        frame(), frame(offset=1.5), "value",
        estimand="common_model_seed", n_bootstrap=25, n_permutations=39,
    )
    assert result["analysis_level"] == "common_model_seed"
    assert result["seed_balanced_W1_A"] == pytest.approx(1.5)
    assert result["actual_random_seed_pairing_status"].startswith("not verified")


def test_joint_factorial_interaction_recovers_difference_in_differences():
    conditions = {
        "wt_vanilla": frame(offset=0),
        "wt_masked": frame(offset=1),
        "mutant_vanilla": frame(offset=2),
        "mutant_masked": frame(offset=5),
    }
    result = paired_factorial_interaction(
        conditions, "value", statistic="mean", n_bootstrap=30, random_seed=8
    )
    assert result["wt_mask_effect"] == pytest.approx(1)
    assert result["mutant_mask_effect"] == pytest.approx(3)
    assert result["masking_by_sequence_interaction"] == pytest.approx(2)
    assert result["interaction_CI_low"] == pytest.approx(2)
    assert result["interaction_CI_high"] == pytest.approx(2)


def test_identical_conditions_and_known_scalar_contrast():
    identical = joint_seed_bootstrap_contrast(
        frame(), frame(), "value", n_bootstrap=30, random_seed=1
    )
    shifted = paired_common_seed_bootstrap(
        frame(), frame(offset=4), "value", n_bootstrap=30, random_seed=1
    )
    assert identical["estimate_B_minus_A"] == pytest.approx(0)
    assert identical["CI_low_B_minus_A"] == pytest.approx(0)
    assert shifted["estimate_B_minus_A"] == pytest.approx(4)
    assert shifted["CI_high_B_minus_A"] == pytest.approx(4)


def test_ratio_records_nonfinite_replicates_without_pseudocount():
    a = frame().assign(value=0.0)
    b = frame().assign(value=1.0)
    result = paired_seed_ratio_of_means(
        a, b, "value", n_bootstrap=25, random_seed=2
    )
    assert np.isinf(result["ratio_of_seed_balanced_protocol_sampling_fractions"])
    assert result["nonfinite_bootstrap_replicates"] == 25
    assert result["pseudocount"] == "none"


def test_retention_uses_all_five_nominal_models_and_missing_is_zero():
    a = pd.DataFrame({"seed": [1, 2], "model_number": [1, 1], "passed": [1, 1]})
    b = pd.DataFrame({"seed": [1, 2], "model_number": [1, 2], "passed": [1, 1]})
    result = paired_seed_retention_contrast(
        a, b, pass_col="passed", nominal_seeds_a=[1, 2],
        n_bootstrap=20, random_seed=3,
    )
    assert result["estimate_A"] == pytest.approx(0.2)
    assert result["estimate_B"] == pytest.approx(0.2)
    assert result["nominal_models_per_seed"] == 5


def test_categorical_vector_statistics_and_low_iqr_flags():
    a = frame().assign(category="x")
    b = frame().assign(category="y")
    summary, categories = paired_seed_categorical_vector_contrast(
        a, b, category_col="category", categories=["x", "y"],
        n_bootstrap=20, random_seed=4,
    )
    assert summary["total_variation_distance"] == pytest.approx(1)
    assert summary["jensen_shannon_divergence_bits"] == pytest.approx(1)
    assert categories.difference_B_minus_A.tolist() == pytest.approx([-1, 1])
    assert low_pooled_iqr_flags(0.08) == {
        "pooled_IQR_below_0.05A": False,
        "pooled_IQR_below_0.10A": True,
        "pooled_IQR_below_0.25A": True,
    }


def test_g402_glycine_primary_metric_cannot_silently_use_side_chain_distance():
    partners = (
        "PHE1519", "VAL1520", "ALA1521", "VAL1522", "ILE1523", "MET1524",
        "ASP1525", "ASN1526", "PHE1527", "ASP1528", "TYR1529", "LEU1530",
        "THR1531", "ARG1532", "ASP1533", "TRP1534", "SER1535",
    )
    valid = pd.DataFrame(columns=[f"CA_GLY402_CA-{partner}_CA" for partner in partners])
    columns = g402_common_ca_columns(valid, "WT")
    assert all(column.startswith("CA_GLY402_CA-") for column in columns)
    legacy_only = pd.DataFrame(columns=[f"shortest_GLY402-{partner}" for partner in partners])
    with pytest.raises(KeyError, match="atomically matched"):
        g402_common_ca_columns(legacy_only, "WT")


def test_paired_and_independent_bootstraps_differ_for_correlated_seeds():
    a, b = frame(), frame(offset=1)
    _, paired = joint_seed_bootstrap(
        make_seed_blocks(a, "value"), make_seed_blocks(b, "value"),
        n_bootstrap=80, random_seed=12, return_samples=True,
    )
    _, independent = seed_block_bootstrap(
        make_seed_blocks(a, "value"), make_seed_blocks(b, "value"),
        n_bootstrap=80, random_seed=12, return_samples=True,
    )
    assert np.std(paired[:, 1]) == pytest.approx(0)
    assert np.std(independent[:, 1]) > 0
