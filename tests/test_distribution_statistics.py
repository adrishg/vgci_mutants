from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared.dataset_selection import load_selected_distance_csv
import shared.distribution_statistics as ds


def frame(rows):
    return ds.parse_trajectory_metadata(pd.DataFrame(rows))


def test_trajectory_parsing_and_identity():
    parsed = frame({
        "pdb_file": [
            "x_seed_12_model_3.r4.pdb", "x_seed_12_model_3.r7.pdb",
            "x_seed_8_model_1.pdb",
        ],
        "CA_X1_CA-Y2_CA": [1.0, 2.0, 3.0],
    })
    assert parsed["seed"].tolist() == [12, 12, 8]
    assert parsed["model_number"].tolist() == [3, 3, 1]
    assert parsed["recycle_number"].tolist()[:2] == [4, 7]
    assert pd.isna(parsed.loc[2, "recycle_number"])
    assert parsed.loc[0, "trajectory_id"] == parsed.loc[1, "trajectory_id"]


def test_parse_fails_loudly():
    with pytest.raises(ValueError, match="Could not parse"):
        frame({"pdb_file": ["unknown.pdb"], "CA_X1_CA-Y2_CA": [1.0]})


def test_trajectory_weights_are_balanced():
    parsed = frame({
        "pdb_file": [
            "x_seed_1_model_1.r1.pdb", "x_seed_1_model_1.r2.pdb",
            "x_seed_1_model_1.r3.pdb", "x_seed_2_model_1.r1.pdb",
        ],
        "CA_X1_CA-Y2_CA": [1., 2., 3., 9.],
    })
    weights = ds.trajectory_weights(parsed, "CA_X1_CA-Y2_CA")
    assert weights.sum() == pytest.approx(1.0)
    totals = weights.groupby(parsed.loc[weights.index, "trajectory_id"]).sum()
    assert totals.tolist() == pytest.approx([.5, .5])


def test_w1_identity_and_symmetry():
    a = [np.array([0., 1.]), np.array([2.])]
    b = [np.array([4., 5.]), np.array([6.])]
    assert ds.trajectory_balanced_w1(a, a) == pytest.approx(0.0)
    assert ds.trajectory_balanced_w1(a, b) == pytest.approx(
        ds.trajectory_balanced_w1(b, a)
    )


def test_paired_permutation_swaps_whole_blocks():
    a = {(1, 1): np.array([11., 12.]), (2, 1): np.array([21., 22., 23.])}
    b = {(1, 1): np.array([111., 112.]), (2, 1): np.array([121., 122., 123.])}
    _, fast_null = ds.paired_block_permutation(
        a, b, n_permutations=8, random_seed=4, return_null=True
    )
    rng = np.random.default_rng(4)
    brute = []
    keys = sorted(a)
    swaps = rng.integers(0, 2, size=(8, len(keys)), dtype=np.int8).astype(bool)
    for swap in swaps:
        left = [b[k] if swap[i] else a[k] for i, k in enumerate(keys)]
        right = [a[k] if swap[i] else b[k] for i, k in enumerate(keys)]
        # Entire unequal-length arrays move sides; no recycle row can split off.
        assert sorted(map(len, [*left, *right])) == [2, 2, 3, 3]
        brute.append(ds.trajectory_balanced_w1(left, right))
    assert fast_null == pytest.approx(brute)


def test_permutation_is_deterministic():
    a = {(i, 1): np.array([float(i)]) for i in range(8)}
    b = {(i, 1): np.array([float(i + 3)]) for i in range(8)}
    first = ds.paired_block_permutation(a, b, n_permutations=99, random_seed=7)
    second = ds.paired_block_permutation(a, b, n_permutations=99, random_seed=7)
    assert first == second


def test_bootstrap_resamples_blocks_not_rows():
    a = {(1, 1): np.array([1., 2.]), (2, 1): np.array([8., 9., 10.])}
    b = {(1, 1): np.array([3., 4.]), (2, 1): np.array([18., 19., 20.])}
    _, samples = ds.paired_block_bootstrap(
        a, b, n_bootstrap=12, random_seed=9, return_samples=True
    )
    rng = np.random.default_rng(9)
    draws = rng.integers(0, 2, size=(12, 2))
    keys = sorted(a)
    for row, draw in zip(samples, draws):
        left = [a[keys[i]] for i in draw]
        right = [b[keys[i]] for i in draw]
        # Every sampled item is a complete original block.
        assert all(tuple(x) in {tuple(v) for v in a.values()} for x in left)
        assert all(tuple(x) in {tuple(v) for v in b.values()} for x in right)
        expected = ds.distribution_metrics(left, right)
        assert row == pytest.approx([
            expected["W1_A"], expected["delta_median_A"],
            expected["log2_IQR_ratio"],
        ])


def test_bh_q_values_are_valid():
    table = pd.DataFrame({
        "comparison_id": ["a", "a", "b"],
        "analysis_level": ["x", "x", "x"],
        "p_W1_permutation": [.01, .4, .03],
    })
    adjusted = ds.adjust_fdr(table)
    assert adjusted[["q_within_comparison", "q_global"]].ge(0).all(axis=None)
    assert adjusted[["q_within_comparison", "q_global"]].le(1).all(axis=None)


def test_mutation_matching_requires_exact_names():
    a = pd.DataFrame({"CA_LEU403_CA-X1_CA": [1.], "pdb_file": ["x"]})
    b = pd.DataFrame({"CA_ALA403_CA-X1_CA": [1.], "pdb_file": ["y"]})
    assert ds.exact_common_distance_columns(a, b) == []
    unmatched = ds.unmatched_distance_columns(a, b)
    assert unmatched.loc[0, "only_in_A"] == "CA_LEU403_CA-X1_CA"
    assert unmatched.loc[0, "only_in_B"] == "CA_ALA403_CA-X1_CA"


def test_missing_final_qc_never_falls_back(tmp_path):
    all_path = tmp_path / "all.csv"
    all_path.write_text("pdb_file,CA_X1_CA-Y2_CA\nx_seed_1_model_1.pdb,1\n")
    options = {"all": all_path, "all_ok_3": tmp_path / "missing.csv"}
    with pytest.raises(FileNotFoundError, match="requested 'all_ok_3'"):
        load_selected_distance_csv(
            "synthetic", options, "all_ok_3", fallback_to_all=False
        )


def test_shifted_broadened_synthetic_signal():
    rows_a, rows_b = [], []
    for seed in range(1, 13):
        for recycle in range(3):
            name = f"x_seed_{seed}_model_1.r{recycle}.pdb"
            rows_a.append({"pdb_file": name, "CA_X1_CA-Y2_CA": seed * .02 + recycle * .01})
            rows_b.append({"pdb_file": name, "CA_X1_CA-Y2_CA": 4 + seed * .15 + recycle * .08})
    result = ds.analyze_distance(
        frame(rows_a), frame(rows_b), "CA_X1_CA-Y2_CA",
        n_permutations=999, n_bootstrap=100, random_seed=33,
    )
    assert result["W1_A"] > 4
    assert result["log2_IQR_ratio"] > 1
    assert result["p_W1_permutation"] < .01
