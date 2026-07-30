import numpy as np

from scripts.ensemble_rmsf_analysis.recompute_all_ok3_local import streamed_moments


def test_streamed_moments_matches_direct_rmsf():
    rng = np.random.default_rng(7)
    coordinates = rng.normal(size=(17, 9, 3)).astype(np.float32)
    present = rng.random((17, 9)) > 0.2
    selection = np.arange(17) % 3 != 0
    counts, mean, rmsf, _ = streamed_moments(
        coordinates, present, selection, chunk_size=4
    )
    chosen = coordinates[selection].astype(float)
    chosen[~present[selection]] = np.nan
    expected_mean = np.nanmean(chosen, axis=0)
    squared_displacement = np.sum((chosen - expected_mean) ** 2, axis=-1)
    expected_rmsf = np.sqrt(np.nanmean(squared_displacement, axis=0))
    np.testing.assert_array_equal(counts, present[selection].sum(axis=0))
    np.testing.assert_allclose(mean, expected_mean, rtol=0, atol=1e-12)
    np.testing.assert_allclose(rmsf, expected_rmsf, rtol=0, atol=1e-12)


def test_streamed_moments_respects_selection():
    coordinates = np.zeros((4, 2, 3), dtype=np.float32)
    coordinates[3] = 100
    present = np.ones((4, 2), dtype=bool)
    selection = np.array([True, True, True, False])
    counts, mean, rmsf, _ = streamed_moments(
        coordinates, present, selection, chunk_size=2
    )
    np.testing.assert_array_equal(counts, [3, 3])
    np.testing.assert_allclose(mean, 0)
    np.testing.assert_allclose(rmsf, 0)
