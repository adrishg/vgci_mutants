import numpy as np

from cav12_ensemble_rmsf.core import apply_fit, kabsch_fit, parse_af2_filename, ranges_to_mask


def test_kabsch_recovers_rigid_transform():
    fixed = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moving = (fixed - np.array([4., -3., 2.])) @ rotation.T
    fit = kabsch_fit(moving, fixed)
    assert fit.rmsd < 1e-10
    assert np.allclose(apply_fit(moving, fit), fixed, atol=1e-10)


def test_parse_af2_recycle_filename():
    parsed = parse_af2_filename(
        "cav12_wt_short_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_159.r3.pdb"
    )
    assert parsed["rank"] == 1
    assert parsed["model_number"] == 3
    assert parsed["seed"] == 159
    assert parsed["recycle_label"] == "r3"
    assert parsed["recycle_index"] == 3
    assert not parsed["is_final_model"]


def test_ranges_to_mask_is_inclusive():
    mask = ranges_to_mask(1, 10, [(2, 4), (8, 8)])
    assert np.flatnonzero(mask).tolist() == [1, 2, 3, 7]
