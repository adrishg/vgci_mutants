import numpy as np
from nav15_ensemble_rmsf.core import kabsch_fit, apply_fit, parse_af2_filename, ranges_to_mask

def test_kabsch_recovers_rigid_transform():
    moving = np.array([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    rotation = np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    fixed = moving @ rotation + np.array([4.,-2.,3.])
    fit = kabsch_fit(moving, fixed)
    transformed = apply_fit(moving, fit)
    assert np.allclose(transformed, fixed, atol=1e-7)
    assert fit.rmsd < 1e-7

def test_parse_recycle_filename():
    parsed = parse_af2_filename('nav15_wt_unrelaxed_rank_001_alphafold2_ptm_model_3_seed_051.r5.pdb')
    assert parsed['recycle_index'] == 5
    assert parsed['recycle_label'] == 'r5'
    assert parsed['model_number'] == 3
    assert parsed['seed'] == 51
    assert parsed['rank'] == 1

def test_ranges_to_mask_is_inclusive():
    mask = ranges_to_mask(1, 10, [(3,5),(9,9)])
    assert np.flatnonzero(mask).tolist() == [2,3,4,8]
