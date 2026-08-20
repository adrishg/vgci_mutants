import numpy as np
import pytest

from analysis.statistics_revision.scripts.run_nav15_regional_rmsd import (
    region_definitions,
    regional_rmsd,
    residues_for_region,
)


def test_nav15_regions_are_fixed_to_repository_native_definitions():
    regions = region_definitions()
    assert residues_for_region(regions["dii_s6"]).tolist() == list(range(719, 748))
    assert residues_for_region(regions["ifm_motif"]).tolist() == [1169, 1170, 1171]
    assert residues_for_region(regions["ifm_receptor_set"]).tolist() == [
        1004, 1157, 1160, 1336, 1343, 1344,
    ]
    pore = residues_for_region(regions["pore_s5_s6_helices"])
    assert len(pore) == 206
    assert len(np.unique(pore)) == len(pore)


def test_regional_rmsd_honors_model_and_reference_presence():
    reference = np.zeros((4, 3), dtype=float)
    models = np.asarray([
        [[0, 0, 0], [3, 0, 0], [9, 0, 0], [20, 0, 0]],
        [[0, 0, 0], [4, 0, 0], [8, 0, 0], [20, 0, 0]],
    ], dtype=float)
    model_present = np.asarray([
        [True, True, False, True],
        [True, True, True, True],
    ])
    reference_present = np.asarray([True, True, True, False])
    region = np.asarray([False, True, True, True])
    rmsd, count, coverage, reference_count = regional_rmsd(
        models, model_present, reference, reference_present, region
    )
    assert reference_count == 2
    assert count.tolist() == [1, 2]
    assert coverage.tolist() == pytest.approx([.5, 1.0])
    assert rmsd.tolist() == pytest.approx([3.0, np.sqrt((4**2 + 8**2) / 2)])
