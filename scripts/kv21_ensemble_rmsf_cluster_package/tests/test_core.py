from __future__ import annotations

import numpy as np

from kv21_ensemble_rmsf.core import (
    apply_fit,
    cyclic_mappings,
    infer_ring_order,
    kabsch_fit,
    one_letter,
    ranges_to_mask,
)


def test_one_letter_uses_uppercase_biopython_keys() -> None:
    assert one_letter("ALA") == "A"
    assert one_letter("Gly") == "G"
    assert one_letter("LEU") == "L"
    assert one_letter("MSE") == "M"


def test_kabsch_recovers_rigid_transform() -> None:
    moving = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    translation = np.asarray([5.0, -2.0, 1.5])
    fixed = moving @ rotation + translation
    fit = kabsch_fit(moving, fixed)
    transformed = apply_fit(moving, fit)
    assert fit.rmsd < 1e-10
    assert np.allclose(transformed, fixed, atol=1e-10)


def test_cyclic_mappings_preserve_order() -> None:
    mappings = cyclic_mappings(["A", "B", "C", "D"], ["A", "B", "C", "D"])
    assert len(mappings) == 4
    assert mappings[0] == {"A": "A", "B": "B", "C": "C", "D": "D"}
    assert mappings[1] == {"A": "B", "B": "C", "C": "D", "D": "A"}
    assert mappings[3] == {"A": "D", "B": "A", "C": "B", "D": "C"}


def test_ranges_to_mask_is_inclusive() -> None:
    mask = ranges_to_mask(1, 10, [[3, 5], [9, 9]])
    assert mask.tolist() == [False, False, True, True, True, False, False, False, True, False]


def test_ring_order_uses_shortest_perimeter_cycle() -> None:
    # Chain labels are intentionally not in geometric order.
    chain_ids = ["A", "B", "C", "D"]
    points = {
        "A": np.asarray([0.0, 0.0, 0.0]),
        "B": np.asarray([1.0, 0.0, 0.0]),
        "C": np.asarray([0.0, 1.0, 0.0]),
        "D": np.asarray([1.0, 1.0, 0.0]),
    }
    coords = np.stack([np.repeat(points[chain][None, :], 3, axis=0) for chain in chain_ids])
    present = np.ones((4, 3), dtype=bool)
    selection = np.ones(3, dtype=bool)
    order = infer_ring_order(coords, present, chain_ids, selection)
    assert order in {("A", "B", "D", "C"), ("A", "C", "D", "B")}
