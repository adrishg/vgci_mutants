import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "filter_rmsd_by_existing_qc.py"
SPEC = importlib.util.spec_from_file_location("filter_rmsd", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_basename_preserves_recycle_suffix():
    assert MODULE.normalized_basename(" //a//model.r10.pdb ") == "model.r10.pdb"
    assert MODULE.normalized_basename("/a/model.r1.pdb") != MODULE.normalized_basename("/a/model.r10.pdb")


def test_one_model_multiple_references_is_not_duplicate():
    frame = pd.DataFrame({
        "dataset": ["wt_vanilla", "wt_vanilla"],
        "pdb_file": ["m.r1.pdb", "m.r1.pdb"],
        "reference_id": ["8SD3", "8SDA"],
    })
    keys = MODULE.model_key(frame)
    assert not pd.DataFrame({
        "dataset": frame.dataset, "key": keys, "reference": frame.reference_id
    }).duplicated().any()


def test_duplicate_model_reference_is_detected():
    frame = pd.DataFrame({
        "dataset": ["wt_vanilla"] * 2, "pdb_file": ["m.r1.pdb"] * 2,
        "reference_id": ["8SD3"] * 2,
    })
    assert frame.assign(key=MODULE.model_key(frame)).duplicated(
        ["dataset", "key", "reference_id"]
    ).sum() == 1


def test_qc_filter_keeps_all_reference_rows():
    frame = pd.DataFrame({
        "pdb_file": ["keep.r1.pdb", "keep.r1.pdb", "drop.r0.pdb", "drop.r0.pdb"],
        "reference_id": ["A", "B", "A", "B"],
    })
    accepted = {"keep.r1.pdb"}
    retained = frame[MODULE.model_key(frame).isin(accepted)]
    assert retained.reference_id.tolist() == ["A", "B"]


def test_unmatched_filename_is_visible():
    rmsd = {"a.r1.pdb", "missing.r1.pdb"}
    manifest = {"a.r1.pdb"}
    assert rmsd - manifest == {"missing.r1.pdb"}
