"""Explicit selection of original or RMSD-filtered distance CSV variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


VALID_SELECTIONS = (
    "all", "all_ok", "all_ok_3", "all_ok_3_structural_qc", "all_ok_3p5",
    "first_converged", "first_100_generated",
)


def distance_csv_options(
    repo_root: str | Path,
    original_path: str | Path,
    channel: str,
    condition: str,
    protocol: str,
) -> dict[str, Path]:
    """Return all three explicit paths without reading any CSV."""
    root = Path(repo_root)
    original = Path(original_path)
    stem = original.stem
    base = stem[:-4] if stem.endswith("_all") else stem
    filtered = root / "rmsd_filtered_distances" / channel / condition / protocol
    threshold_3p5 = root / "rmsd_threshold_sensitivity" / channel / "3p5A" / condition / protocol
    return {
        "all": original,
        "all_ok": filtered / f"{base}_all_ok_5.csv",
        # Keep the publication-facing 3 Å tables beside their source CSVs so
        # notebooks cloned from GitHub can switch datasets without recreating
        # the repository's intermediate threshold-sensitivity directory.
        "all_ok_3": original.parent / f"{base}_all_ok_rmsd_3A.csv",
        "all_ok_3_structural_qc": original.parent / f"{base}_all_ok_rmsd_3A_structural_qc.csv",
        "all_ok_3p5": threshold_3p5 / f"{base}_all_ok_rmsd_3p5A.csv",
        "first_converged": filtered / f"{base}_earliest_converged.csv",
        "first_100_generated": filtered / f"{base}_first_100_generated.csv",
    }


def load_selected_distance_csv(
    label: str,
    options: dict[str, Path],
    selection: str,
    *,
    fallback_to_all: bool = True,
) -> pd.DataFrame:
    """Load the requested variant, visibly falling back only when authorized."""
    if selection not in VALID_SELECTIONS:
        raise ValueError(f"selection must be one of {VALID_SELECTIONS}; got {selection!r}")
    selected = options[selection]
    actual = selection
    if not selected.is_file():
        if not fallback_to_all:
            raise FileNotFoundError(f"{label}: requested {selection!r} file does not exist: {selected}")
        selected = options["all"]
        actual = "all"
        print(f"WARNING: {label}: {selection!r} is unavailable; using original 'all': {selected}")
    else:
        print(f"{label}: requested={selection}; actual={actual}; path={selected}")
    frame = pd.read_csv(selected)
    frame.attrs["dataset_label"] = label
    frame.attrs["requested_selection"] = selection
    frame.attrs["actual_selection"] = actual
    frame.attrs["source_path"] = str(selected)
    return frame
