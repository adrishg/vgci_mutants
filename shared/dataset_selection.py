"""Explicit selection of original or RMSD-filtered distance CSV variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


VALID_SELECTIONS = (
    "all", "all_ok", "all_ok_3", "all_ok_3_structural_qc",
    "all_ok_3_structural_interface_qc", "all_ok_3p5",
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
    base = original.stem
    # Some distance exports use ``_all_all`` to distinguish the complete
    # recalculated table. Dataset variants should still receive one clean suffix.
    while base.endswith("_all"):
        base = base[:-4]
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
        "all_ok_3_structural_interface_qc": original.parent / f"{base}_all_ok_rmsd_3A_structural_interface_qc.csv",
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


def apply_kv21_interface_qc(
    frame: pd.DataFrame,
    *,
    threshold_A: float = 27.0,
    trajectory_level: bool = True,
) -> pd.DataFrame:
    """Remove Kv2.1 trajectories with a detached pore–VSD interface.

    The existing structural-QC CSVs verify the G377 model-numbered
    selectivity-filter ring. A converged model can still retain that ring while
    separating the S6/C-terminal end of the pore from the neighboring
    voltage-sensor region. The six K429/E425/K422-to-N181/V184 contacts in each
    chain provide a direct check for this failure.

    Grossly failed trajectories begin above approximately 32.9 Å. One
    additional L403A vanilla recycle reaches 27.35 Å even though every other
    recycle in that trajectory remains near 20–21 Å. The default 27 Å cutoff
    removes this isolated excursion while remaining above the largest
    coordinate-derived 8SD3/8SDA reference distance (25.51 Å).
    """
    residue_pairs = (
        ("LYS429", "ASN181"), ("LYS429", "VAL184"),
        ("GLU425", "ASN181"), ("GLU425", "VAL184"),
        ("LYS422", "ASN181"), ("LYS422", "VAL184"),
    )
    columns = [
        f"CA_CA_{chain}_{first}_CA-{chain}_{second}_CA"
        for chain in "ABCD"
        for first, second in residue_pairs
    ]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Kv2.1 interface QC requires 24 distance columns; missing {missing}")
    if "pdb_file" not in frame.columns:
        raise KeyError("Kv2.1 interface QC requires the 'pdb_file' column")

    interface_max = frame[columns].apply(pd.to_numeric, errors="coerce").max(axis=1)
    directly_failed = interface_max.gt(threshold_A)
    trajectory = frame["pdb_file"].astype(str).str.replace(
        r"\.r\d+\.pdb$", ".pdb", regex=True
    )
    failed_trajectories = set(trajectory[directly_failed])
    rejected = trajectory.isin(failed_trajectories) if trajectory_level else directly_failed

    result = frame.loc[~rejected].copy()
    result.attrs.update(frame.attrs)
    result.attrs["kv21_interface_qc_threshold_A"] = float(threshold_A)
    result.attrs["kv21_interface_qc_trajectory_level"] = bool(trajectory_level)
    result.attrs["kv21_interface_qc_rejected_rows"] = int(rejected.sum())
    result.attrs["kv21_interface_qc_rejected_trajectories"] = len(failed_trajectories)
    result.attrs["kv21_interface_qc_max_retained_A"] = float(
        result[columns].apply(pd.to_numeric, errors="coerce").max(axis=1).max()
    )

    label = frame.attrs.get("dataset_label", "Kv2.1 ensemble")
    print(
        f"{label}: pore–VSD interface QC retained {len(result)}/{len(frame)} rows; "
        f"rejected {int(rejected.sum())} rows from {len(failed_trajectories)} trajectories; "
        f"maximum retained interface distance={result.attrs['kv21_interface_qc_max_retained_A']:.2f} Å"
    )
    return result
