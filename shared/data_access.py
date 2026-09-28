"""Resolution of data paths against the working tree, a local cache, or Hugging Face.

Large derived tables are published in the companion dataset rather than tracked
in Git. Small provenance files stay in the repository and are returned as they
are. ``resolve`` always hands back a real local path, so it serves ``np.load``
and ``open`` as readily as ``pandas.read_csv``.

Set ``VGIC_DATA_ROOT`` to a complete existing copy to suppress fetching, and
``VGIC_OFFLINE`` to fail rather than reach the network.
"""

from __future__ import annotations

import os
from pathlib import Path

HF_REPO = "adrishgz/vgic-mutant-structural-ensembles"
HF_REVISION = os.environ.get("VGIC_HF_REVISION", "main")

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.environ.get("VGIC_CACHE", Path.home() / ".cache" / "vgic-mutants"))

_EXTRA_ROOT = os.environ.get("VGIC_DATA_ROOT")
EXTRA_ROOTS = [Path(_EXTRA_ROOT)] if _EXTRA_ROOT else []

_CHANNELS = {"cav12": "Cav1.2", "kv21": "Kv2.1", "nav15": "Nav1.5"}

# The dataset already uses ``distances`` for the pre-QC tables, so that name is kept.
_BUCKETS = {
    "dataDistances": "distances",
    "dataRMSD": "rmsd",
    "dataRMSF": "rmsf",
    "dataExtra": "extra",
    "experimental": "experimental",
    "rmsd_threshold_sensitivity": "sensitivity",
    "rmsd_filtered_distances": "archive/rmsd_filtered_distances",
    "rmsd_convergence_filtering": "convergence",
}

_KEPT_TOP_LEVEL = ("docs", "notebooks", "scripts", "tests", "shared", "paperFigures")
_KEPT_SECOND = ("rmsd_convergence_filtering", "experimental")
_KEPT_PREFIXES = ("analysis/statistics_revision/",)


class DataUnavailable(RuntimeError):
    """A required file is neither present locally nor retrievable."""


def is_kept_in_repo(rel_path: str | Path) -> bool:
    """True when this path is version-controlled rather than published."""
    parts = Path(rel_path).parts
    if not parts:
        return False
    if parts[0] in _KEPT_TOP_LEVEL:
        return True
    if str(Path(rel_path)).startswith(_KEPT_PREFIXES):
        return True
    return len(parts) > 1 and parts[1] in _KEPT_SECOND


def hf_path(rel_path: str | Path) -> str:
    """Map a repository-relative path to its path inside the dataset."""
    parts = Path(rel_path).parts
    if len(parts) < 2:
        raise ValueError(f"Cannot map {rel_path!r} to a dataset path")

    channel = _CHANNELS.get(parts[0])
    if channel is None:
        return str(Path(*parts))

    bucket = _BUCKETS.get(parts[1])
    if bucket is None:
        raise ValueError(f"No dataset bucket defined for {parts[0]}/{parts[1]}")
    return str(Path("data") / channel / bucket / Path(*parts[2:]))


def _is_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            return handle.read(43).startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError:
        return False


def _usable(path: Path) -> bool:
    return path.is_file() and not _is_lfs_pointer(path)


def resolve(rel_path: str | Path) -> Path:
    """Return a readable local path, fetching it once if it is not already here."""
    rel = Path(rel_path)
    if rel.is_absolute():
        rel = rel.relative_to(REPO_ROOT)

    for root in (REPO_ROOT, *EXTRA_ROOTS):
        if _usable(root / rel):
            return root / rel

    cached = CACHE_ROOT / rel
    if _usable(cached):
        return cached

    if os.environ.get("VGIC_OFFLINE"):
        raise DataUnavailable(
            f"{rel} is not available locally and VGIC_OFFLINE is set. "
            f"Clear VGIC_OFFLINE, or point VGIC_DATA_ROOT at a full copy."
        )
    return _fetch(rel, cached)


def _fetch(rel: Path, destination: Path) -> Path:
    from huggingface_hub import hf_hub_download

    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            revision=HF_REVISION,
            filename=hf_path(rel),
        )
    except Exception as error:  # noqa: BLE001 - re-raised with the path for context
        raise DataUnavailable(
            f"{rel} is not in the working tree and could not be fetched from "
            f"{HF_REPO} as {hf_path(rel)}: {error}"
        ) from error

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        try:
            destination.symlink_to(Path(downloaded).resolve())
        except OSError:
            import shutil

            shutil.copy2(downloaded, destination)
    return destination


def read_csv(rel_path: str | Path, **kwargs):
    import pandas as pd

    return pd.read_csv(resolve(rel_path), **kwargs)


def load_npy(rel_path: str | Path, **kwargs):
    """NumPy cannot read an array over HTTP, so the file is materialised first."""
    import numpy as np

    return np.load(resolve(rel_path), **kwargs)
