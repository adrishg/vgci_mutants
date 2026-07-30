"""Optional RMSF recomputation from aligned coordinates and model selections."""

from __future__ import annotations
import numpy as np


def rmsf_from_aligned_coordinates(
    coordinates: np.ndarray, present: np.ndarray, selection: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate descriptive ensemble RMSF for an explicit boolean model subset."""
    if selection.dtype != bool or selection.shape != (coordinates.shape[0],):
        raise ValueError("selection must be a boolean vector with one entry per model")
    chosen = coordinates[selection].astype(float, copy=True)
    valid = present[selection].astype(bool)
    chosen[~valid] = np.nan
    mean = np.nanmean(chosen, axis=0)
    # Plain sum preserves NaN for unresolved residues.  ``nansum`` would turn
    # an entirely missing coordinate triplet into zero and bias RMSF downward
    # when the subsequent mean includes that artificial zero.
    squared = np.sum((chosen - mean) ** 2, axis=-1)
    rmsf = np.sqrt(np.nanmean(squared, axis=0))
    coverage = valid.mean(axis=0)
    return rmsf, coverage
