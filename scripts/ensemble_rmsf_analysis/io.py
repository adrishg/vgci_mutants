"""Schema-aware loading of precomputed RMSF profiles."""

from __future__ import annotations

from pathlib import Path
import pandas as pd

PROFILE_FILES = {
    "kv21": (
        "kv21_all_ok_3_symmetry_averaged_profiles.csv",
        "kv21_all_models_symmetry_averaged_profiles.csv",
    ),
    "nav15": (
        "nav15_all_ok_3_per_residue_profiles.csv",
        "nav15_all_models_per_residue_profiles.csv",
    ),
    "cav12": (
        "cav12_all_ok_3_per_residue_profiles.csv",
        "cav12_all_models_per_residue_profiles.csv",
    ),
}


def discover_rmsf_inputs(repo_root: str | Path, channel: str) -> pd.DataFrame:
    root = Path(repo_root) / channel / "dataRMSF"
    rows = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".npy", ".npz", ".json"}:
            row = {
                "path": str(path.relative_to(repo_root)), "suffix": path.suffix.lower(),
                "size_bytes": path.stat().st_size, "rows": None, "columns": None,
            }
            if path.suffix.lower() == ".csv":
                head = pd.read_csv(path, nrows=0)
                row["columns"] = "|".join(head.columns)
                with path.open("rb") as handle:
                    row["rows"] = max(sum(1 for _ in handle) - 1, 0)
            rows.append(row)
    return pd.DataFrame(rows)


def detect_profile_schema(frame: pd.DataFrame, channel: str) -> dict[str, str]:
    residue = next((c for c in ("raw_residue_number", "residue_number", "position") if c in frame), None)
    if channel == "kv21":
        rmsf = next((c for c in ("symmetry_averaged_rmsf_A", "ensemble_rmsf_A") if c in frame), None)
        coverage = next((c for c in ("mean_chain_coverage_fraction", "coverage_fraction") if c in frame), None)
    else:
        rmsf = next((c for c in ("ensemble_rmsf_A", "symmetry_averaged_rmsf_A") if c in frame), None)
        coverage = next((c for c in ("coverage_fraction", "mean_chain_coverage_fraction") if c in frame), None)
    required = {
        "dataset": "dataset" if "dataset" in frame else None,
        "condition": "sequence_condition" if "sequence_condition" in frame else None,
        "protocol": "protocol" if "protocol" in frame else None,
        "residue": residue, "rmsf": rmsf, "coverage": coverage,
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"{channel} RMSF schema missing required fields: {missing}")
    return required  # type: ignore[return-value]


def load_primary_profile(repo_root: str | Path, channel: str) -> tuple[pd.DataFrame, dict[str, str], Path]:
    profile_root = Path(repo_root) / channel / "dataRMSF" / "profiles"
    candidates = [profile_root / name for name in PROFILE_FILES[channel]]
    path = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
    if not path.is_file():
        raise FileNotFoundError(
            "Primary RMSF profile not found; checked: " + ", ".join(map(str, candidates))
        )
    frame = pd.read_csv(path)
    schema = detect_profile_schema(frame, channel)
    return frame, schema, path
