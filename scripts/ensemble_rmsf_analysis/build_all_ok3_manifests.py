#!/usr/bin/env python3
"""Build RMSF subset manifests using the established All-OK-3 rule."""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


def select_all_ok3(frame: pd.DataFrame) -> pd.Series:
    required = {
        "seed", "model_number", "recycle_number", "is_base",
        "rmsd_to_previous_available", "aligned_coverage_to_previous",
    }
    missing = required - set(frame)
    if missing:
        raise KeyError(f"QC manifest lacks All-OK-3 fields: {sorted(missing)}")
    selected = pd.Series(False, index=frame.index)
    eligible = frame.loc[~frame.is_base.fillna(False).astype(bool)]
    for _, trajectory in eligible.groupby(["seed", "model_number"], sort=False):
        trajectory = trajectory.sort_values("recycle_number")
        passed = (
            pd.to_numeric(trajectory.rmsd_to_previous_available, errors="coerce").le(3.0)
            & pd.to_numeric(
                trajectory.aligned_coverage_to_previous, errors="coerce"
            ).ge(0.90)
        )
        starts = [index for index in range(len(trajectory)) if bool(passed.iloc[index:].all())]
        if starts:
            selected.loc[trajectory.iloc[starts[0]:].index] = True
    return selected


def dataset_from_directory(channel: str, directory: str) -> str:
    identity = directory.lower()
    identity = re.sub(
        rf"^{channel.lower()}_rmsd_convergence_", "", identity, flags=re.IGNORECASE
    )
    return identity


def build(repo_root: Path) -> pd.DataFrame:
    summaries = []
    for channel in ("kv21", "nav15", "cav12"):
        source_root = repo_root / channel / "rmsd_convergence_filtering"
        output_root = repo_root / channel / "dataRMSF" / "qc"
        output_root.mkdir(parents=True, exist_ok=True)
        frames = []
        for path in sorted(source_root.glob("*/all_models_manifest.csv")):
            dataset = dataset_from_directory(channel, path.parent.name)
            if dataset.endswith("_test"):
                continue
            frame = pd.read_csv(path, low_memory=False)
            frame["dataset"] = dataset
            frame["all_ok_3"] = select_all_ok3(frame)
            frame["qc_source_manifest"] = str(path.relative_to(repo_root))
            frames.append(frame)
            summaries.append({
                "channel": channel,
                "dataset": dataset,
                "total_manifest_models": len(frame),
                "all_ok_3_models": int(frame.all_ok_3.sum()),
                "excluded_models": int((~frame.all_ok_3).sum()),
                "source_manifest": str(path.relative_to(repo_root)),
            })
        if not frames:
            raise FileNotFoundError(f"No convergence manifests found for {channel}")
        combined = pd.concat(frames, ignore_index=True, sort=False)
        combined.to_csv(
            output_root / f"{channel}_all_ok3_selection_manifest.csv", index=False
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(
        repo_root / "scripts" / "ensemble_rmsf_analysis" / "all_ok3_selection_summary.csv",
        index=False,
    )
    return summary


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    print(build(root).to_string(index=False))
