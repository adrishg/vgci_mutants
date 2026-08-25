#!/usr/bin/env python3
"""Finalize immutable-input verification and publication-run provenance."""

from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import platform
import subprocess


ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> str:
    values = []
    for package in ("numpy", "pandas", "scipy", "matplotlib", "seaborn", "pytest", "nbclient"):
        try:
            values.append(f"{package}={version(package)}")
        except PackageNotFoundError:
            values.append(f"{package}=unavailable")
    return "; ".join(values)


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "analysis/statistics_revision/paired_seed_v2")
    args = parser.parse_args()
    out = args.analysis_dir if args.analysis_dir.is_absolute() else ROOT / args.analysis_dir
    audit = out / "audit"

    source_rows = []
    with (audit / "SOURCE_INPUT_HASHES.tsv").open(newline="") as handle:
        for record in csv.DictReader(handle, delimiter="\t"):
            path = ROOT / record["relative_path"]
            observed = sha256(path) if path.is_file() else ""
            source_rows.append({
                "relative_path": record["relative_path"],
                "expected_sha256": record["sha256"],
                "observed_sha256": observed,
                "unchanged": observed == record["sha256"],
            })
    write_tsv(
        audit / "SOURCE_HASH_VERIFICATION.tsv",
        ["relative_path", "expected_sha256", "observed_sha256", "unchanged"],
        source_rows,
    )
    if not all(row["unchanged"] for row in source_rows):
        raise RuntimeError("At least one frozen source input changed")

    git_start = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    packages = package_versions()
    environment_rows = [
        {"key": "date", "value": date.today().isoformat()},
        {"key": "git_starting_commit", "value": git_start},
        {"key": "python", "value": platform.python_version()},
        {"key": "platform", "value": platform.platform()},
        {"key": "packages", "value": packages},
    ]
    write_tsv(audit / "SOFTWARE_ENVIRONMENT.tsv", ["key", "value"], environment_rows)

    common = {
        "date": date.today().isoformat(),
        "git_starting_commit": git_start,
        "python": platform.python_version(),
        "package_versions": packages,
        "source_hash_manifest": "audit/SOURCE_INPUT_HASHES.tsv",
        "output_hash_manifest": "audit/OUTPUT_HASHES.tsv",
        "analysis_mode": "publication",
        "nominal_and_contributing_counts": "per comparison; DATASET_INVENTORY.tsv and SEED_REGISTRY.tsv",
        "model_seed_trajectory_counts": "per condition; master_cohort_flow_summary.csv",
        "retained_snapshot_counts": "per condition; master_cohort_flow_summary.csv",
        "missing_input_warnings": "UNRESOLVED_BLOCKERS.md",
        "within_trajectory_reduction": "prespecified median, fraction, earliest/latest retained value, or binary event",
        "within_seed_weighting": "equal available AF2 model weight within recorded nominal label",
        "seed_resampling": "joint whole-recorded-label bootstrap; actual RNG equality unavailable",
    }
    specifications = [
        ("focal_and_qc", "python analysis/statistics_revision/scripts/run_paired_seed_v2.py --mode publication --output-dir analysis/statistics_revision/paired_seed_v2", 20260824, 2000, 9999, "primary joint nominal-label marginal contrast; common-label sensitivities separate"),
        ("full_distance_panel", "python analysis/statistics_revision/scripts/run_paired_seed_full_panel.py --mode publication --output-dir analysis/statistics_revision/paired_seed_v2/full_panel", 20260824, 200, 0, "seed-balanced W1/signed shift/breadth discovery effects and whole-label rank recurrence"),
        ("breadth_and_rmsf", "MPLCONFIGDIR=/tmp/vgci_mpl python analysis/statistics_revision/scripts/run_paired_seed_breadth_rmsf.py --mode publication --output-dir analysis/statistics_revision/paired_seed_v2", 20260824, 2000, 0, "whole-label IQR/MAD/W1 breadth and whole-label positional RMSF dispersion"),
        ("reduced_depth", "python analysis/statistics_revision/scripts/run_paired_seed_reduced_depth.py --mode publication --output-dir analysis/statistics_revision/paired_seed_v2", 20269824, 2000, 0, "1,000 repeated draws of 20 common recorded labels; retrospective stability"),
        ("qc_adjusted_yield", "python analysis/statistics_revision/scripts/run_qc_adjusted_target_yields.py --mode publication --output-dir analysis/statistics_revision/paired_seed_v2", 20260824, 2000, 0, "nominal-denominator target yield reported separately from survivor geometry"),
    ]
    run_rows = []
    for analysis, command, random_seed, bootstrap, permutation, estimand in specifications:
        run_rows.append(common | {
            "analysis": analysis,
            "exact_command": command,
            "random_seed": random_seed,
            "bootstrap_replicates": bootstrap,
            "permutation_replicates": permutation,
            "statistical_estimand": estimand,
        })
    run_fields = [
        "analysis", "date", "git_starting_commit", "python", "package_versions", "random_seed",
        "bootstrap_replicates", "permutation_replicates", "source_hash_manifest", "output_hash_manifest",
        "analysis_mode", "exact_command", "nominal_and_contributing_counts", "model_seed_trajectory_counts",
        "retained_snapshot_counts", "missing_input_warnings", "statistical_estimand",
        "within_trajectory_reduction", "within_seed_weighting", "seed_resampling",
    ]
    write_tsv(audit / "PUBLICATION_RUN_REGISTRY.tsv", run_fields, run_rows)

    explicit = [
        ROOT / "docs/MASK_REGISTRY.tsv", ROOT / "docs/CONSTRUCT_REGISTRY.tsv",
        ROOT / "docs/OUTCOME_REGISTRY.tsv", ROOT / "docs/KNOWN_ISSUES.md",
        ROOT / "docs/A3M_UPLOAD_CHECKLIST.md", ROOT / "shared/paired_seed_statistics.py",
        ROOT / "tests/test_paired_seed_statistics.py",
    ]
    patterns = [
        "analysis/statistics_revision/scripts/*paired_seed*.py",
        "analysis/statistics_revision/scripts/run_qc_adjusted_target_yields.py",
        "paperFigures/*Figure_S7*", "paperFigures/*Figure_S9*", "paperFigures/*Figure_S10*",
        "paperFigures/build_figure_s7_notebook.py", "paperFigures/build_figure_s9_notebook.py",
        "paperFigures/build_figure_s10_notebook.py",
        "docs/figures/supplementary_figure_s7/*", "docs/figures/supplementary_figure_s9/*",
        "docs/figures/supplementary_figure_s10/*",
        "docs/tables/supplementary_figure_s7/*", "docs/tables/supplementary_figure_s9/*",
        "docs/tables/supplementary_figure_s10/*",
    ]
    files = {path for path in out.rglob("*") if path.is_file() and path.name != "OUTPUT_HASHES.tsv"}
    files.update(path for path in explicit if path.is_file())
    for pattern in patterns:
        files.update(path for path in ROOT.glob(pattern) if path.is_file())
    output_rows = [
        {"relative_path": path.relative_to(ROOT), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(files)
    ]
    write_tsv(audit / "OUTPUT_HASHES.tsv", ["relative_path", "size_bytes", "sha256"], output_rows)
    print(f"verified {len(source_rows)} frozen inputs; hashed {len(output_rows)} revision outputs")


if __name__ == "__main__":
    main()
