#!/usr/bin/env python3
"""Audit and filter experimental-comparison RMSD tables with existing QC.

The unit of selection is an AlphaFold model basename.  Every reference row for
an accepted model is retained.  Source RMSD tables are never modified.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


CHANNEL_DIR = {"kv21": "kv21", "nav15": "nav15", "cav12": "cav12"}
EXPECTED_REFERENCES = {
    "kv21": ["8SD3", "8SDA"],
    "nav15": ["8VYJ", "8VYK", "7DTC", "6UZ3", "7FBS", "8T6L"],
    "cav12": ["8HLP", "8WE6", "8FD7"],
}


def normalized_basename(value: object) -> str:
    """Normalize only path presentation; preserve ranks, seeds and .rN."""
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\\", "/")
    text = re.sub(r"/+", "/", text)
    return text.rsplit("/", 1)[-1]


def model_key(frame: pd.DataFrame) -> pd.Series:
    for column in ("pdb_file", "model_path"):
        if column in frame:
            keys = frame[column].map(normalized_basename)
            if keys.ne("").any():
                return keys
    raise KeyError("RMSD table needs pdb_file or model_path")


def ok3_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    """Reproduce shared.dataset_selection.select_manifest_rows(..., all_ok_3)."""
    required = {
        "seed", "model_number", "recycle_number", "is_base",
        "rmsd_to_previous_available", "aligned_coverage_to_previous",
    }
    missing = required - set(manifest)
    if missing:
        raise KeyError(f"QC manifest lacks all_ok_3 fields: {sorted(missing)}")
    selected = []
    work = manifest.loc[~manifest["is_base"].fillna(False).astype(bool)].copy()
    for _, trajectory in work.groupby(["seed", "model_number"], sort=False):
        trajectory = trajectory.sort_values("recycle_number")
        passed = (
            pd.to_numeric(trajectory["rmsd_to_previous_available"], errors="coerce").le(3.0)
            & pd.to_numeric(
                trajectory["aligned_coverage_to_previous"], errors="coerce"
            ).ge(0.9)
        )
        starts = [i for i in range(len(trajectory)) if bool(passed.iloc[i:].all())]
        if starts:
            selected.append(trajectory.iloc[starts[0]:])
    return pd.concat(selected, ignore_index=True) if selected else work.iloc[0:0]


def classify_column(name: str) -> dict[str, str]:
    low = name.lower()
    metric = (
        "rmsd" if "rmsd" in low else
        "coverage" if "coverage" in low else
        "atom_count" if "atom" in low and any(x in low for x in ("matched", "requested")) else
        "residue_count" if "residue" in low and any(x in low for x in ("matched", "requested")) else
        "distance_or_geometry" if any(x in low for x in ("distance", "diagonal", "aspect")) else
        "status" if any(x in low for x in ("status", "reason", "error", "valid")) else
        "metadata"
    )
    alignment = (
        "core_aligned" if "core_aligned" in low else
        "local_aligned" if "local_aligned" in low else
        "mapping_core" if "mapping" in low and "core" in low else ""
    )
    atom = "C-alpha" if "__ca__" in low or "_ca_" in low else ("backbone" if "__bb__" in low else "")
    chain = next((f"chain_{x}" for x in "abcd" if f"chain_{x}" in low), "")
    region_rules = [
        ("hydrophobic_nexus", "hydrophobic_nexus"), ("distal_s6", "distal_s6"),
        ("s6", "s6"), ("pore", "pore"), ("transmembrane", "transmembrane"),
        ("l403", "l403_region"), ("f412", "f412_region"), ("mutation", "mutation_region"),
        ("ifm", "ifm_or_pocket"), ("linker", "linker"), ("ctd", "ctd"),
        ("vsd", "vsd"), ("core", "stable_or_alignment_core"), ("whole", "whole_structure"),
    ]
    region = next((label for token, label in region_rules if token in low), "")
    return {
        "column": name, "metric_family": metric, "region_family": region,
        "alignment_frame": alignment, "atom_selection": atom, "chain_or_domain": chain,
    }


def audit_table(frame: pd.DataFrame, channel: str, audit_dir: Path) -> None:
    audit_dir.mkdir(parents=True, exist_ok=True)
    keys = model_key(frame)
    ref = frame["reference_id"].astype("string") if "reference_id" in frame else pd.Series("", index=frame.index)
    dataset = frame["dataset"].astype("string") if "dataset" in frame else pd.Series("", index=frame.index)
    duplicate = pd.DataFrame({"dataset": dataset, "model_key": keys, "reference_id": ref})
    duplicate = duplicate[duplicate.duplicated(keep=False)].sort_values(list(duplicate.columns))
    duplicate.to_csv(audit_dir / "duplicate_model_reference_rows.csv", index=False)

    summary = [
        ("total_rows", len(frame)),
        ("unique_model_basenames", keys.nunique()),
        ("missing_pdb_file", int(frame["pdb_file"].isna().sum()) if "pdb_file" in frame else len(frame)),
        ("missing_reference_id", int(frame["reference_id"].isna().sum()) if "reference_id" in frame else len(frame)),
        ("duplicate_model_reference_rows", len(duplicate)),
    ]
    pd.DataFrame(summary, columns=["metric", "value"]).to_csv(audit_dir / "qc_summary.csv", index=False)

    grouping_columns = [
        c for c in ("dataset", "sequence_condition", "protocol", "reference_id") if c in frame
    ]
    for column in grouping_columns:
        frame.groupby(column, dropna=False).size().rename("rows").reset_index().to_csv(
            audit_dir / f"rows_by_{column}.csv", index=False
        )
    combo = [c for c in ("sequence_condition", "protocol", "reference_id") if c in frame]
    if combo:
        frame.groupby(combo, dropna=False).size().rename("rows").reset_index().to_csv(
            audit_dir / "rows_by_condition_protocol_reference.csv", index=False
        )

    classifications = pd.DataFrame(classify_column(c) for c in frame.columns)
    classifications.to_csv(audit_dir / "column_classification.csv", index=False)
    rmsd_cols = classifications.loc[classifications.metric_family.eq("rmsd"), "column"].tolist()
    all_missing = [c for c in rmsd_cols if frame[c].isna().all()]
    pd.DataFrame({"column": all_missing}).to_csv(audit_dir / "all_missing_rmsd_columns.csv", index=False)

    numeric = frame.select_dtypes(include=[np.number])
    nonfinite = [
        {"column": c, "nonfinite_count": int((~np.isfinite(numeric[c].dropna())).sum())}
        for c in numeric if (~np.isfinite(numeric[c].dropna())).any()
    ]
    pd.DataFrame(nonfinite, columns=["column", "nonfinite_count"]).to_csv(
        audit_dir / "nonfinite_columns.csv", index=False
    )
    coverage_cols = [c for c in frame if "coverage" in c.lower()]
    if coverage_cols:
        frame[coverage_cols].apply(pd.to_numeric, errors="coerce").describe(
            percentiles=[.05, .25, .5, .75, .95]
        ).T.to_csv(audit_dir / "coverage_distributions.csv")
    status_cols = [c for c in frame if "status" in c.lower()]
    status_rows = []
    for column in status_cols:
        for value, count in frame[column].value_counts(dropna=False).items():
            status_rows.append({"column": column, "value": value, "rows": count})
    pd.DataFrame(status_rows).to_csv(audit_dir / "measurement_status_counts.csv", index=False)
    actual = sorted(ref.dropna().unique())
    pd.DataFrame({
        "expected_reference": EXPECTED_REFERENCES[channel],
        "present": [x in actual for x in EXPECTED_REFERENCES[channel]],
    }).to_csv(audit_dir / "expected_references.csv", index=False)


def discover_manifests(repo: Path, channel: str) -> dict[str, Path]:
    root = repo / CHANNEL_DIR[channel] / "rmsd_convergence_filtering"
    result = {}
    prefix = f"{channel}_rmsd_convergence_"
    for path in sorted(root.glob("*/all_models_manifest.csv")):
        identity = path.parent.name.lower()
        identity = re.sub(r"^(kv21|nav15|cav12)_rmsd_convergence_", "", identity)
        if identity.endswith("_test") or identity.startswith("g490r_"):
            continue
        result[identity] = path
    return result


def structural_qc_allowlists(repo: Path, channel: str) -> dict[str, set[str]]:
    if channel != "kv21":
        return {}
    result = {}
    for path in (repo / "kv21" / "dataDistances").glob("*structural_interface_qc.csv"):
        low = path.name.lower()
        condition = next((x for x in ("wt", "l403a", "f412l") if f"_{x}_" in low), None)
        protocol = next((x for x in ("vanilla", "masked") if x in low), None)
        if condition and protocol:
            values = pd.read_csv(path, usecols=["pdb_file"])["pdb_file"].map(normalized_basename)
            result[f"{condition}_{protocol}"] = set(values)
    return result


def filter_rmsd(
    source: Path, channel: str, repo: Path, output: Path,
    diagnostics_output: Path, dry_run: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(source, low_memory=False)
    audit_table(frame, channel, source.parent / "qc")
    keys = model_key(frame)
    frame = frame.copy()
    frame["dataset"] = frame["dataset"].astype(str).str.lower()
    frame["_normalized_model_key"] = keys
    manifests = discover_manifests(repo, channel)
    identities = sorted(frame["dataset"].astype(str).str.lower().unique())
    unknown = sorted(set(identities) - set(manifests))
    if unknown:
        raise RuntimeError(f"No unique QC manifest for RMSD dataset(s): {unknown}")

    qc_rows, diagnostics = [], []
    for identity in identities:
        manifest_path = manifests[identity]
        manifest = pd.read_csv(manifest_path, low_memory=False)
        manifest_keys = model_key(manifest)
        if manifest_keys.duplicated().any():
            dupes = sorted(manifest_keys[manifest_keys.duplicated(keep=False)].unique())
            raise RuntimeError(f"Ambiguous duplicate manifest keys in {manifest_path}: {dupes[:5]}")
        ok3 = ok3_manifest(manifest)
        ok3_keys = set(model_key(ok3))
        manifest_set = set(manifest_keys)
        rmsd_set = set(frame.loc[frame.dataset.astype(str).str.lower().eq(identity), "_normalized_model_key"])
        all_ok_map = dict(zip(manifest_keys, manifest.get("all_ok", False)))
        earliest_map = dict(zip(manifest_keys, manifest.get("earliest_converged_selected", False)))
        first100_pairs = (
            manifest.loc[manifest.get("all_ok", False).fillna(False), ["seed", "model_number"]]
            .drop_duplicates().sort_values(["seed", "model_number"]).head(100)
        )
        first100_pair_set = set(map(tuple, first100_pairs.to_numpy()))
        for i, key in zip(manifest.index, manifest_keys):
            qc_rows.append({
                "dataset": identity, "_normalized_model_key": key,
                "all_ok": bool(all_ok_map.get(key, False)),
                "all_ok_3": key in ok3_keys,
                "earliest_converged_selected": bool(earliest_map.get(key, False)),
                "first100": (manifest.at[i, "seed"], manifest.at[i, "model_number"]) in first100_pair_set,
            })
        diagnostics.append({
            "dataset": identity, "rmsd_unique_models": len(rmsd_set),
            "qc_manifest_unique_models": len(manifest_set),
            "exactly_matched_models": len(rmsd_set & manifest_set),
            "rmsd_models_missing_from_qc": len(rmsd_set - manifest_set),
            "qc_models_missing_from_rmsd": len(manifest_set - rmsd_set),
            "rmsd_duplicate_model_reference_rows": int(
                frame.loc[frame.dataset.astype(str).str.lower().eq(identity)]
                .duplicated(["_normalized_model_key", "reference_id"]).sum()
            ),
            "manifest_duplicate_keys": int(manifest_keys.duplicated().sum()),
            "ok3_models_in_manifest": len(ok3_keys),
            "manifest_path": str(manifest_path.relative_to(repo)),
        })
    qc = pd.DataFrame(qc_rows)
    merged = frame.merge(qc, on=["dataset", "_normalized_model_key"], how="left", validate="many_to_one")
    unmatched = merged["all_ok_3"].isna()
    if unmatched.any():
        bad = sorted(merged.loc[unmatched, "_normalized_model_key"].unique())
        pd.DataFrame({"unmatched_model": bad}).to_csv(
            diagnostics_output.with_name(diagnostics_output.stem + "_unmatched.csv"), index=False
        )
        raise RuntimeError(f"{len(bad)} RMSD models are unmatched; see diagnostics")
    retained = merged.loc[merged["all_ok_3"].astype(bool)].copy()
    diag = pd.DataFrame(diagnostics)
    retained_counts = (
        retained.groupby(["dataset", "sequence_condition", "protocol", "reference_id"])
        .agg(rows=("pdb_file", "size"), unique_models=("_normalized_model_key", "nunique"))
        .reset_index()
    )
    diag.to_csv(diagnostics_output, index=False)
    retained_counts.to_csv(
        diagnostics_output.with_name(diagnostics_output.stem + "_retained_counts.csv"), index=False
    )
    if not dry_run:
        retained.drop(columns=["_normalized_model_key"]).to_csv(output, index=False)
        allowlists = structural_qc_allowlists(repo, channel)
        if allowlists:
            accepted = retained.apply(
                lambda row: row["_normalized_model_key"] in allowlists.get(str(row["dataset"]).lower(), set()),
                axis=1,
            )
            qc_output = output.with_name(output.stem + "_QC.csv")
            retained.loc[accepted].drop(columns=["_normalized_model_key"]).to_csv(qc_output, index=False)
    return retained, diag


def filter_large_rmsd(
    source: Path, channel: str, repo: Path, output: Path,
    diagnostics_output: Path, dry_run: bool = False, chunksize: int = 2500,
) -> tuple[int, int, pd.DataFrame]:
    """Memory-bounded equivalent for multi-gigabyte wide RMSD tables."""
    header = pd.read_csv(source, nrows=0)
    meta_columns = [
        c for c in (
            "dataset", "sequence_condition", "protocol", "pdb_file", "model_path",
            "reference_id", "analysis_status", "measurement_status",
        ) if c in header
    ]
    meta = pd.read_csv(source, usecols=meta_columns, low_memory=False)
    meta["dataset"] = meta["dataset"].astype(str).str.lower()
    keys = model_key(meta)
    meta = meta.assign(_normalized_model_key=keys)
    audit_dir = source.parent / "dataRMSD" / "qc" if source.parent.name != "dataRMSD" else source.parent / "qc"
    audit_dir.mkdir(parents=True, exist_ok=True)
    classifications = pd.DataFrame(classify_column(c) for c in header.columns)
    classifications.to_csv(audit_dir / "column_classification.csv", index=False)
    duplicate = meta[["dataset", "_normalized_model_key", "reference_id"]]
    duplicate = duplicate[duplicate.duplicated(keep=False)]
    duplicate.to_csv(audit_dir / "duplicate_model_reference_rows.csv", index=False)
    pd.DataFrame([
        ("total_rows", len(meta)), ("unique_model_basenames", keys.nunique()),
        ("missing_pdb_file", int(meta["pdb_file"].isna().sum()) if "pdb_file" in meta else len(meta)),
        ("missing_reference_id", int(meta["reference_id"].isna().sum())),
        ("duplicate_model_reference_rows", len(duplicate)),
    ], columns=["metric", "value"]).to_csv(audit_dir / "qc_summary.csv", index=False)
    for column in [c for c in ("dataset", "sequence_condition", "protocol", "reference_id") if c in meta]:
        meta.groupby(column, dropna=False).size().rename("rows").reset_index().to_csv(
            audit_dir / f"rows_by_{column}.csv", index=False
        )
    meta.groupby(
        ["sequence_condition", "protocol", "reference_id"], dropna=False
    ).size().rename("rows").reset_index().to_csv(
        audit_dir / "rows_by_condition_protocol_reference.csv", index=False
    )
    status_rows = []
    for column in [c for c in meta if "status" in c.lower()]:
        for value, count in meta[column].value_counts(dropna=False).items():
            status_rows.append({"column": column, "value": value, "rows": count})
    pd.DataFrame(status_rows).to_csv(audit_dir / "measurement_status_counts.csv", index=False)
    actual = sorted(meta.reference_id.dropna().astype(str).unique())
    pd.DataFrame({
        "expected_reference": EXPECTED_REFERENCES[channel],
        "present": [x in actual for x in EXPECTED_REFERENCES[channel]],
    }).to_csv(audit_dir / "expected_references.csv", index=False)

    # Wide numeric audits use a bounded deterministic sample while missing/all-
    # missing checks are accumulated exactly over chunks.
    rmsd_cols = classifications.loc[classifications.metric_family.eq("rmsd"), "column"].tolist()
    coverage_cols = classifications.loc[classifications.metric_family.eq("coverage"), "column"].tolist()
    inspect_cols = rmsd_cols + coverage_cols
    nonmissing = {c: 0 for c in rmsd_cols}
    nonfinite_counts = {c: 0 for c in inspect_cols}
    samples = []
    for chunk in pd.read_csv(source, usecols=inspect_cols, chunksize=chunksize, low_memory=False):
        numeric = chunk.apply(pd.to_numeric, errors="coerce")
        for c in rmsd_cols:
            nonmissing[c] += int(numeric[c].notna().sum())
        for c in inspect_cols:
            nonfinite_counts[c] += int((~np.isfinite(numeric[c].dropna())).sum())
        samples.append(numeric[coverage_cols].head(min(100, len(numeric))))
    pd.DataFrame({"column": [c for c, n in nonmissing.items() if n == 0]}).to_csv(
        audit_dir / "all_missing_rmsd_columns.csv", index=False
    )
    pd.DataFrame([
        {"column": c, "nonfinite_count": n} for c, n in nonfinite_counts.items() if n
    ], columns=["column", "nonfinite_count"]).to_csv(audit_dir / "nonfinite_columns.csv", index=False)
    if coverage_cols:
        pd.concat(samples, ignore_index=True).describe(
            percentiles=[.05, .25, .5, .75, .95]
        ).T.to_csv(audit_dir / "coverage_distributions.csv")

    manifests = discover_manifests(repo, channel)
    identities = sorted(meta.dataset.astype(str).str.lower().unique())
    unknown = sorted(set(identities) - set(manifests))
    if unknown:
        raise RuntimeError(f"No unique QC manifest for RMSD dataset(s): {unknown}")
    qc_parts, diagnostics = [], []
    for identity in identities:
        manifest_path = manifests[identity]
        manifest = pd.read_csv(manifest_path, low_memory=False)
        manifest_keys = model_key(manifest)
        if manifest_keys.duplicated().any():
            raise RuntimeError(f"Ambiguous duplicate manifest keys: {manifest_path}")
        ok3_keys = set(model_key(ok3_manifest(manifest)))
        manifest_set = set(manifest_keys)
        rmsd_set = set(meta.loc[meta.dataset.astype(str).str.lower().eq(identity), "_normalized_model_key"])
        all_ok = manifest["all_ok"].fillna(False).astype(bool)
        first_pairs = set(map(tuple, manifest.loc[all_ok, ["seed", "model_number"]]
                              .drop_duplicates().sort_values(["seed", "model_number"]).head(100).to_numpy()))
        qc_parts.append(pd.DataFrame({
            "dataset": identity, "_normalized_model_key": manifest_keys,
            "all_ok": all_ok,
            "all_ok_3": manifest_keys.isin(ok3_keys),
            "earliest_converged_selected": manifest["earliest_converged_selected"].fillna(False).astype(bool),
            "first100": [
                (seed, model) in first_pairs
                for seed, model in zip(manifest.seed, manifest.model_number)
            ],
        }))
        diagnostics.append({
            "dataset": identity, "rmsd_unique_models": len(rmsd_set),
            "qc_manifest_unique_models": len(manifest_set),
            "exactly_matched_models": len(rmsd_set & manifest_set),
            "rmsd_models_missing_from_qc": len(rmsd_set - manifest_set),
            "qc_models_missing_from_rmsd": len(manifest_set - rmsd_set),
            "rmsd_duplicate_model_reference_rows": int(
                meta.loc[meta.dataset.astype(str).str.lower().eq(identity)]
                .duplicated(["_normalized_model_key", "reference_id"]).sum()
            ),
            "manifest_duplicate_keys": int(manifest_keys.duplicated().sum()),
            "ok3_models_in_manifest": len(ok3_keys),
            "manifest_path": str(manifest_path.relative_to(repo)),
        })
    qc = pd.concat(qc_parts, ignore_index=True)
    selected_meta = meta.merge(qc, on=["dataset", "_normalized_model_key"], how="left", validate="many_to_one")
    if selected_meta.all_ok_3.isna().any():
        bad = sorted(selected_meta.loc[selected_meta.all_ok_3.isna(), "_normalized_model_key"].unique())
        pd.DataFrame({"unmatched_model": bad}).to_csv(
            diagnostics_output.with_name(diagnostics_output.stem + "_unmatched.csv"), index=False
        )
        raise RuntimeError(f"{len(bad)} RMSD models are unmatched")
    accepted = set(zip(
        selected_meta.loc[selected_meta.all_ok_3.astype(bool), "dataset"].astype(str),
        selected_meta.loc[selected_meta.all_ok_3.astype(bool), "_normalized_model_key"],
    ))
    qc_lookup = qc.set_index(["dataset", "_normalized_model_key"])[
        ["all_ok", "all_ok_3", "earliest_converged_selected", "first100"]
    ].to_dict("index")
    diag = pd.DataFrame(diagnostics)
    diag.to_csv(diagnostics_output, index=False)
    retained_meta = selected_meta.loc[selected_meta.all_ok_3.astype(bool)]
    retained_meta.groupby(
        ["dataset", "sequence_condition", "protocol", "reference_id"]
    ).agg(rows=("pdb_file", "size"), unique_models=("_normalized_model_key", "nunique")).reset_index().to_csv(
        diagnostics_output.with_name(diagnostics_output.stem + "_retained_counts.csv"), index=False
    )
    if dry_run:
        return retained_meta._normalized_model_key.nunique(), len(retained_meta), diag

    allowlists = structural_qc_allowlists(repo, channel)
    qc_output = output.with_name(output.stem + "_QC.csv") if allowlists else None
    for path in [output, qc_output]:
        if path and path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    first = True
    for chunk in pd.read_csv(source, chunksize=chunksize, low_memory=False):
        chunk_keys = model_key(chunk)
        identities_chunk = chunk.dataset.astype(str).str.lower()
        keep = [(d, k) in accepted for d, k in zip(identities_chunk, chunk_keys)]
        part = chunk.loc[keep].copy()
        part_keys = chunk_keys.loc[keep]
        part_ids = identities_chunk.loc[keep]
        flags = [qc_lookup[(d, k)] for d, k in zip(part_ids, part_keys)]
        for flag in ("all_ok", "all_ok_3", "earliest_converged_selected", "first100"):
            part[flag] = [x[flag] for x in flags]
        part.to_csv(output, mode="w" if first else "a", header=first, index=False)
        if qc_output:
            strict = [
                k in allowlists.get(d, set()) for d, k in zip(part_ids, part_keys)
            ]
            part.loc[strict].to_csv(qc_output, mode="w" if first else "a", header=first, index=False)
        first = False
    return retained_meta._normalized_model_key.nunique(), len(retained_meta), diag


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--channel", required=True, choices=sorted(CHANNEL_DIR))
    parser.add_argument("--rmsd-csv", required=True, type=Path)
    parser.add_argument("--qc-manifest", type=Path, help="Reserved for a future single-dataset override")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--diagnostics-output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.rmsd_csv.resolve()
    if not source.is_file():
        parser.error(f"RMSD CSV not found: {source}")
    if args.output and args.output.resolve() == source:
        parser.error("--output must not overwrite the source CSV")
    output = (args.output or source.with_name(source.stem + "_OK3.csv")).resolve()
    diagnostics = (
        args.diagnostics_output
        or source.parent / "qc" / f"{source.stem}_join_diagnostics.csv"
    ).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_size > 1_000_000_000:
        unique_retained, rows_retained, diag = filter_large_rmsd(
            source, args.channel, args.repo_root.resolve(), output, diagnostics, args.dry_run
        )
        retained = None
    else:
        retained, diag = filter_rmsd(
            source, args.channel, args.repo_root.resolve(), output, diagnostics, args.dry_run
        )
        unique_retained, rows_retained = retained["_normalized_model_key"].nunique(), len(retained)
    print(diag.to_string(index=False))
    print(f"Retained {unique_retained} models / {rows_retained} rows")
    print("Dry run: no filtered RMSD written" if args.dry_run else f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
