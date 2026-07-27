#!/usr/bin/env python3
"""Filter existing distance tables using RMSD-convergence manifests.

Distances are never recalculated. Rows are joined to manifests by the exact
normalized PDB basename, including any .rN recycle suffix.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


RMSD_REPORT_NAMES = {
    "all_models_manifest.csv",
    "all_ok_models.csv",
    "earliest_converged_one_per_seed_model.csv",
    "preconvergence_models.csv",
    "trajectory_summary.csv",
    "threshold_sensitivity_summary.csv",
    "discovered_pdb_files.csv",
}
DERIVED_DISTANCE_SUFFIXES = (
    "_filtered.csv",
    "_all_ok_5.csv",
    "_all_ok_rmsd_3A.csv",
    "_all_ok_rmsd_3A_structural_qc.csv",
    "_all_ok_rmsd_3p5A.csv",
    "_all_ok_rmsd_4A.csv",
    "_earliest_converged.csv",
    "_first_100_generated.csv",
)
AUDIT_COLUMNS = [
    "distance_csv", "manifest_csv", "channel", "condition", "protocol",
    "match_status", "original_row_count", "distance_unique_pdb_count",
    "manifest_row_count", "manifest_unique_pdb_count",
    "all_ok_manifest_count", "earliest_converged_manifest_count",
    "first_100_generated_manifest_count", "all_ok_matched_row_count",
    "earliest_converged_matched_row_count", "first_100_generated_matched_row_count",
    "unmatched_distance_row_count", "all_ok_manifest_missing_from_distances",
    "earliest_manifest_missing_from_distances", "distance_duplicate_key_count",
    "manifest_duplicate_key_count", "all_ok_output",
    "earliest_converged_output", "first_100_generated_output", "notes",
]


@dataclass(frozen=True, order=True)
class Identity:
    channel: str
    condition: str
    protocol: str


def normalized_basename(value: object) -> str:
    return os.path.basename(str(value).strip())


def true_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def canonical_channel(text: str) -> str | None:
    compact = re.sub(r"[^a-z0-9]", "", text.lower())
    if "cav12" in compact or "cav1.2" in text.lower():
        return "Cav12"
    if "nav15" in compact or "nav1.5" in text.lower():
        return "Nav15"
    if "kv21" in compact or "kv2.1" in text.lower():
        return "Kv21"
    return None


def clean_condition(value: str) -> str:
    value = value.strip("_- ")
    return "WT" if value.lower() == "wt" else value.upper()


def split_condition_protocol(tail: str, fallback_condition: str = "") -> tuple[str, str] | None:
    """Split a path/filename tail without collapsing exact mask variants."""
    tail = re.sub(r"(?i)_distances(?:_all)?$", "", tail.strip("_-"))
    tokens = [token for token in tail.split("_") if token]
    protocol_index = next(
        (i for i, token in enumerate(tokens) if re.fullmatch(r"(?i)(?:vanilla|masked|mask\w*)(?:af2)?", token)),
        None,
    )
    if protocol_index is None:
        return None
    condition_tokens = tokens[:protocol_index]
    protocol_tokens = tokens[protocol_index:]
    protocol_tokens[0] = re.sub(r"(?i)af2$", "", protocol_tokens[0]).lower()
    condition = "_".join(condition_tokens) or fallback_condition
    if not condition:
        return None
    return clean_condition(condition), "_".join(protocol_tokens)


def distance_identity(path: Path) -> Identity | None:
    channel = canonical_channel(str(path))
    if not channel:
        return None
    stem = path.stem
    marker = re.search(r"(?i)(?:cav(?:1\.2|12)|nav(?:1\.5|15)|kv(?:2\.1|21))[_-](.+)$", stem)
    if not marker:
        return None
    split = split_condition_protocol(marker.group(1))
    return Identity(channel, *split) if split else None


def manifest_identity(path: Path, first_row: dict[str, str] | None) -> Identity | None:
    channel = canonical_channel(str(path))
    if not channel:
        return None
    parent = path.parent.name
    tail = re.sub(r"(?i)^.*?_rmsd_convergence_", "", parent)
    fallback = ""
    if first_row and first_row.get("condition", "").lower() not in {"", "unknown", "none", "nan"}:
        fallback = first_row["condition"]
    split = split_condition_protocol(tail, fallback)
    return Identity(channel, *split) if split else None


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def discover_distance_csvs(root: Path) -> tuple[list[Path], list[dict[str, str]]]:
    found, unclassified = [], []
    for path in sorted(root.rglob("*.csv")):
        if path.name in RMSD_REPORT_NAMES or path.name.endswith(DERIVED_DISTANCE_SUFFIXES):
            continue
        try:
            fields, _ = read_csv(path)
        except (OSError, UnicodeError, csv.Error) as error:
            unclassified.append({"file": str(path), "file_type": "csv", "reason": f"unreadable: {error}"})
            continue
        if "pdb_file" not in fields:
            continue
        identity = distance_identity(path)
        if identity is None:
            unclassified.append({"file": str(path), "file_type": "distance_csv", "reason": "identity not classifiable"})
        else:
            found.append(path)
    return found, unclassified


def discover_manifests(root: Path) -> tuple[dict[Path, Identity], list[dict[str, str]]]:
    found, unclassified = {}, []
    for path in sorted(root.rglob("all_models_manifest.csv")):
        fields, rows = read_csv(path)
        if not {"all_ok", "earliest_converged_selected"}.issubset(fields):
            unclassified.append({"file": str(path), "file_type": "manifest", "reason": "missing selection columns"})
            continue
        identity = manifest_identity(path, rows[0] if rows else None)
        if identity is None:
            unclassified.append({"file": str(path), "file_type": "manifest", "reason": "identity not classifiable"})
        else:
            found[path] = identity
    return found, unclassified


def duplicate_counts(keys: Iterable[str]) -> dict[str, int]:
    return {key: count for key, count in Counter(keys).items() if key and count > 1}


def output_names(path: Path) -> tuple[str, str, str]:
    stem = path.stem
    base = stem[:-4] if stem.endswith("_all") else stem
    return (
        f"{base}_all_ok_5.csv",
        f"{base}_earliest_converged.csv",
        f"{base}_first_100_generated.csv",
    )


def numeric_sort_value(value: object) -> tuple[int, str]:
    text = str(value).strip()
    try:
        return int(float(text)), text
    except ValueError:
        return sys.maxsize, text


def first_generated_trajectory_keys(
    manifest_rows: list[dict[str, str]], manifest_keys: list[str], limit: int = 100,
) -> set[str]:
    """Select earliest-converged PDBs for the first seed/model trajectories.

    Rank is intentionally excluded: seed then model_number is the available
    generation identity, while rank is an AlphaFold confidence ordering.
    """
    selected = [
        (row, key) for row, key in zip(manifest_rows, manifest_keys)
        if true_value(row.get("all_ok")) and true_value(row.get("earliest_converged_selected"))
    ]
    selected.sort(key=lambda item: (
        numeric_sort_value(item[0].get("seed")),
        numeric_sort_value(item[0].get("model_number")),
        item[1],
    ))
    return {key for _, key in selected[:limit]}


def write_rows(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_audit_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def trajectory_violations(rows: list[dict[str, str]]) -> dict[tuple[str, str, str, str], set[str]]:
    groups: dict[tuple[str, str, str, str], set[str]] = defaultdict(set)
    for row in rows:
        if not true_value(row.get("earliest_converged_selected")):
            continue
        key = (row.get("condition", ""), row.get("protocol", ""), row.get("model_number", ""), row.get("seed", ""))
        source = row.get("pdb_basename") or normalized_basename(row.get("pdb_file", ""))
        groups[key].add(source)
    return {key: values for key, values in groups.items() if len(values) > 1}


def process_pair(
    distance_path: Path,
    manifest_path: Path,
    identity: Identity,
    output_root: Path,
    dry_run: bool,
    detail: dict[str, list[dict[str, object]]],
) -> dict[str, object]:
    distance_fields, distance_rows = read_csv(distance_path)
    manifest_fields, manifest_rows = read_csv(manifest_path)
    manifest_key_field = "pdb_basename" if "pdb_basename" in manifest_fields else "pdb_file"
    distance_keys = [normalized_basename(row.get("pdb_file", "")) for row in distance_rows]
    manifest_keys = [normalized_basename(row.get(manifest_key_field, "")) for row in manifest_rows]
    distance_dupes = duplicate_counts(distance_keys)
    manifest_dupes = duplicate_counts(manifest_keys)
    distance_set, manifest_set = set(distance_keys), set(manifest_keys)
    all_ok_set = {key for key, row in zip(manifest_keys, manifest_rows) if true_value(row.get("all_ok"))}
    earliest_set = {key for key, row in zip(manifest_keys, manifest_rows) if true_value(row.get("earliest_converged_selected"))}
    first_100_set = first_generated_trajectory_keys(manifest_rows, manifest_keys)
    all_ok_rows = [row for key, row in zip(distance_keys, distance_rows) if key in all_ok_set]
    earliest_rows = [row for key, row in zip(distance_keys, distance_rows) if key in earliest_set]
    first_100_rows = [row for key, row in zip(distance_keys, distance_rows) if key in first_100_set]
    unmatched = [(index, key, row) for index, (key, row) in enumerate(zip(distance_keys, distance_rows), 2) if key not in manifest_set]
    violations = trajectory_violations(manifest_rows)
    notes = []
    status = "matched"
    if manifest_dupes:
        status = "blocked_manifest_duplicate_keys"
        notes.append("manifest normalized keys are not unique")
    if violations:
        status = "blocked_earliest_trajectory_violation"
        notes.append(f"{len(violations)} earliest-selected model/seed groups contain multiple PDBs")
    out_dir = output_root / identity.channel / identity.condition / identity.protocol
    all_name, earliest_name, first_100_name = output_names(distance_path)
    all_output, earliest_output = out_dir / all_name, out_dir / earliest_name
    first_100_output = out_dir / first_100_name
    if status == "matched" and not dry_run:
        write_rows(all_output, distance_fields, all_ok_rows)
        write_rows(earliest_output, distance_fields, earliest_rows)
        write_rows(first_100_output, distance_fields, first_100_rows)
        if any(len(rows) > len(distance_rows) for rows in (all_ok_rows, earliest_rows, first_100_rows)):
            raise AssertionError("filtered row count exceeds original row count")
        if any(normalized_basename(row["pdb_file"]) not in all_ok_set for row in all_ok_rows):
            raise AssertionError("all-ok output contains an unselected row")
        if any(normalized_basename(row["pdb_file"]) not in earliest_set for row in earliest_rows):
            raise AssertionError("earliest output contains an unselected row")
        if any(normalized_basename(row["pdb_file"]) not in first_100_set for row in first_100_rows):
            raise AssertionError("first-100 output contains an unselected row")
    source = {"distance_csv": str(distance_path), "manifest_csv": str(manifest_path)}
    detail["unmatched_distance_rows"].extend(
        {**source, "source_row_number": index, "normalized_pdb_key": key, "pdb_file": row.get("pdb_file", "")}
        for index, key, row in unmatched
    )
    detail["unmatched_manifest_rows"].extend(
        {**source, "selection": selection, "normalized_pdb_key": key}
        for selection, selected in (("all_ok", all_ok_set), ("earliest_converged", earliest_set))
        for key in sorted(selected - distance_set)
    )
    detail["duplicate_distance_keys"].extend(
        {**source, "normalized_pdb_key": key, "duplicate_row_count": count}
        for key, count in sorted(distance_dupes.items())
    )
    detail["duplicate_manifest_keys"].extend(
        {**source, "normalized_pdb_key": key, "duplicate_row_count": count}
        for key, count in sorted(manifest_dupes.items())
    )
    return {
        **source, "channel": identity.channel, "condition": identity.condition,
        "protocol": identity.protocol, "match_status": status,
        "original_row_count": len(distance_rows),
        "distance_unique_pdb_count": len(distance_set),
        "manifest_row_count": len(manifest_rows),
        "manifest_unique_pdb_count": len(manifest_set),
        "all_ok_manifest_count": len(all_ok_set),
        "earliest_converged_manifest_count": len(earliest_set),
        "first_100_generated_manifest_count": len(first_100_set),
        "all_ok_matched_row_count": len(all_ok_rows),
        "earliest_converged_matched_row_count": len(earliest_rows),
        "first_100_generated_matched_row_count": len(first_100_rows),
        "unmatched_distance_row_count": len(unmatched),
        "all_ok_manifest_missing_from_distances": len(all_ok_set - distance_set),
        "earliest_manifest_missing_from_distances": len(earliest_set - distance_set),
        "distance_duplicate_key_count": len(distance_dupes),
        "manifest_duplicate_key_count": len(manifest_dupes),
        "all_ok_output": str(all_output) if status == "matched" else "",
        "earliest_converged_output": str(earliest_output) if status == "matched" else "",
        "first_100_generated_output": str(first_100_output) if status == "matched" else "",
        "notes": "; ".join(notes),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances-root", type=Path, required=True)
    parser.add_argument("--rmsd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    distances, unclassified_distance = discover_distance_csvs(args.distances_root.resolve())
    manifests, unclassified_manifest = discover_manifests(args.rmsd_root.resolve())
    by_identity: dict[Identity, list[Path]] = defaultdict(list)
    for path, identity in manifests.items():
        by_identity[identity].append(path)
    manifest_key_cache: dict[Path, set[str]] = {}
    for path in manifests:
        fields, rows = read_csv(path)
        key_field = "pdb_basename" if "pdb_basename" in fields else "pdb_file"
        manifest_key_cache[path] = {normalized_basename(row.get(key_field, "")) for row in rows}
    detail = defaultdict(list)
    audit = []
    ambiguous = 0
    missing = 0
    for distance in distances:
        identity = distance_identity(distance)
        assert identity is not None
        _, preview_rows = read_csv(distance)
        preview_keys = {normalized_basename(row.get("pdb_file", "")) for row in preview_rows}
        protocol_family = "masked" if identity.protocol.startswith("mask") else identity.protocol
        identity_compatible = [
            path for path, candidate_identity in manifests.items()
            if candidate_identity.channel == identity.channel
            and candidate_identity.condition == identity.condition
            and ("test" not in candidate_identity.protocol.lower() or "test" in identity.protocol.lower())
            and ("masked" if candidate_identity.protocol.startswith("mask") else candidate_identity.protocol) == protocol_family
        ]
        overlap_counts = {
            path: len(preview_keys & manifest_key_cache[path])
            for path in identity_compatible
        }
        overlap_counts = {path: count for path, count in overlap_counts.items() if count}
        # Related masked protocols can share a small number of generic basenames.
        # Prefer a uniquely best exact-basename overlap (normally 100% for the
        # intended manifest); call the match ambiguous only when the best score
        # is tied.
        best_overlap = max(overlap_counts.values(), default=0)
        candidates = [
            path for path, count in overlap_counts.items()
            if count == best_overlap
        ]
        print(f"{distance} -> {identity.channel}/{identity.condition}/{identity.protocol}")
        if len(candidates) != 1:
            status = "ambiguous" if len(candidates) > 1 else "no_manifest"
            ambiguous += status == "ambiguous"
            missing += status == "no_manifest"
            descriptions = [f"{path} (overlap={overlap_counts[path]})" for path in candidates]
            print(f"  {status}: {', '.join(descriptions) or 'no candidate with any exact basename overlap'}")
            detail["ambiguous_file_matches"].append({
                "distance_csv": str(distance), "channel": identity.channel,
                "condition": identity.condition, "protocol": identity.protocol,
                "candidate_count": len(candidates),
                "candidate_manifests": " | ".join(descriptions), "status": status,
            })
            if status == "no_manifest":
                detail["unmatched_distance_rows"].extend(
                    {
                        "distance_csv": str(distance), "manifest_csv": "",
                        "source_row_number": index,
                        "normalized_pdb_key": normalized_basename(row.get("pdb_file", "")),
                        "pdb_file": row.get("pdb_file", ""),
                    }
                    for index, row in enumerate(preview_rows, 2)
                )
            audit.append({
                "distance_csv": str(distance), "manifest_csv": "",
                "channel": identity.channel, "condition": identity.condition,
                "protocol": identity.protocol, "match_status": status,
                "original_row_count": len(preview_rows),
                "distance_unique_pdb_count": len(preview_keys),
                "unmatched_distance_row_count": len(preview_rows) if status == "no_manifest" else 0,
                "distance_duplicate_key_count": len(duplicate_counts(
                    normalized_basename(row.get("pdb_file", "")) for row in preview_rows
                )),
                "notes": "no exact identity match" if not candidates else "multiple exact identity matches",
            })
            continue
        print(f"  manifest: {candidates[0]}")
        result = process_pair(distance, candidates[0], identity, args.output_root.resolve(), args.dry_run, detail)
        audit.append(result)
        print(
            "  rows: "
            f"original={result['original_row_count']}, "
            f"all_ok={result['all_ok_matched_row_count']}, "
            f"earliest={result['earliest_converged_matched_row_count']}, "
            f"first_100={result['first_100_generated_matched_row_count']}, "
            f"unmatched={result['unmatched_distance_row_count']}; "
            f"duplicates(distance/manifest)={result['distance_duplicate_key_count']}/"
            f"{result['manifest_duplicate_key_count']}; status={result['match_status']}"
        )
    matched = sum(row.get("match_status") == "matched" for row in audit)
    totals = {
        "original_distance_csvs_discovered": len(distances),
        "matched_successfully": matched,
        "ambiguous": ambiguous,
        "without_manifest": missing,
        "total_original_rows": sum(int(row.get("original_row_count") or 0) for row in audit),
        "total_all_ok_rows": sum(int(row.get("all_ok_matched_row_count") or 0) for row in audit),
        "total_earliest_converged_rows": sum(int(row.get("earliest_converged_matched_row_count") or 0) for row in audit),
        "total_first_100_generated_rows": sum(int(row.get("first_100_generated_matched_row_count") or 0) for row in audit),
        "total_unmatched_rows": sum(int(row.get("unmatched_distance_row_count") or 0) for row in audit),
    }
    print("\nSummary")
    for key, value in totals.items():
        print(f"  {key}: {value}")
    if args.dry_run:
        print("  audit: dry run; no files written")
        return 0
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    write_audit_csv(output_root / "global_join_audit.csv", AUDIT_COLUMNS, audit)
    with (output_root / "global_join_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": totals, "files": audit}, handle, indent=2)
    reports = {
        "unmatched_distance_rows": ["distance_csv", "manifest_csv", "source_row_number", "normalized_pdb_key", "pdb_file"],
        "unmatched_manifest_rows": ["distance_csv", "manifest_csv", "selection", "normalized_pdb_key"],
        "duplicate_distance_keys": ["distance_csv", "manifest_csv", "normalized_pdb_key", "duplicate_row_count"],
        "duplicate_manifest_keys": ["distance_csv", "manifest_csv", "normalized_pdb_key", "duplicate_row_count"],
        "ambiguous_file_matches": ["distance_csv", "channel", "condition", "protocol", "candidate_count", "candidate_manifests", "status"],
    }
    for name, fields in reports.items():
        write_audit_csv(output_root / f"{name}.csv", fields, detail[name])
    unclassified = unclassified_distance + unclassified_manifest
    write_audit_csv(output_root / "unclassified_files.csv", ["file", "file_type", "reason"], unclassified)
    print(f"  audit: {output_root / 'global_join_audit.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
