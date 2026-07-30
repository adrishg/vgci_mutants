#!/usr/bin/env python3
"""Rebuild all-ok distance CSVs at custom RMSD thresholds without PDB files.

Uses the stored successive-RMSD and alignment-coverage values in each
all_models_manifest.csv, then joins the selected basenames to existing distance
CSVs. Original distance values and source CSVs are never modified.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

from filter_all_distance_csvs_from_rmsd import (
    Identity,
    distance_identity,
    discover_distance_csvs,
    discover_manifests,
    normalized_basename,
    read_csv,
    write_rows,
)


def number(value: object, default: float = float("nan")) -> float:
    try:
        return float(str(value).strip())
    except ValueError:
        return default


def threshold_label(threshold: float) -> str:
    return f"{threshold:g}".replace(".", "p") + "A"


def recompute_selection(
    manifest_rows: list[dict[str, str]], threshold: float, minimum_coverage: float
) -> tuple[set[str], set[str], dict[tuple[str, str], int]]:
    """Return all-ok keys, earliest keys, and earliest recycle per trajectory.

    A recycle starts the converged suffix only when its own incoming transition
    passes the RMSD/coverage criteria and every later incoming transition also
    passes.  This deliberately excludes r0 (which has no incoming transition)
    and prevents a terminal recycle from passing vacuously when its transition
    from the preceding recycle failed.
    """
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in manifest_rows:
        if str(row.get("is_base", "")).lower() == "true":
            continue
        groups[(row.get("seed", ""), row.get("model_number", ""))].append(row)

    all_ok, earliest, starts = set(), set(), {}
    for trajectory, rows in groups.items():
        rows.sort(key=lambda row: number(row.get("recycle_number"), 999))
        by_recycle = {int(number(row.get("recycle_number"))): row for row in rows}
        if not by_recycle:
            continue
        start = None
        for candidate in sorted(by_recycle):
            candidate_and_later = [
                row for recycle, row in by_recycle.items() if recycle >= candidate
            ]
            if candidate > 0 and all(
                number(row.get("rmsd_to_previous_available")) <= threshold
                and number(row.get("aligned_coverage_to_previous"), 0) >= minimum_coverage
                for row in candidate_and_later
            ):
                start = candidate
                break
        if start is None:
            continue
        starts[trajectory] = start
        for recycle, row in by_recycle.items():
            key = normalized_basename(row.get("pdb_basename") or row.get("pdb_file", ""))
            if recycle == start:
                earliest.add(key)
            if recycle >= start:
                all_ok.add(key)
    return all_ok, earliest, starts


def compatible(a: Identity, b: Identity) -> bool:
    family_a = "masked" if a.protocol.startswith("mask") else a.protocol
    family_b = "masked" if b.protocol.startswith("mask") else b.protocol
    return a.channel == b.channel and a.condition == b.condition and family_a == family_b


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distances-root", type=Path, required=True)
    parser.add_argument("--rmsd-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threshold", type=float, action="append", required=True)
    parser.add_argument("--minimum-coverage", type=float, default=0.9)
    parser.add_argument(
        "--selectivity-filter-max-distance", type=float,
        help=("Optionally write a second structural-QC CSV after rejecting whole "
              "trajectories whose G377 interchain C-alpha ring exceeds this distance."),
    )
    parser.add_argument("--channel", default="Kv21")
    args = parser.parse_args()

    distances, _ = discover_distance_csvs(args.distances_root.resolve())
    manifests, _ = discover_manifests(args.rmsd_root.resolve())
    audit: list[dict[str, object]] = []

    for distance_path in distances:
        identity = distance_identity(distance_path)
        if identity is None or identity.channel != args.channel:
            continue
        fields, distance_rows = read_csv(distance_path)
        distance_keys = {normalized_basename(row.get("pdb_file", "")) for row in distance_rows}
        candidates = []
        for manifest_path, manifest_id in manifests.items():
            if not compatible(identity, manifest_id):
                continue
            _, manifest_rows = read_csv(manifest_path)
            manifest_keys = {
                normalized_basename(row.get("pdb_basename") or row.get("pdb_file", ""))
                for row in manifest_rows
            }
            overlap = len(distance_keys & manifest_keys)
            if overlap:
                candidates.append((overlap, manifest_path, manifest_rows))
        if not candidates:
            continue
        overlap, manifest_path, manifest_rows = max(candidates, key=lambda item: item[0])

        for threshold in args.threshold:
            all_ok, earliest, starts = recompute_selection(
                manifest_rows, threshold, args.minimum_coverage
            )
            selected_rows = [
                row for row in distance_rows
                if normalized_basename(row.get("pdb_file", "")) in all_ok
            ]
            label = threshold_label(threshold)
            base = distance_path.stem.removesuffix("_all")
            output = (
                args.output_root.resolve() / identity.channel / label
                / identity.condition / identity.protocol
                / f"{base}_all_ok_rmsd_{label}.csv"
            )
            write_rows(output, fields, selected_rows)
            structural_qc_output = ""
            structural_qc_rows = selected_rows
            rejected_trajectories: set[tuple[str, str]] = set()
            if args.selectivity_filter_max_distance is not None:
                # A low successive-recycle RMSD only says that AlphaFold stopped
                # changing the model.  During this analysis we found one stable
                # F412L trajectory whose selectivity-filter loops were flipped
                # outward: its G377 ring was ~29/41 Å but later RMSDs were <1 Å.
                # Keep this pore-integrity check separate and explicit so we do
                # not mistake a converged, broken tetramer for useful variability.
                g377_columns = [
                    field for field in fields
                    if field.startswith("CA_CA_") and field.count("GLY377") == 2
                ]
                key_to_trajectory = {
                    normalized_basename(row.get("pdb_basename") or row.get("pdb_file", "")):
                    (row.get("seed", ""), row.get("model_number", ""))
                    for row in manifest_rows
                }
                for row in selected_rows:
                    values = [number(row.get(column)) for column in g377_columns]
                    if values and max(values) > args.selectivity_filter_max_distance:
                        key = normalized_basename(row.get("pdb_file", ""))
                        rejected_trajectories.add(key_to_trajectory[key])
                structural_qc_rows = [
                    row for row in selected_rows
                    if key_to_trajectory[normalized_basename(row.get("pdb_file", ""))]
                    not in rejected_trajectories
                ]
                qc_path = output.with_name(output.stem + "_structural_qc.csv")
                write_rows(qc_path, fields, structural_qc_rows)
                structural_qc_output = str(qc_path)
            audit.append({
                "channel": identity.channel,
                "condition": identity.condition,
                "protocol": identity.protocol,
                "threshold_A": threshold,
                "minimum_coverage": args.minimum_coverage,
                "source_distance_csv": str(distance_path),
                "source_manifest": str(manifest_path),
                "basename_overlap": overlap,
                "source_rows": len(distance_rows),
                "selected_rows": len(selected_rows),
                "structural_qc_max_g377_A": args.selectivity_filter_max_distance,
                "structural_qc_rejected_trajectories": len(rejected_trajectories),
                "structural_qc_rows": len(structural_qc_rows),
                "structural_qc_output_csv": structural_qc_output,
                "selected_trajectories": len(starts),
                "r0_trajectories": sum(start == 0 for start in starts.values()),
                "output_csv": str(output),
            })
            print(f"{identity.condition}/{identity.protocol} {threshold:g} Å: {len(selected_rows)} rows -> {output}")

    audit_path = args.output_root.resolve() / args.channel / "custom_rmsd_threshold_audit.csv"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(audit[0]) if audit else []
    with audit_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if fields:
            writer.writeheader()
            writer.writerows(audit)
    print(f"Audit: {audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
