from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .core import (
    apply_fit,
    canonicalize_transformed,
    chain_to_raw_index_map,
    choose_best_mapping,
    discover_structures,
    evaluate_cyclic_raw_fits,
    finite_rmsd,
    infer_ring_order,
    load_config,
    load_structure,
    mapping_text,
    parse_af2_filename,
    ranges_to_mask,
    read_fasta,
    structure_to_raw_ca,
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def config_paths(config_path: str | Path) -> tuple[dict[str, Any], Path, Path]:
    cfg = load_config(config_path)
    project_root = Path(cfg["project"]["project_root"]).expanduser().resolve()
    output_root = Path(cfg["project"]["output_root"]).expanduser().resolve()
    return cfg, project_root, output_root


def residue_axis(cfg: Mapping[str, Any]) -> tuple[int, int, np.ndarray]:
    start = int(cfg["residue_axis"]["start"])
    end = int(cfg["residue_axis"]["end"])
    if end < start:
        raise ValueError("residue_axis.end must be >= residue_axis.start")
    return start, end, np.arange(start, end + 1, dtype=np.int32)


def build_manifest(cfg: Mapping[str, Any]) -> pd.DataFrame:
    recursive = bool(cfg.get("analysis", {}).get("recursive", True))
    rows: list[dict[str, Any]] = []
    seen_realpaths: set[str] = set()
    for dataset in cfg.get("datasets", []):
        root = Path(dataset["path"]).expanduser().resolve()
        structures = discover_structures(root, recursive=recursive)
        for path in structures:
            realpath = str(path.resolve())
            if realpath in seen_realpaths:
                raise ValueError(f"Duplicate structure discovered in multiple datasets: {realpath}")
            seen_realpaths.add(realpath)
            row = {
                "dataset": str(dataset["name"]),
                "sequence_condition": str(dataset["sequence_condition"]),
                "protocol": str(dataset["protocol"]),
                "dataset_root": str(root),
                "model_path": realpath,
            }
            row.update(parse_af2_filename(path.name))
            rows.append(row)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        columns = [
            "manifest_index", "dataset", "sequence_condition", "protocol", "dataset_root", "model_path",
            "pdb_file", "filename_stem", "trajectory_id", "recycle_label", "recycle_index",
            "is_final_model", "rank", "model_number", "seed", "relaxation", "af2_protocol",
        ]
        return pd.DataFrame(columns=columns)
    manifest = manifest.sort_values(["dataset", "model_path"], kind="stable").reset_index(drop=True)
    manifest.insert(0, "manifest_index", np.arange(len(manifest), dtype=np.int64))
    return manifest


def write_manifest_and_counts(cfg: Mapping[str, Any], output_dir: str | Path) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg)
    manifest_path = output_dir / "kv21_model_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    counts = (
        manifest.groupby(["dataset", "sequence_condition", "protocol"], dropna=False)
        .agg(
            number_of_structures=("manifest_index", "size"),
            number_of_trajectories=("trajectory_id", "nunique"),
            number_of_final_models=("is_final_model", "sum"),
            number_of_seeds=("seed", "nunique"),
        )
        .reset_index()
        if not manifest.empty
        else pd.DataFrame()
    )
    counts_path = output_dir / "kv21_dataset_counts.csv"
    counts.to_csv(counts_path, index=False)
    return manifest_path, counts_path


def inspect_inputs(cfg: Mapping[str, Any], output_dir: str | Path, max_models_per_dataset: int = 2) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path, counts_path = write_manifest_and_counts(cfg, output_dir)
    manifest = pd.read_csv(manifest_path)
    raw_start, raw_end, _ = residue_axis(cfg)
    chain_ids = [str(x) for x in cfg["model_chains"]]
    aligner_cfg = cfg.get("sequence_alignment", {})
    fasta_by_condition = {condition: read_fasta(path) for condition, path in cfg["sequence_fastas"].items()}

    dataset_rows: list[dict[str, Any]] = []
    for dataset in cfg.get("datasets", []):
        path = Path(dataset["path"]).expanduser().resolve()
        dataset_rows.append(
            {
                "dataset": dataset["name"],
                "path": str(path),
                "exists": path.exists(),
                "is_directory": path.is_dir(),
                "structure_count": int((manifest["dataset"] == dataset["name"]).sum()) if not manifest.empty else 0,
            }
        )
    dataset_status_path = output_dir / "kv21_dataset_path_status.csv"
    pd.DataFrame(dataset_rows).to_csv(dataset_status_path, index=False)

    fasta_rows: list[dict[str, Any]] = []
    for condition, path in cfg["sequence_fastas"].items():
        sequence = fasta_by_condition[condition]
        fasta_rows.append(
            {
                "sequence_condition": condition,
                "fasta_path": str(Path(path).resolve()),
                "sequence_length": len(sequence),
                "raw_axis_length": raw_end - raw_start + 1,
                "length_matches_axis": len(sequence) == raw_end - raw_start + 1,
            }
        )
    fasta_status_path = output_dir / "kv21_fasta_status.csv"
    pd.DataFrame(fasta_rows).to_csv(fasta_status_path, index=False)

    inventory_rows: list[dict[str, Any]] = []
    if not manifest.empty:
        selected = manifest.groupby("dataset", group_keys=False).head(max_models_per_dataset)
        for row in selected.itertuples(index=False):
            result: dict[str, Any] = {
                "manifest_index": int(row.manifest_index),
                "dataset": row.dataset,
                "model_path": row.model_path,
                "status": "ok",
                "error": "",
            }
            try:
                structure = load_structure(row.model_path)
                sequence = fasta_by_condition[str(row.sequence_condition)]
                _, present, identities, report = structure_to_raw_ca(
                    structure,
                    chain_ids,
                    sequence,
                    raw_start,
                    raw_end,
                    aligner_cfg,
                )
                result.update(report)
                result["all_required_chains_present"] = all(chain in structure.chains for chain in chain_ids)
                result["total_mapped_ca"] = int(present.sum())
                result["nonempty_identity_positions"] = int((identities != "").sum())
            except Exception as exc:
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            inventory_rows.append(result)
    inventory_path = output_dir / "kv21_structure_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(inventory_path, index=False)

    reference_rows: list[dict[str, Any]] = []
    for reference in cfg["references"]["structures"]:
        row: dict[str, Any] = {
            "reference_id": reference["id"],
            "path": str(Path(reference["path"]).expanduser().resolve()),
            "status": "ok",
            "error": "",
        }
        try:
            structure = load_structure(reference["path"])
            sequence = fasta_by_condition[reference["sequence_condition"]]
            _, present, _, report = structure_to_raw_ca(
                structure,
                [str(x) for x in reference["chains"]],
                sequence,
                raw_start,
                raw_end,
                aligner_cfg,
            )
            row.update(report)
            row["total_mapped_ca"] = int(present.sum())
        except Exception as exc:
            row["status"] = "error"
            row["error"] = f"{type(exc).__name__}: {exc}"
        reference_rows.append(row)
    reference_status_path = output_dir / "kv21_reference_inventory.csv"
    pd.DataFrame(reference_rows).to_csv(reference_status_path, index=False)

    return {
        "manifest": manifest_path,
        "counts": counts_path,
        "dataset_status": dataset_status_path,
        "fasta_status": fasta_status_path,
        "structure_inventory": inventory_path,
        "reference_inventory": reference_status_path,
    }


def prepare_references(cfg: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_start, raw_end, raw_numbers = residue_axis(cfg)
    canonical_chains = [str(x) for x in cfg["canonical_chains"]]
    aligner_cfg = cfg.get("sequence_alignment", {})
    fasta_by_condition = {condition: read_fasta(path) for condition, path in cfg["sequence_fastas"].items()}
    references = {str(item["id"]): item for item in cfg["references"]["structures"]}
    anchor_id = str(cfg["references"]["anchor_id"])
    if anchor_id not in references:
        raise ValueError(f"Anchor reference {anchor_id!r} is not listed under references.structures")

    anchor_cfg = references[anchor_id]
    anchor_structure = load_structure(anchor_cfg["path"])
    anchor_coords, anchor_present, anchor_identities, anchor_report = structure_to_raw_ca(
        anchor_structure,
        [str(x) for x in anchor_cfg["chains"]],
        fasta_by_condition[str(anchor_cfg["sequence_condition"])],
        raw_start,
        raw_end,
        aligner_cfg,
    )
    # The anchor chain IDs define the canonical chain order.
    if [str(x) for x in anchor_cfg["chains"]] != canonical_chains:
        raise ValueError("The anchor reference chain order must match canonical_chains")

    core_mask = ranges_to_mask(raw_start, raw_end, cfg["alignment_core"]["raw_ranges"])
    anchor_ring_order = infer_ring_order(anchor_coords, anchor_present, canonical_chains, core_mask)
    minimum_matched_ca = int(cfg["alignment_core"]["minimum_matched_ca"])
    minimum_coverage = float(cfg["alignment_core"]["minimum_coverage"])

    reference_ids: list[str] = []
    reference_coords: list[np.ndarray] = []
    reference_present: list[np.ndarray] = []
    reference_identities: list[np.ndarray] = []
    reference_pdb_residue_numbers: list[np.ndarray] = []
    reference_source_chain_ids: list[np.ndarray] = []
    mapping_rows: list[dict[str, Any]] = []

    for reference_id, reference_cfg in references.items():
        structure = load_structure(reference_cfg["path"])
        moving_chains = [str(x) for x in reference_cfg["chains"]]
        moving_coords, moving_present, moving_identities, report = structure_to_raw_ca(
            structure,
            moving_chains,
            fasta_by_condition[str(reference_cfg["sequence_condition"])],
            raw_start,
            raw_end,
            aligner_cfg,
        )
        moving_pdb_numbers = np.full((len(moving_chains), len(raw_numbers)), -1, dtype=np.int32)
        for moving_chain_index, moving_chain_id in enumerate(moving_chains):
            chain = structure.chains.get(moving_chain_id)
            if chain is None:
                continue
            raw_map = chain_to_raw_index_map(
                chain,
                fasta_by_condition[str(reference_cfg["sequence_condition"])],
                aligner_cfg,
            )
            for chain_residue_index, raw_position in raw_map.items():
                if raw_start <= raw_position <= raw_end:
                    moving_pdb_numbers[moving_chain_index, raw_position - raw_start] = chain.residues[chain_residue_index].resseq
        if reference_id == anchor_id:
            transformed = moving_coords
            canonical_coords = moving_coords
            canonical_present = moving_present
            canonical_identities = moving_identities
            shift = 0
            orientation = "anchor"
            moving_ring_order = anchor_ring_order
            fixed_ring_order = anchor_ring_order
            mapping = {chain: chain for chain in canonical_chains}
            canonical_pdb_numbers = moving_pdb_numbers
            canonical_source_chains = np.asarray(canonical_chains, dtype="U4")
            core_rmsd = 0.0
            mapping_gap = math.nan
            matched_atoms = int((core_mask[None, :] & anchor_present).sum())
            coverage = 1.0
        else:
            candidates = evaluate_cyclic_raw_fits(
                moving_coords,
                moving_present,
                moving_chains,
                anchor_coords,
                anchor_present,
                canonical_chains,
                core_mask,
                minimum_matched_ca,
                minimum_coverage,
            )
            best, mapping_gap = choose_best_mapping(candidates)
            assert best.fit is not None
            transformed = apply_fit(moving_coords, best.fit)
            canonical_coords, canonical_present, canonical_identities = canonicalize_transformed(
                transformed,
                moving_present,
                moving_identities,
                moving_chains,
                canonical_chains,
                best.mapping,
            )
            canonical_pdb_numbers = np.full((len(canonical_chains), len(raw_numbers)), -1, dtype=np.int32)
            canonical_source_chains = np.full(len(canonical_chains), "", dtype="U4")
            moving_index_lookup = {chain: index for index, chain in enumerate(moving_chains)}
            canonical_index_lookup = {chain: index for index, chain in enumerate(canonical_chains)}
            for moving_chain_id, canonical_chain_id in best.mapping.items():
                mi = moving_index_lookup[moving_chain_id]
                ci = canonical_index_lookup[canonical_chain_id]
                canonical_pdb_numbers[ci] = moving_pdb_numbers[mi]
                canonical_source_chains[ci] = moving_chain_id
            shift = best.shift
            orientation = best.orientation
            moving_ring_order = best.moving_ring_order
            fixed_ring_order = best.fixed_ring_order
            mapping = best.mapping
            core_rmsd = best.fit.rmsd
            matched_atoms = best.matched_atoms
            coverage = best.coverage
        reference_ids.append(reference_id)
        reference_coords.append(canonical_coords.astype(np.float32))
        reference_present.append(canonical_present)
        reference_identities.append(canonical_identities)
        reference_pdb_residue_numbers.append(canonical_pdb_numbers)
        reference_source_chain_ids.append(canonical_source_chains)
        mapping_rows.append(
            {
                "reference_id": reference_id,
                "reference_state": reference_cfg.get("state", ""),
                "reference_path": str(Path(reference_cfg["path"]).resolve()),
                "sequence_condition": reference_cfg["sequence_condition"],
                "cyclic_shift": shift,
                "mapping_orientation": orientation,
                "moving_ring_order": "-".join(moving_ring_order),
                "anchor_ring_order": "-".join(fixed_ring_order),
                "chain_mapping": mapping_text(mapping),
                "core_ca_rmsd_to_anchor_A": core_rmsd,
                "mapping_rmsd_gap_A": mapping_gap,
                "matched_core_ca": matched_atoms,
                "core_coverage": coverage,
                **report,
            }
        )

    npz_path = output_dir / "kv21_aligned_references.npz"
    np.savez_compressed(
        npz_path,
        reference_ids=np.asarray(reference_ids, dtype="U16"),
        coords=np.stack(reference_coords, axis=0),
        present=np.stack(reference_present, axis=0),
        identities=np.stack(reference_identities, axis=0),
        pdb_residue_numbers=np.stack(reference_pdb_residue_numbers, axis=0),
        source_chain_ids=np.stack(reference_source_chain_ids, axis=0),
        raw_residue_numbers=raw_numbers,
        canonical_chains=np.asarray(canonical_chains, dtype="U4"),
        core_mask=core_mask,
    )
    mapping_path = output_dir / "kv21_reference_alignment_report.csv"
    pd.DataFrame(mapping_rows).to_csv(mapping_path, index=False)

    annotation_rows: list[dict[str, Any]] = []
    blocks = cfg.get("masking", {}).get("blocks", [])
    adjacent_window = int(cfg.get("masking", {}).get("adjacent_sequence_window", 5))
    direct_positions: set[int] = set()
    block_by_position: dict[int, list[str]] = {}
    for block in blocks:
        start, end = sorted((int(block["raw_start"]), int(block["raw_end"])))
        for raw in range(start, end + 1):
            direct_positions.add(raw)
            block_by_position.setdefault(raw, []).append(str(block["name"]))

    anchor_index = reference_ids.index(anchor_id)
    anchor_reference_coords = reference_coords[anchor_index]
    anchor_reference_present = reference_present[anchor_index]
    for axis_index, raw in enumerate(raw_numbers.tolist()):
        if direct_positions:
            seq_distance = min(abs(raw - masked) for masked in direct_positions)
        else:
            seq_distance = math.nan
        directly_masked = raw in direct_positions
        adjacent = (not directly_masked) and math.isfinite(seq_distance) and seq_distance <= adjacent_window
        row: dict[str, Any] = {
            "raw_residue_number": raw,
            "directly_masked": directly_masked,
            "mask_names": ";".join(block_by_position.get(raw, [])),
            "sequence_distance_to_nearest_mask": seq_distance,
            "adjacent_to_mask": adjacent,
            "mask_category": "directly_masked" if directly_masked else ("adjacent_to_mask" if adjacent else "unmasked"),
        }
        for condition, sequence in fasta_by_condition.items():
            row[f"residue_identity_{condition}"] = sequence[raw - raw_start] if 0 <= raw - raw_start < len(sequence) else ""

        same_chain_distances: list[float] = []
        neighboring_chain_distances: list[float] = []
        for chain_index in range(len(canonical_chains)):
            if not anchor_reference_present[chain_index, axis_index]:
                continue
            point = anchor_reference_coords[chain_index, axis_index]
            for mask_raw in direct_positions:
                mask_index = mask_raw - raw_start
                if mask_index < 0 or mask_index >= len(raw_numbers):
                    continue
                if anchor_reference_present[chain_index, mask_index]:
                    same_chain_distances.append(float(np.linalg.norm(point - anchor_reference_coords[chain_index, mask_index])))
                for neighbor_index in range(len(canonical_chains)):
                    if neighbor_index == chain_index or not anchor_reference_present[neighbor_index, mask_index]:
                        continue
                    neighboring_chain_distances.append(float(np.linalg.norm(point - anchor_reference_coords[neighbor_index, mask_index])))
        row["8SD3_min_ca_distance_to_mask_same_chain_A"] = min(same_chain_distances) if same_chain_distances else math.nan
        row["8SD3_min_ca_distance_to_mask_neighbor_chain_A"] = min(neighboring_chain_distances) if neighboring_chain_distances else math.nan
        combined = same_chain_distances + neighboring_chain_distances
        row["8SD3_min_ca_distance_to_mask_any_chain_A"] = min(combined) if combined else math.nan
        annotation_rows.append(row)

    annotation_path = output_dir / "kv21_residue_annotations.csv"
    pd.DataFrame(annotation_rows).to_csv(annotation_path, index=False)
    return {"references": npz_path, "mapping_report": mapping_path, "annotations": annotation_path}


def align_manifest_shard(
    cfg: Mapping[str, Any],
    manifest_path: str | Path,
    references_path: str | Path,
    output_dir: str | Path,
    task_id: int,
    task_count: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"aligned_part_{task_id:04d}.npz"
    metadata_path = output_dir / f"aligned_part_{task_id:04d}_metadata.csv"
    if (npz_path.exists() or metadata_path.exists()) and not overwrite:
        raise FileExistsError(f"Shard output exists for task {task_id}; use --overwrite or --resume in the Slurm wrapper")

    manifest = pd.read_csv(manifest_path)
    selected = manifest.loc[manifest["manifest_index"] % task_count == task_id].copy()
    selected = selected.sort_values("manifest_index")
    references = np.load(references_path, allow_pickle=False)
    reference_ids = references["reference_ids"].astype(str).tolist()
    anchor_id = str(cfg["references"]["anchor_id"])
    anchor_index = reference_ids.index(anchor_id)
    anchor_coords = references["coords"][anchor_index].astype(np.float64)
    anchor_present = references["present"][anchor_index].astype(bool)
    core_mask = references["core_mask"].astype(bool)

    raw_start, raw_end, raw_numbers = residue_axis(cfg)
    moving_chains = [str(x) for x in cfg["model_chains"]]
    canonical_chains = [str(x) for x in cfg["canonical_chains"]]
    aligner_cfg = cfg.get("sequence_alignment", {})
    fasta_by_condition = {condition: read_fasta(path) for condition, path in cfg["sequence_fastas"].items()}
    minimum_matched_ca = int(cfg["alignment_core"]["minimum_matched_ca"])
    minimum_coverage = float(cfg["alignment_core"]["minimum_coverage"])
    dtype = np.dtype(str(cfg.get("analysis", {}).get("coordinate_dtype", "float32")))

    n = len(selected)
    coords_output = np.full((n, len(canonical_chains), len(raw_numbers), 3), np.nan, dtype=dtype)
    present_output = np.zeros((n, len(canonical_chains), len(raw_numbers)), dtype=bool)
    identity_output = np.full((n, len(canonical_chains), len(raw_numbers)), "", dtype="U1")
    metadata_rows: list[dict[str, Any]] = []

    for local_index, row in enumerate(selected.itertuples(index=False)):
        meta = row._asdict()
        meta.update(
            {
                "alignment_success": False,
                "alignment_error": "",
                "best_cyclic_shift": math.nan,
                "mapping_orientation": "",
                "moving_ring_order": "",
                "anchor_ring_order": "",
                "chain_mapping": "",
                "best_mapping_core_ca_rmsd_A": math.nan,
                "mapping_rmsd_gap_A": math.nan,
                "matched_core_ca": 0,
                "requested_core_ca": int(core_mask.sum() * len(canonical_chains)),
                "core_coverage": math.nan,
            }
        )
        try:
            structure = load_structure(row.model_path)
            sequence = fasta_by_condition[str(row.sequence_condition)]
            moving_coords, moving_present, moving_identities, report = structure_to_raw_ca(
                structure,
                moving_chains,
                sequence,
                raw_start,
                raw_end,
                aligner_cfg,
            )
            candidates = evaluate_cyclic_raw_fits(
                moving_coords,
                moving_present,
                moving_chains,
                anchor_coords,
                anchor_present,
                canonical_chains,
                core_mask,
                minimum_matched_ca,
                minimum_coverage,
            )
            best, mapping_gap = choose_best_mapping(candidates)
            assert best.fit is not None
            transformed = apply_fit(moving_coords, best.fit)
            canonical_coords, canonical_present, canonical_identities = canonicalize_transformed(
                transformed,
                moving_present,
                moving_identities,
                moving_chains,
                canonical_chains,
                best.mapping,
            )
            coords_output[local_index] = canonical_coords.astype(dtype, copy=False)
            present_output[local_index] = canonical_present
            identity_output[local_index] = canonical_identities
            meta.update(report)
            meta.update(
                {
                    "alignment_success": True,
                    "best_cyclic_shift": best.shift,
                    "mapping_orientation": best.orientation,
                    "moving_ring_order": "-".join(best.moving_ring_order),
                    "anchor_ring_order": "-".join(best.fixed_ring_order),
                    "chain_mapping": mapping_text(best.mapping),
                    "best_mapping_core_ca_rmsd_A": best.fit.rmsd,
                    "mapping_rmsd_gap_A": mapping_gap,
                    "matched_core_ca": best.matched_atoms,
                    "requested_core_ca": best.requested_atoms,
                    "core_coverage": best.coverage,
                }
            )
            for reference_index, reference_id in enumerate(reference_ids):
                rmsd, matched, coverage = finite_rmsd(
                    canonical_coords,
                    references["coords"][reference_index],
                    canonical_present,
                    references["present"][reference_index],
                )
                meta[f"whole_matched_ca_rmsd_to_{reference_id}_A"] = rmsd
                meta[f"whole_matched_ca_count_to_{reference_id}"] = matched
                meta[f"whole_matched_ca_coverage_to_{reference_id}"] = coverage
                core_reference_present = references["present"][reference_index] & core_mask[None, :]
                core_rmsd, core_matched, core_cov = finite_rmsd(
                    canonical_coords,
                    references["coords"][reference_index],
                    canonical_present & core_mask[None, :],
                    core_reference_present,
                )
                meta[f"core_ca_rmsd_to_{reference_id}_A"] = core_rmsd
                meta[f"core_ca_count_to_{reference_id}"] = core_matched
                meta[f"core_ca_coverage_to_{reference_id}"] = core_cov
        except Exception as exc:
            meta["alignment_error"] = f"{type(exc).__name__}: {exc}"
        metadata_rows.append(meta)

    manifest_sha256 = sha256_file(manifest_path)
    references_sha256 = sha256_file(references_path)
    np.savez_compressed(
        npz_path,
        manifest_index=selected["manifest_index"].to_numpy(dtype=np.int64),
        manifest_sha256=np.asarray(manifest_sha256),
        references_sha256=np.asarray(references_sha256),
        task_id=np.asarray(task_id, dtype=np.int32),
        task_count=np.asarray(task_count, dtype=np.int32),
        coords=coords_output,
        present=present_output,
        identities=identity_output,
        raw_residue_numbers=raw_numbers,
        canonical_chains=np.asarray(canonical_chains, dtype="U4"),
    )
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    return {"npz": npz_path, "metadata": metadata_path}


def merge_shards(
    manifest_path: str | Path,
    parts_dir: str | Path,
    output_dir: str | Path,
    allow_missing: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    manifest = pd.read_csv(manifest_path).sort_values("manifest_index").reset_index(drop=True)
    parts_dir = Path(parts_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coords_path = output_dir / "kv21_aligned_ca_coordinates.npy"
    present_path = output_dir / "kv21_aligned_ca_present.npy"
    identities_path = output_dir / "kv21_aligned_residue_identities.npy"
    metadata_path = output_dir / "kv21_alignment_metadata.csv"
    raw_numbers_path = output_dir / "kv21_raw_residue_numbers.npy"
    canonical_chains_path = output_dir / "kv21_canonical_chains.npy"
    for path in (coords_path, present_path, identities_path, metadata_path, raw_numbers_path, canonical_chains_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}. Use --overwrite to replace merged outputs.")

    part_paths = sorted(parts_dir.glob("aligned_part_*.npz"))
    if not part_paths:
        raise FileNotFoundError(f"No aligned_part_*.npz files found in {parts_dir}")
    first = np.load(part_paths[0], allow_pickle=False)
    expected_manifest_sha256 = sha256_file(manifest_path)
    first_manifest_sha256 = str(first["manifest_sha256"].item()) if "manifest_sha256" in first.files else ""
    if first_manifest_sha256 != expected_manifest_sha256:
        raise ValueError("Alignment shards were created from a different model manifest. Remove stale parts and rerun the array.")
    expected_references_sha256 = str(first["references_sha256"].item()) if "references_sha256" in first.files else ""
    n_models = len(manifest)
    n_chains = first["coords"].shape[1]
    n_residues = first["coords"].shape[2]
    coordinate_dtype = first["coords"].dtype
    coords_memmap = np.lib.format.open_memmap(coords_path, mode="w+", dtype=coordinate_dtype, shape=(n_models, n_chains, n_residues, 3))
    coords_memmap[:] = np.nan
    present_memmap = np.lib.format.open_memmap(present_path, mode="w+", dtype=bool, shape=(n_models, n_chains, n_residues))
    present_memmap[:] = False
    identities_memmap = np.lib.format.open_memmap(identities_path, mode="w+", dtype="U1", shape=(n_models, n_chains, n_residues))
    identities_memmap[:] = ""

    seen: set[int] = set()
    metadata_frames: list[pd.DataFrame] = []
    for part_path in part_paths:
        part = np.load(part_path, allow_pickle=False)
        part_manifest_sha256 = str(part["manifest_sha256"].item()) if "manifest_sha256" in part.files else ""
        part_references_sha256 = str(part["references_sha256"].item()) if "references_sha256" in part.files else ""
        if part_manifest_sha256 != expected_manifest_sha256:
            raise ValueError(f"Stale shard with a different manifest: {part_path}")
        if part_references_sha256 != expected_references_sha256:
            raise ValueError(f"Shard uses a different reference bundle: {part_path}")
        indexes = part["manifest_index"].astype(int)
        duplicates = [int(index) for index in indexes if int(index) in seen]
        if duplicates:
            raise ValueError(f"Duplicate manifest indexes in shards: {duplicates[:10]}")
        for index in indexes:
            seen.add(int(index))
        coords_memmap[indexes] = part["coords"]
        present_memmap[indexes] = part["present"]
        identities_memmap[indexes] = part["identities"]
        metadata_part = part_path.with_name(part_path.stem + "_metadata.csv")
        if not metadata_part.exists():
            raise FileNotFoundError(metadata_part)
        metadata_frames.append(pd.read_csv(metadata_part))

    coords_memmap.flush()
    present_memmap.flush()
    identities_memmap.flush()
    expected = set(manifest["manifest_index"].astype(int).tolist())
    missing = sorted(expected - seen)
    if missing and not allow_missing:
        raise ValueError(f"Missing {len(missing)} manifest indexes; first missing values: {missing[:20]}")

    metadata = pd.concat(metadata_frames, ignore_index=True) if metadata_frames else pd.DataFrame()
    if metadata["manifest_index"].duplicated().any():
        duplicates = metadata.loc[metadata["manifest_index"].duplicated(keep=False), "manifest_index"].tolist()
        raise ValueError(f"Duplicate metadata indexes: {duplicates[:20]}")
    metadata = manifest.merge(metadata.drop(columns=[column for column in manifest.columns if column in metadata.columns and column != "manifest_index"]), on="manifest_index", how="left")
    metadata.to_csv(metadata_path, index=False)
    np.save(raw_numbers_path, first["raw_residue_numbers"])
    np.save(canonical_chains_path, first["canonical_chains"])

    failed_path = output_dir / "kv21_failed_models.csv"
    metadata.loc[metadata["alignment_success"] != True].to_csv(failed_path, index=False)  # noqa: E712
    summary_path = output_dir / "kv21_merge_summary.json"
    summary = {
        "manifest_rows": n_models,
        "shard_rows_found": len(seen),
        "missing_rows": len(missing),
        "successful_alignments": int((metadata["alignment_success"] == True).sum()),  # noqa: E712
        "failed_alignments": int((metadata["alignment_success"] != True).sum()),  # noqa: E712
        "coordinates_shape": [n_models, n_chains, n_residues, 3],
        "coordinate_dtype": str(coordinate_dtype),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        "coordinates": coords_path,
        "present": present_path,
        "identities": identities_path,
        "metadata": metadata_path,
        "failed": failed_path,
        "summary": summary_path,
    }


def _subset_metadata(
    metadata: pd.DataFrame,
    subset_query: str | None,
    subset_manifest: str | Path | None,
    subset_column: str | None,
    subset_key: str,
) -> pd.DataFrame:
    selected = metadata.loc[metadata["alignment_success"] == True].copy()  # noqa: E712
    if subset_query:
        selected = selected.query(subset_query, engine="python")
    if subset_manifest:
        subset_df = pd.read_csv(subset_manifest)
        if subset_key not in subset_df.columns:
            raise ValueError(f"Subset manifest does not contain key column {subset_key!r}")
        if subset_column:
            if subset_column not in subset_df.columns:
                raise ValueError(f"Subset manifest does not contain selection column {subset_column!r}")
            values = subset_df[subset_column]
            if values.dtype != bool:
                values = values.astype(str).str.lower().isin({"true", "1", "yes", "y"})
            subset_df = subset_df.loc[values]
        allowed = set(subset_df[subset_key].astype(str))
        if subset_key not in selected.columns:
            raise ValueError(f"Alignment metadata does not contain key column {subset_key!r}")
        selected = selected.loc[selected[subset_key].astype(str).isin(allowed)]
    return selected


def calculate_profiles(
    cfg: Mapping[str, Any],
    merged_dir: str | Path,
    references_path: str | Path,
    annotations_path: str | Path,
    output_dir: str | Path,
    subset_name: str = "all_models",
    subset_query: str | None = None,
    subset_manifest: str | Path | None = None,
    subset_column: str | None = None,
    subset_key: str = "pdb_file",
    minimum_residue_coverage: float | None = None,
) -> dict[str, Path]:
    merged_dir = Path(merged_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(merged_dir / "kv21_alignment_metadata.csv")
    selected = _subset_metadata(metadata, subset_query, subset_manifest, subset_column, subset_key)
    if selected.empty:
        raise ValueError("The selected subset contains no successful aligned models")
    coords = np.load(merged_dir / "kv21_aligned_ca_coordinates.npy", mmap_mode="r")
    present = np.load(merged_dir / "kv21_aligned_ca_present.npy", mmap_mode="r")
    identities = np.load(merged_dir / "kv21_aligned_residue_identities.npy", mmap_mode="r")
    raw_numbers = np.load(merged_dir / "kv21_raw_residue_numbers.npy")
    canonical_chains = np.load(merged_dir / "kv21_canonical_chains.npy").astype(str)
    references = np.load(references_path, allow_pickle=False)
    reference_ids = references["reference_ids"].astype(str).tolist()
    annotations = pd.read_csv(annotations_path)
    coverage_threshold = float(minimum_residue_coverage if minimum_residue_coverage is not None else cfg.get("analysis", {}).get("minimum_residue_coverage", 0.80))

    chain_rows: list[dict[str, Any]] = []
    group_summary_rows: list[dict[str, Any]] = []
    per_model_reference_rows: list[dict[str, Any]] = []

    for dataset_cfg in cfg.get("datasets", []):
        dataset_name = str(dataset_cfg["name"])
        group = selected.loc[selected["dataset"] == dataset_name]
        if group.empty:
            continue
        model_indexes = group["manifest_index"].astype(int).to_numpy()
        group_coords = np.asarray(coords[model_indexes], dtype=np.float64)
        group_present = np.asarray(present[model_indexes], dtype=bool)
        n_models = len(model_indexes)

        for chain_index, chain_id in enumerate(canonical_chains):
            chain_coords = group_coords[:, chain_index, :, :]
            chain_present = group_present[:, chain_index, :]
            n_present = chain_present.sum(axis=0)
            coverage = n_present / n_models
            masked_coords = np.where(chain_present[:, :, None], chain_coords, np.nan)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_coords = np.nanmean(masked_coords, axis=0)
                squared_distance = np.sum((masked_coords - mean_coords[None, :, :]) ** 2, axis=2)
                rmsf = np.sqrt(np.nanmean(squared_distance, axis=0))
            rmsf[coverage < coverage_threshold] = np.nan

            for residue_index, raw in enumerate(raw_numbers):
                row: dict[str, Any] = {
                    "channel": cfg["project"]["channel"],
                    "dataset": dataset_name,
                    "sequence_condition": dataset_cfg["sequence_condition"],
                    "protocol": dataset_cfg["protocol"],
                    "subset": subset_name,
                    "chain": chain_id,
                    "raw_residue_number": int(raw),
                    "residue_identity": _modal_identity(identities[model_indexes, chain_index, residue_index]),
                    "number_of_models": n_models,
                    "number_with_residue_resolved": int(n_present[residue_index]),
                    "coverage_fraction": float(coverage[residue_index]),
                    "coverage_threshold": coverage_threshold,
                    "passes_coverage_threshold": bool(coverage[residue_index] >= coverage_threshold),
                    "ensemble_mean_x_A": float(mean_coords[residue_index, 0]) if np.isfinite(mean_coords[residue_index, 0]) else math.nan,
                    "ensemble_mean_y_A": float(mean_coords[residue_index, 1]) if np.isfinite(mean_coords[residue_index, 1]) else math.nan,
                    "ensemble_mean_z_A": float(mean_coords[residue_index, 2]) if np.isfinite(mean_coords[residue_index, 2]) else math.nan,
                    "ensemble_rmsf_A": float(rmsf[residue_index]) if np.isfinite(rmsf[residue_index]) else math.nan,
                }
                for reference_index, reference_id in enumerate(reference_ids):
                    reference_coord = references["coords"][reference_index, chain_index, residue_index].astype(np.float64)
                    reference_is_present = bool(references["present"][reference_index, chain_index, residue_index])
                    valid_models = chain_present[:, residue_index]
                    if reference_is_present and valid_models.any() and np.isfinite(mean_coords[residue_index]).all():
                        deviations = chain_coords[valid_models, residue_index, :] - reference_coord[None, :]
                        row[f"mean_coordinate_distance_to_{reference_id}_A"] = float(np.linalg.norm(mean_coords[residue_index] - reference_coord))
                        row[f"rms_deviation_to_{reference_id}_A"] = float(np.sqrt(np.mean(np.sum(deviations * deviations, axis=1))))
                        row[f"reference_{reference_id}_available"] = True
                        row[f"reference_{reference_id}_raw_identity"] = str(references["identities"][reference_index, chain_index, residue_index])
                        pdb_number = int(references["pdb_residue_numbers"][reference_index, chain_index, residue_index])
                        row[f"reference_{reference_id}_pdb_residue_number"] = pdb_number if pdb_number >= 0 else math.nan
                        row[f"reference_{reference_id}_source_chain"] = str(references["source_chain_ids"][reference_index, chain_index])
                    else:
                        row[f"mean_coordinate_distance_to_{reference_id}_A"] = math.nan
                        row[f"rms_deviation_to_{reference_id}_A"] = math.nan
                        row[f"reference_{reference_id}_available"] = False
                        row[f"reference_{reference_id}_raw_identity"] = ""
                        row[f"reference_{reference_id}_pdb_residue_number"] = math.nan
                        row[f"reference_{reference_id}_source_chain"] = ""
                chain_rows.append(row)

        for local_model_index, metadata_row in enumerate(group.itertuples(index=False)):
            model_coords = group_coords[local_model_index]
            model_present = group_present[local_model_index]
            for reference_index, reference_id in enumerate(reference_ids):
                rmsd, matched, reference_coverage = finite_rmsd(
                    model_coords,
                    references["coords"][reference_index],
                    model_present,
                    references["present"][reference_index],
                )
                per_model_reference_rows.append(
                    {
                        "manifest_index": int(metadata_row.manifest_index),
                        "dataset": dataset_name,
                        "sequence_condition": dataset_cfg["sequence_condition"],
                        "protocol": dataset_cfg["protocol"],
                        "subset": subset_name,
                        "pdb_file": metadata_row.pdb_file,
                        "model_path": metadata_row.model_path,
                        "trajectory_id": metadata_row.trajectory_id,
                        "seed": metadata_row.seed,
                        "model_number": metadata_row.model_number,
                        "recycle_label": metadata_row.recycle_label,
                        "reference_id": reference_id,
                        "whole_matched_ca_rmsd_A": rmsd,
                        "matched_ca": matched,
                        "reference_coverage": reference_coverage,
                    }
                )

    chain_df = pd.DataFrame(chain_rows).merge(annotations, on="raw_residue_number", how="left")
    chain_path = output_dir / f"kv21_{subset_name}_chain_resolved_profiles.csv"
    chain_df.to_csv(chain_path, index=False)

    chain_comparison_df = _build_chain_masked_vanilla_comparisons(chain_df, reference_ids)
    chain_comparison_path = output_dir / f"kv21_{subset_name}_chain_resolved_masked_vs_vanilla.csv"
    chain_comparison_df.to_csv(chain_comparison_path, index=False)

    symmetry_df = _build_symmetry_profiles(chain_df, reference_ids)
    symmetry_path = output_dir / f"kv21_{subset_name}_symmetry_averaged_profiles.csv"
    symmetry_df.to_csv(symmetry_path, index=False)

    comparison_df = _build_masked_vanilla_comparisons(symmetry_df, reference_ids)
    comparison_path = output_dir / f"kv21_{subset_name}_masked_vs_vanilla_comparisons.csv"
    comparison_df.to_csv(comparison_path, index=False)

    per_model_reference_path = output_dir / f"kv21_{subset_name}_per_model_reference_rmsd.csv"
    pd.DataFrame(per_model_reference_rows).to_csv(per_model_reference_path, index=False)

    group_summary = _build_group_summary(symmetry_df)
    group_summary_path = output_dir / f"kv21_{subset_name}_whole_protein_and_mask_summary.csv"
    group_summary.to_csv(group_summary_path, index=False)

    selected_manifest_path = output_dir / f"kv21_{subset_name}_selected_models.csv"
    selected.to_csv(selected_manifest_path, index=False)
    return {
        "chain_profiles": chain_path,
        "chain_comparisons": chain_comparison_path,
        "symmetry_profiles": symmetry_path,
        "comparisons": comparison_path,
        "per_model_reference": per_model_reference_path,
        "group_summary": group_summary_path,
        "selected_models": selected_manifest_path,
    }


def _modal_identity(values: np.ndarray) -> str:
    cleaned = [str(value) for value in values.tolist() if str(value)]
    if not cleaned:
        return ""
    counts = pd.Series(cleaned).value_counts()
    return str(counts.index[0])


def _build_chain_masked_vanilla_comparisons(chain_df: pd.DataFrame, reference_ids: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["channel", "sequence_condition", "subset", "chain", "raw_residue_number"]
    for key_values, group in chain_df.groupby(keys, sort=False, dropna=False):
        by_protocol = {str(row["protocol"]): row for _, row in group.iterrows()}
        if "masked" not in by_protocol or "vanilla" not in by_protocol:
            continue
        masked = by_protocol["masked"]
        vanilla = by_protocol["vanilla"]
        row: dict[str, Any] = dict(zip(keys, key_values))
        for column in (
            "residue_identity", "directly_masked", "mask_names", "mask_category",
            "sequence_distance_to_nearest_mask", "adjacent_to_mask",
            "8SD3_min_ca_distance_to_mask_same_chain_A",
            "8SD3_min_ca_distance_to_mask_neighbor_chain_A",
            "8SD3_min_ca_distance_to_mask_any_chain_A",
        ):
            if column in masked.index:
                row[column] = masked[column]
        row["masked_dataset"] = masked["dataset"]
        row["vanilla_dataset"] = vanilla["dataset"]
        row["masked_rmsf_A"] = masked["ensemble_rmsf_A"]
        row["vanilla_rmsf_A"] = vanilla["ensemble_rmsf_A"]
        if pd.notna(masked["ensemble_rmsf_A"]) and pd.notna(vanilla["ensemble_rmsf_A"]):
            row["masked_minus_vanilla_rmsf_A"] = float(masked["ensemble_rmsf_A"] - vanilla["ensemble_rmsf_A"])
            row["masked_divided_by_vanilla_rmsf"] = float(masked["ensemble_rmsf_A"] / vanilla["ensemble_rmsf_A"]) if vanilla["ensemble_rmsf_A"] != 0 else math.nan
        else:
            row["masked_minus_vanilla_rmsf_A"] = math.nan
            row["masked_divided_by_vanilla_rmsf"] = math.nan
        for reference_id in reference_ids:
            for metric in ("mean_coordinate_distance", "rms_deviation"):
                column = f"{metric}_to_{reference_id}_A"
                masked_value = masked[column]
                vanilla_value = vanilla[column]
                row[f"masked_{column}"] = masked_value
                row[f"vanilla_{column}"] = vanilla_value
                row[f"masked_minus_vanilla_{column}"] = float(masked_value - vanilla_value) if pd.notna(masked_value) and pd.notna(vanilla_value) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_symmetry_profiles(chain_df: pd.DataFrame, reference_ids: Sequence[str]) -> pd.DataFrame:
    group_columns = ["channel", "dataset", "sequence_condition", "protocol", "subset", "raw_residue_number"]
    first_columns = [
        "residue_identity", "number_of_models", "coverage_threshold", "directly_masked", "mask_names",
        "sequence_distance_to_nearest_mask", "adjacent_to_mask", "mask_category",
        "residue_identity_wt", "residue_identity_l403a", "residue_identity_f412l",
        "8SD3_min_ca_distance_to_mask_same_chain_A", "8SD3_min_ca_distance_to_mask_neighbor_chain_A",
        "8SD3_min_ca_distance_to_mask_any_chain_A",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in chain_df.groupby(group_columns, sort=False, dropna=False):
        row = dict(zip(group_columns, keys))
        for column in first_columns:
            if column in group.columns:
                row[column] = group[column].iloc[0]
        valid_rmsf = group["ensemble_rmsf_A"].dropna().to_numpy(dtype=float)
        row["chains_with_valid_rmsf"] = len(valid_rmsf)
        row["symmetry_averaged_rmsf_A"] = float(np.mean(valid_rmsf)) if len(valid_rmsf) else math.nan
        row["chain_to_chain_rmsf_std_A"] = float(np.std(valid_rmsf, ddof=0)) if len(valid_rmsf) else math.nan
        row["chain_min_rmsf_A"] = float(np.min(valid_rmsf)) if len(valid_rmsf) else math.nan
        row["chain_max_rmsf_A"] = float(np.max(valid_rmsf)) if len(valid_rmsf) else math.nan
        row["mean_chain_coverage_fraction"] = float(group["coverage_fraction"].mean())
        row["minimum_chain_coverage_fraction"] = float(group["coverage_fraction"].min())
        for reference_id in reference_ids:
            for metric in ("mean_coordinate_distance", "rms_deviation"):
                column = f"{metric}_to_{reference_id}_A"
                values = group[column].dropna().to_numpy(dtype=float)
                row[f"symmetry_averaged_{column}"] = float(np.mean(values)) if len(values) else math.nan
                row[f"chain_std_{column}"] = float(np.std(values, ddof=0)) if len(values) else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _build_masked_vanilla_comparisons(symmetry_df: pd.DataFrame, reference_ids: Sequence[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["channel", "sequence_condition", "subset", "raw_residue_number"]
    for key_values, group in symmetry_df.groupby(keys, sort=False, dropna=False):
        protocols = {str(row.protocol): row for row in group.itertuples(index=False)}
        if "masked" not in protocols or "vanilla" not in protocols:
            continue
        masked = protocols["masked"]
        vanilla = protocols["vanilla"]
        row: dict[str, Any] = dict(zip(keys, key_values))
        row.update(
            {
                "masked_dataset": masked.dataset,
                "vanilla_dataset": vanilla.dataset,
                "residue_identity": masked.residue_identity,
                "directly_masked": masked.directly_masked,
                "mask_names": masked.mask_names,
                "mask_category": masked.mask_category,
                "sequence_distance_to_nearest_mask": masked.sequence_distance_to_nearest_mask,
                "masked_symmetry_averaged_rmsf_A": masked.symmetry_averaged_rmsf_A,
                "vanilla_symmetry_averaged_rmsf_A": vanilla.symmetry_averaged_rmsf_A,
            }
        )
        if math.isfinite(masked.symmetry_averaged_rmsf_A) and math.isfinite(vanilla.symmetry_averaged_rmsf_A):
            row["masked_minus_vanilla_rmsf_A"] = masked.symmetry_averaged_rmsf_A - vanilla.symmetry_averaged_rmsf_A
            row["masked_divided_by_vanilla_rmsf"] = masked.symmetry_averaged_rmsf_A / vanilla.symmetry_averaged_rmsf_A if vanilla.symmetry_averaged_rmsf_A != 0 else math.nan
        else:
            row["masked_minus_vanilla_rmsf_A"] = math.nan
            row["masked_divided_by_vanilla_rmsf"] = math.nan
        for reference_id in reference_ids:
            for metric in ("mean_coordinate_distance", "rms_deviation"):
                column = f"symmetry_averaged_{metric}_to_{reference_id}_A"
                masked_value = getattr(masked, column)
                vanilla_value = getattr(vanilla, column)
                row[f"masked_{metric}_to_{reference_id}_A"] = masked_value
                row[f"vanilla_{metric}_to_{reference_id}_A"] = vanilla_value
                row[f"masked_minus_vanilla_{metric}_to_{reference_id}_A"] = masked_value - vanilla_value if math.isfinite(masked_value) and math.isfinite(vanilla_value) else math.nan
        # Copy structural-distance annotations by column name rather than namedtuple attribute.
        source = group.loc[group["protocol"] == "masked"].iloc[0]
        for column in (
            "8SD3_min_ca_distance_to_mask_same_chain_A",
            "8SD3_min_ca_distance_to_mask_neighbor_chain_A",
            "8SD3_min_ca_distance_to_mask_any_chain_A",
            "adjacent_to_mask",
        ):
            if column in source.index:
                row[column] = source[column]
        rows.append(row)
    return pd.DataFrame(rows)


def _build_group_summary(symmetry_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dataset, condition, protocol, subset), group in symmetry_df.groupby(["dataset", "sequence_condition", "protocol", "subset"], sort=False):
        for category_name, category_group in [("whole_protein", group), *[(str(category), sub) for category, sub in group.groupby("mask_category", dropna=False)]]:
            values = category_group["symmetry_averaged_rmsf_A"].dropna().to_numpy(dtype=float)
            rows.append(
                {
                    "dataset": dataset,
                    "sequence_condition": condition,
                    "protocol": protocol,
                    "subset": subset,
                    "residue_class": category_name,
                    "number_of_residues": len(category_group),
                    "number_with_valid_rmsf": len(values),
                    "median_rmsf_A": float(np.median(values)) if len(values) else math.nan,
                    "mean_rmsf_A": float(np.mean(values)) if len(values) else math.nan,
                    "maximum_rmsf_A": float(np.max(values)) if len(values) else math.nan,
                    "integrated_rmsf_A": float(np.sum(values)) if len(values) else math.nan,
                    "median_coverage_fraction": float(category_group["mean_chain_coverage_fraction"].median()),
                }
            )
    return pd.DataFrame(rows)
