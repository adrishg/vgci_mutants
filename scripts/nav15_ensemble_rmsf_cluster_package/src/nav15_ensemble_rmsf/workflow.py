from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .core import (
    apply_fit,
    chain_to_raw_ca,
    discover_structures,
    finite_rmsd,
    kabsch_fit,
    load_structure,
    parse_af2_filename,
    ranges_to_mask,
    read_fasta,
)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def residue_axis(cfg: Mapping[str, Any]) -> tuple[int, int, np.ndarray]:
    start = int(cfg["residue_axis"]["start"])
    end = int(cfg["residue_axis"]["end"])
    if end < start:
        raise ValueError("residue_axis.end must be >= residue_axis.start")
    return start, end, np.arange(start, end + 1, dtype=np.int32)


def build_manifest(cfg: Mapping[str, Any]) -> pd.DataFrame:
    recursive = bool(cfg.get("analysis", {}).get("recursive", True))
    rows: list[dict[str, Any]] = []
    for dataset in cfg["datasets"]:
        root = Path(dataset["path"])
        for path in discover_structures(root, recursive=recursive):
            parsed = parse_af2_filename(path.name)
            rows.append(
                {
                    "dataset": str(dataset["name"]),
                    "sequence_condition": str(dataset["sequence_condition"]),
                    "protocol": str(dataset["protocol"]),
                    "dataset_root": str(root.resolve()),
                    "model_path": str(path),
                    **parsed,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        columns = [
            "manifest_index", "dataset", "sequence_condition", "protocol", "dataset_root",
            "model_path", "pdb_file", "filename_stem", "trajectory_id", "recycle_label",
            "recycle_index", "is_final_model", "rank", "model_number", "seed", "relaxation",
            "af2_protocol",
        ]
        return pd.DataFrame(columns=columns)
    frame = frame.sort_values(["dataset", "model_path"]).reset_index(drop=True)
    frame.insert(0, "manifest_index", np.arange(len(frame), dtype=np.int64))
    return frame


def inspect_inputs(cfg: Mapping[str, Any], output_dir: str | Path, max_models_per_dataset: int = 3) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(cfg)
    manifest_path = output_dir / "nav15_model_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    counts_path = output_dir / "nav15_dataset_counts.csv"
    if manifest.empty:
        counts = pd.DataFrame(columns=["dataset", "structure_count"])
    else:
        counts = manifest.groupby(["dataset", "sequence_condition", "protocol"], as_index=False).size().rename(columns={"size": "structure_count"})
    counts.to_csv(counts_path, index=False)

    path_rows = []
    for dataset in cfg["datasets"]:
        root = Path(dataset["path"])
        structures = discover_structures(root, recursive=bool(cfg.get("analysis", {}).get("recursive", True)))
        path_rows.append(
            {
                "dataset": dataset["name"],
                "path": str(root),
                "exists": root.exists(),
                "is_directory": root.is_dir(),
                "structure_count": len(structures),
            }
        )
    dataset_status_path = output_dir / "nav15_dataset_path_status.csv"
    pd.DataFrame(path_rows).to_csv(dataset_status_path, index=False)

    fasta_rows = []
    for condition, path_value in cfg["sequence_fastas"].items():
        path = Path(path_value)
        row = {"sequence_condition": condition, "path": str(path), "exists": path.exists()}
        if path.exists():
            try:
                sequence = read_fasta(path)
                row.update({"status": "ok", "length": len(sequence), "first_20": sequence[:20], "last_20": sequence[-20:]})
            except Exception as exc:
                row.update({"status": f"error:{type(exc).__name__}:{exc}", "length": math.nan})
        else:
            row.update({"status": "missing", "length": math.nan})
        fasta_rows.append(row)
    fasta_status_path = output_dir / "nav15_fasta_status.csv"
    pd.DataFrame(fasta_rows).to_csv(fasta_status_path, index=False)

    raw_start, raw_end, _ = residue_axis(cfg)
    aligner_cfg = cfg.get("sequence_alignment", {})
    fasta_by_condition = {
        condition: read_fasta(path)
        for condition, path in cfg["sequence_fastas"].items()
        if Path(path).exists()
    }
    inventory_rows = []
    if not manifest.empty:
        for dataset_name, group in manifest.groupby("dataset", sort=True):
            for _, row in group.head(max_models_per_dataset).iterrows():
                result = {
                    "dataset": dataset_name,
                    "model_path": row["model_path"],
                    "sequence_condition": row["sequence_condition"],
                }
                try:
                    structure = load_structure(row["model_path"])
                    sequence = fasta_by_condition[str(row["sequence_condition"])]
                    coords, present, identities, pdb_numbers, report = chain_to_raw_ca(
                        structure, str(cfg.get("model_chain", "A")), sequence, raw_start, raw_end, aligner_cfg
                    )
                    result.update(report)
                    result["chain_ids_found"] = ";".join(sorted(structure.chains))
                    result["raw_axis_ca_coverage"] = float(present.mean())
                except Exception as exc:
                    result["status"] = f"error:{type(exc).__name__}:{exc}"
                inventory_rows.append(result)
    structure_inventory_path = output_dir / "nav15_structure_inventory.csv"
    pd.DataFrame(inventory_rows).to_csv(structure_inventory_path, index=False)

    reference_rows = []
    for reference in cfg["references"]["structures"]:
        path = Path(reference["path"])
        row = {
            "reference_id": reference["id"],
            "state": reference.get("state", ""),
            "relationship": reference.get("relationship", ""),
            "path": str(path),
            "exists": path.exists(),
            "chain": reference.get("chain", "A"),
            "sequence_condition": reference["sequence_condition"],
        }
        if path.exists() and str(reference["sequence_condition"]) in fasta_by_condition:
            try:
                structure = load_structure(path)
                coords, present, identities, pdb_numbers, report = chain_to_raw_ca(
                    structure,
                    str(reference.get("chain", "A")),
                    fasta_by_condition[str(reference["sequence_condition"])],
                    raw_start,
                    raw_end,
                    aligner_cfg,
                )
                row.update(report)
                row["chain_ids_found"] = ";".join(sorted(structure.chains))
                row["raw_axis_ca_coverage"] = float(present.mean())
            except Exception as exc:
                row["status"] = f"error:{type(exc).__name__}:{exc}"
        reference_rows.append(row)
    reference_inventory_path = output_dir / "nav15_reference_inventory.csv"
    pd.DataFrame(reference_rows).to_csv(reference_inventory_path, index=False)

    return {
        "manifest": manifest_path,
        "counts": counts_path,
        "dataset_status": dataset_status_path,
        "fasta_status": fasta_status_path,
        "structure_inventory": structure_inventory_path,
        "reference_inventory": reference_inventory_path,
    }


def prepare_references(cfg: Mapping[str, Any], output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_start, raw_end, raw_numbers = residue_axis(cfg)
    aligner_cfg = cfg.get("sequence_alignment", {})
    fasta_by_condition = {condition: read_fasta(path) for condition, path in cfg["sequence_fastas"].items()}
    reference_cfgs = {str(item["id"]): item for item in cfg["references"]["structures"]}
    anchor_id = str(cfg["references"]["anchor_id"])
    if anchor_id not in reference_cfgs:
        raise ValueError(f"Anchor reference {anchor_id!r} is not listed")
    core_mask = ranges_to_mask(raw_start, raw_end, cfg["alignment_core"]["raw_ranges"])
    minimum_matched_ca = int(cfg["alignment_core"]["minimum_matched_ca"])
    minimum_coverage = float(cfg["alignment_core"]["minimum_coverage"])

    anchor_cfg = reference_cfgs[anchor_id]
    anchor_structure = load_structure(anchor_cfg["path"])
    anchor_coords, anchor_present, anchor_identities, anchor_pdb_numbers, anchor_report = chain_to_raw_ca(
        anchor_structure,
        str(anchor_cfg.get("chain", "A")),
        fasta_by_condition[str(anchor_cfg["sequence_condition"])],
        raw_start,
        raw_end,
        aligner_cfg,
    )
    anchor_core_count = int((core_mask & anchor_present).sum())
    if anchor_core_count < minimum_matched_ca:
        raise ValueError(f"Anchor {anchor_id} has only {anchor_core_count} alignment-core Cα atoms")

    reference_ids = []
    coords_list = []
    present_list = []
    identities_list = []
    pdb_numbers_list = []
    report_rows = []

    for reference_id, reference_cfg in reference_cfgs.items():
        structure = load_structure(reference_cfg["path"])
        coords, present, identities, pdb_numbers, report = chain_to_raw_ca(
            structure,
            str(reference_cfg.get("chain", "A")),
            fasta_by_condition[str(reference_cfg["sequence_condition"])],
            raw_start,
            raw_end,
            aligner_cfg,
        )
        if reference_id == anchor_id:
            transformed = coords
            core_rmsd = 0.0
            matched = anchor_core_count
            coverage = 1.0
        else:
            valid = core_mask & present & anchor_present
            matched = int(valid.sum())
            requested = int((core_mask & anchor_present).sum())
            coverage = matched / requested if requested else math.nan
            if matched < minimum_matched_ca:
                raise ValueError(f"Reference {reference_id} has only {matched} matched core Cα atoms")
            if not math.isfinite(coverage) or coverage < minimum_coverage:
                raise ValueError(f"Reference {reference_id} core coverage {coverage:.3f} is below {minimum_coverage:.3f}")
            fit = kabsch_fit(coords[valid], anchor_coords[valid])
            transformed = apply_fit(coords, fit)
            core_rmsd = fit.rmsd
        reference_ids.append(reference_id)
        coords_list.append(transformed.astype(np.float32))
        present_list.append(present)
        identities_list.append(identities)
        pdb_numbers_list.append(pdb_numbers)
        report_rows.append(
            {
                "reference_id": reference_id,
                "state": reference_cfg.get("state", ""),
                "relationship": reference_cfg.get("relationship", ""),
                "species": reference_cfg.get("species", ""),
                "reference_path": str(Path(reference_cfg["path"]).resolve()),
                "sequence_condition": reference_cfg["sequence_condition"],
                "chain": reference_cfg.get("chain", "A"),
                "core_ca_rmsd_to_anchor_A": core_rmsd,
                "matched_core_ca": matched,
                "core_coverage": coverage,
                **report,
            }
        )

    bundle_path = output_dir / "nav15_aligned_references.npz"
    np.savez_compressed(
        bundle_path,
        reference_ids=np.asarray(reference_ids, dtype="U16"),
        coords=np.stack(coords_list),
        present=np.stack(present_list),
        identities=np.stack(identities_list),
        pdb_residue_numbers=np.stack(pdb_numbers_list),
        raw_residue_numbers=raw_numbers,
        core_mask=core_mask,
        anchor_id=np.asarray(anchor_id),
    )
    report_path = output_dir / "nav15_reference_alignment_report.csv"
    pd.DataFrame(report_rows).to_csv(report_path, index=False)

    annotation_rows = []
    region_blocks = cfg.get("annotations", {}).get("raw_regions", [])
    fasta_wt = fasta_by_condition.get("wt", "")
    fasta_qqq = fasta_by_condition.get("qqq", "")
    for axis_index, raw in enumerate(raw_numbers.tolist()):
        region_names = [
            str(block["name"])
            for block in region_blocks
            if int(block["raw_start"]) <= raw <= int(block["raw_end"])
        ]
        annotation_rows.append(
            {
                "raw_residue_number": raw,
                "residue_identity_wt": fasta_wt[axis_index] if axis_index < len(fasta_wt) else "",
                "residue_identity_qqq": fasta_qqq[axis_index] if axis_index < len(fasta_qqq) else "",
                "alignment_core": bool(core_mask[axis_index]),
                "annotation_regions": ";".join(region_names),
                "ifm_motif": any(name == "IFM_motif" for name in region_names),
                "mask_annotation_status": "deferred",
            }
        )
    annotations_path = output_dir / "nav15_residue_annotations.csv"
    pd.DataFrame(annotation_rows).to_csv(annotations_path, index=False)
    return {"references": bundle_path, "mapping_report": report_path, "annotations": annotations_path}


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
        raise FileExistsError(f"Shard output exists for task {task_id}; use --overwrite")

    manifest = pd.read_csv(manifest_path)
    selected = manifest.loc[manifest["manifest_index"].astype(int) % task_count == task_id].copy()
    selected = selected.sort_values("manifest_index").reset_index(drop=True)
    references = np.load(references_path, allow_pickle=False)
    reference_ids = references["reference_ids"].astype(str).tolist()
    anchor_id = str(references["anchor_id"].item())
    anchor_index = reference_ids.index(anchor_id)
    anchor_coords = references["coords"][anchor_index].astype(np.float64)
    anchor_present = references["present"][anchor_index].astype(bool)
    core_mask = references["core_mask"].astype(bool)
    raw_numbers = references["raw_residue_numbers"].astype(np.int32)
    raw_start, raw_end = int(raw_numbers[0]), int(raw_numbers[-1])
    n_residues = len(raw_numbers)
    coord_dtype = np.dtype(cfg.get("analysis", {}).get("coordinate_dtype", "float32"))
    coords_output = np.full((len(selected), n_residues, 3), np.nan, dtype=coord_dtype)
    present_output = np.zeros((len(selected), n_residues), dtype=bool)
    identity_output = np.full((len(selected), n_residues), "", dtype="U1")
    minimum_matched_ca = int(cfg["alignment_core"]["minimum_matched_ca"])
    minimum_coverage = float(cfg["alignment_core"]["minimum_coverage"])
    aligner_cfg = cfg.get("sequence_alignment", {})
    fasta_by_condition = {condition: read_fasta(path) for condition, path in cfg["sequence_fastas"].items()}

    metadata_rows = []
    for output_index, row in selected.iterrows():
        meta = row.to_dict()
        meta.update({"alignment_success": False, "alignment_error": ""})
        try:
            structure = load_structure(row["model_path"])
            condition = str(row["sequence_condition"])
            coords, present, identities, pdb_numbers, report = chain_to_raw_ca(
                structure,
                str(cfg.get("model_chain", "A")),
                fasta_by_condition[condition],
                raw_start,
                raw_end,
                aligner_cfg,
            )
            valid = core_mask & present & anchor_present
            matched = int(valid.sum())
            requested = int((core_mask & anchor_present).sum())
            coverage = matched / requested if requested else math.nan
            if matched < minimum_matched_ca:
                raise ValueError(f"matched_core_ca<{minimum_matched_ca}: {matched}")
            if not math.isfinite(coverage) or coverage < minimum_coverage:
                raise ValueError(f"core_coverage<{minimum_coverage:.3f}: {coverage:.3f}")
            fit = kabsch_fit(coords[valid], anchor_coords[valid])
            transformed = apply_fit(coords, fit)
            coords_output[output_index] = transformed.astype(coord_dtype)
            present_output[output_index] = present
            identity_output[output_index] = identities
            meta.update(
                {
                    "alignment_success": True,
                    "core_ca_rmsd_to_anchor_A": fit.rmsd,
                    "matched_core_ca": matched,
                    "core_coverage": coverage,
                    **report,
                }
            )
            for reference_index, reference_id in enumerate(reference_ids):
                whole_rmsd, whole_count, whole_coverage = finite_rmsd(
                    transformed,
                    references["coords"][reference_index],
                    present,
                    references["present"][reference_index],
                )
                core_rmsd, core_count, core_coverage = finite_rmsd(
                    transformed,
                    references["coords"][reference_index],
                    present & core_mask,
                    references["present"][reference_index] & core_mask,
                )
                meta[f"whole_matched_ca_rmsd_to_{reference_id}_A"] = whole_rmsd
                meta[f"whole_ca_count_to_{reference_id}"] = whole_count
                meta[f"whole_ca_coverage_to_{reference_id}"] = whole_coverage
                meta[f"core_ca_rmsd_to_{reference_id}_A"] = core_rmsd
                meta[f"core_ca_count_to_{reference_id}"] = core_count
                meta[f"core_ca_coverage_to_{reference_id}"] = core_coverage
        except Exception as exc:
            meta["alignment_error"] = f"{type(exc).__name__}: {exc}"
        metadata_rows.append(meta)

    np.savez_compressed(
        npz_path,
        manifest_index=selected["manifest_index"].to_numpy(dtype=np.int64),
        manifest_sha256=np.asarray(sha256_file(manifest_path)),
        references_sha256=np.asarray(sha256_file(references_path)),
        task_id=np.asarray(task_id, dtype=np.int32),
        task_count=np.asarray(task_count, dtype=np.int32),
        coords=coords_output,
        present=present_output,
        identities=identity_output,
        raw_residue_numbers=raw_numbers,
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
    paths = {
        "coordinates": output_dir / "nav15_aligned_ca_coordinates.npy",
        "present": output_dir / "nav15_aligned_ca_present.npy",
        "identities": output_dir / "nav15_aligned_residue_identities.npy",
        "metadata": output_dir / "nav15_alignment_metadata.csv",
        "raw_numbers": output_dir / "nav15_raw_residue_numbers.npy",
    }
    for path in paths.values():
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}")
    part_paths = sorted(parts_dir.glob("aligned_part_*.npz"))
    if not part_paths:
        raise FileNotFoundError(f"No aligned_part_*.npz files found in {parts_dir}")
    first = np.load(part_paths[0], allow_pickle=False)
    expected_manifest_hash = sha256_file(manifest_path)
    if str(first["manifest_sha256"].item()) != expected_manifest_hash:
        raise ValueError("Alignment shards were created from a different manifest")
    reference_hash = str(first["references_sha256"].item())
    n_models = len(manifest)
    n_residues = first["coords"].shape[1]
    coords_mm = np.lib.format.open_memmap(paths["coordinates"], mode="w+", dtype=first["coords"].dtype, shape=(n_models, n_residues, 3))
    coords_mm[:] = np.nan
    present_mm = np.lib.format.open_memmap(paths["present"], mode="w+", dtype=bool, shape=(n_models, n_residues))
    present_mm[:] = False
    identities_mm = np.lib.format.open_memmap(paths["identities"], mode="w+", dtype="U1", shape=(n_models, n_residues))
    identities_mm[:] = ""
    seen: set[int] = set()
    metadata_frames = []
    for part_path in part_paths:
        part = np.load(part_path, allow_pickle=False)
        if str(part["manifest_sha256"].item()) != expected_manifest_hash:
            raise ValueError(f"Stale shard: {part_path}")
        if str(part["references_sha256"].item()) != reference_hash:
            raise ValueError(f"Different reference bundle: {part_path}")
        indexes = part["manifest_index"].astype(int)
        duplicate = [int(index) for index in indexes if int(index) in seen]
        if duplicate:
            raise ValueError(f"Duplicate manifest indexes: {duplicate[:10]}")
        seen.update(int(index) for index in indexes)
        coords_mm[indexes] = part["coords"]
        present_mm[indexes] = part["present"]
        identities_mm[indexes] = part["identities"]
        metadata_file = part_path.with_name(part_path.stem + "_metadata.csv")
        if not metadata_file.exists():
            raise FileNotFoundError(metadata_file)
        metadata_frames.append(pd.read_csv(metadata_file))
    coords_mm.flush(); present_mm.flush(); identities_mm.flush()
    missing = sorted(set(manifest["manifest_index"].astype(int)) - seen)
    if missing and not allow_missing:
        raise ValueError(f"Missing {len(missing)} manifest indexes; first: {missing[:20]}")
    metadata = pd.concat(metadata_frames, ignore_index=True)
    if metadata["manifest_index"].duplicated().any():
        raise ValueError("Duplicate metadata manifest indexes")
    extra_columns = [column for column in metadata.columns if column not in manifest.columns or column == "manifest_index"]
    metadata = manifest.merge(metadata[extra_columns], on="manifest_index", how="left")
    metadata.to_csv(paths["metadata"], index=False)
    np.save(paths["raw_numbers"], first["raw_residue_numbers"])
    failed_path = output_dir / "nav15_failed_models.csv"
    metadata.loc[metadata["alignment_success"] != True].to_csv(failed_path, index=False)  # noqa: E712
    summary_path = output_dir / "nav15_merge_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "manifest_rows": n_models,
                "shard_rows_found": len(seen),
                "missing_rows": len(missing),
                "successful_alignments": int((metadata["alignment_success"] == True).sum()),  # noqa: E712
                "failed_alignments": int((metadata["alignment_success"] != True).sum()),  # noqa: E712
                "coordinates_shape": [n_models, n_residues, 3],
                "coordinate_dtype": str(first["coords"].dtype),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {**paths, "failed": failed_path, "summary": summary_path}


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
            raise ValueError(f"Subset manifest missing key {subset_key!r}")
        if subset_column:
            if subset_column not in subset_df.columns:
                raise ValueError(f"Subset manifest missing selection column {subset_column!r}")
            values = subset_df[subset_column]
            if values.dtype != bool:
                values = values.astype(str).str.lower().isin({"true", "1", "yes", "y"})
            subset_df = subset_df.loc[values]
        allowed = set(subset_df[subset_key].astype(str))
        selected = selected.loc[selected[subset_key].astype(str).isin(allowed)]
    return selected


def _modal_identity(values: np.ndarray) -> str:
    nonempty = values[values != ""]
    if nonempty.size == 0:
        return ""
    unique, counts = np.unique(nonempty, return_counts=True)
    return str(unique[np.argmax(counts)])


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
    metadata = pd.read_csv(merged_dir / "nav15_alignment_metadata.csv")
    selected = _subset_metadata(metadata, subset_query, subset_manifest, subset_column, subset_key)
    if selected.empty:
        raise ValueError("Selected subset contains no successful aligned models")
    coords = np.load(merged_dir / "nav15_aligned_ca_coordinates.npy", mmap_mode="r")
    present = np.load(merged_dir / "nav15_aligned_ca_present.npy", mmap_mode="r")
    identities = np.load(merged_dir / "nav15_aligned_residue_identities.npy", mmap_mode="r")
    raw_numbers = np.load(merged_dir / "nav15_raw_residue_numbers.npy")
    refs = np.load(references_path, allow_pickle=False)
    reference_ids = refs["reference_ids"].astype(str).tolist()
    annotations = pd.read_csv(annotations_path)
    minimum_coverage = float(
        minimum_residue_coverage
        if minimum_residue_coverage is not None
        else cfg.get("analysis", {}).get("minimum_residue_coverage", 0.80)
    )

    selected_indices = selected["manifest_index"].astype(int).to_numpy()
    selected_manifest_path = output_dir / f"nav15_{subset_name}_selected_models.csv"
    selected.to_csv(selected_manifest_path, index=False)
    profile_rows = []
    for group_keys, group in selected.groupby(["dataset", "sequence_condition", "protocol"], sort=True):
        dataset, sequence_condition, protocol = group_keys
        indexes = group["manifest_index"].astype(int).to_numpy()
        group_coords = np.asarray(coords[indexes], dtype=np.float64)
        group_present = np.asarray(present[indexes], dtype=bool)
        group_identities = np.asarray(identities[indexes])
        n_models = len(indexes)
        counts = group_present.sum(axis=0)
        coverage = counts / n_models
        sums = np.where(group_present[..., None], group_coords, 0.0).sum(axis=0)
        means = np.full((len(raw_numbers), 3), np.nan, dtype=np.float64)
        valid_count = counts > 0
        means[valid_count] = sums[valid_count] / counts[valid_count, None]
        squared = np.where(group_present, np.sum((group_coords - means[None, :, :]) ** 2, axis=2), 0.0).sum(axis=0)
        rmsf = np.full(len(raw_numbers), np.nan)
        rmsf[valid_count] = np.sqrt(squared[valid_count] / counts[valid_count])
        rmsf[coverage < minimum_coverage] = np.nan
        modal_identity = np.array([_modal_identity(group_identities[:, i]) for i in range(len(raw_numbers))], dtype=object)

        reference_metrics: dict[str, dict[str, np.ndarray]] = {}
        for ref_index, ref_id in enumerate(reference_ids):
            ref_coords = refs["coords"][ref_index].astype(np.float64)
            ref_present = refs["present"][ref_index].astype(bool)
            mean_distance = np.full(len(raw_numbers), np.nan)
            comparable = valid_count & ref_present
            mean_distance[comparable] = np.linalg.norm(means[comparable] - ref_coords[comparable], axis=1)
            model_ref_valid = group_present & ref_present[None, :]
            model_ref_counts = model_ref_valid.sum(axis=0)
            model_ref_sq = np.where(
                model_ref_valid,
                np.sum((group_coords - ref_coords[None, :, :]) ** 2, axis=2),
                0.0,
            ).sum(axis=0)
            rms_deviation = np.full(len(raw_numbers), np.nan)
            has = model_ref_counts > 0
            rms_deviation[has] = np.sqrt(model_ref_sq[has] / model_ref_counts[has])
            rms_deviation[(model_ref_counts / n_models) < minimum_coverage] = np.nan
            reference_metrics[ref_id] = {
                "mean_distance": mean_distance,
                "rms_deviation": rms_deviation,
                "count": model_ref_counts,
                "coverage": model_ref_counts / n_models,
            }

        for axis_index, raw in enumerate(raw_numbers.tolist()):
            row = {
                "subset": subset_name,
                "dataset": dataset,
                "sequence_condition": sequence_condition,
                "protocol": protocol,
                "raw_residue_number": raw,
                "residue_identity": modal_identity[axis_index],
                "number_of_models": n_models,
                "number_with_residue_resolved": int(counts[axis_index]),
                "coverage_fraction": float(coverage[axis_index]),
                "ensemble_mean_x_A": means[axis_index, 0],
                "ensemble_mean_y_A": means[axis_index, 1],
                "ensemble_mean_z_A": means[axis_index, 2],
                "ensemble_rmsf_A": rmsf[axis_index],
            }
            for ref_id in reference_ids:
                metrics = reference_metrics[ref_id]
                row[f"mean_distance_to_{ref_id}_A"] = metrics["mean_distance"][axis_index]
                row[f"rms_deviation_to_{ref_id}_A"] = metrics["rms_deviation"][axis_index]
                row[f"number_comparable_to_{ref_id}"] = int(metrics["count"][axis_index])
                row[f"coverage_comparable_to_{ref_id}"] = float(metrics["coverage"][axis_index])
            profile_rows.append(row)

    profiles = pd.DataFrame(profile_rows).merge(annotations, on="raw_residue_number", how="left")
    profiles_path = output_dir / f"nav15_{subset_name}_per_residue_profiles.csv"
    profiles.to_csv(profiles_path, index=False)

    comparison_rows = []
    metric_columns = ["ensemble_rmsf_A"] + [
        metric
        for ref_id in reference_ids
        for metric in (f"mean_distance_to_{ref_id}_A", f"rms_deviation_to_{ref_id}_A")
    ]
    for condition, condition_df in profiles.groupby("sequence_condition", sort=True):
        vanilla = condition_df.loc[condition_df["protocol"] == "vanilla"]
        if vanilla.empty:
            continue
        vanilla = vanilla.drop_duplicates("raw_residue_number").set_index("raw_residue_number")
        for protocol, protocol_df in condition_df.groupby("protocol", sort=True):
            if protocol == "vanilla":
                continue
            protocol_df = protocol_df.drop_duplicates("raw_residue_number").set_index("raw_residue_number")
            common = vanilla.index.intersection(protocol_df.index)
            for raw in common:
                row = {
                    "subset": subset_name,
                    "sequence_condition": condition,
                    "comparison_protocol": protocol,
                    "baseline_protocol": "vanilla",
                    "raw_residue_number": int(raw),
                }
                for metric in metric_columns:
                    baseline = vanilla.at[raw, metric]
                    comparison = protocol_df.at[raw, metric]
                    row[f"{metric}_vanilla"] = baseline
                    row[f"{metric}_{protocol}"] = comparison
                    row[f"delta_{metric}"] = comparison - baseline if pd.notna(comparison) and pd.notna(baseline) else math.nan
                    row[f"ratio_{metric}"] = comparison / baseline if pd.notna(comparison) and pd.notna(baseline) and baseline != 0 else math.nan
                comparison_rows.append(row)
    comparisons = pd.DataFrame(comparison_rows)
    if not comparisons.empty:
        comparisons = comparisons.merge(annotations, on="raw_residue_number", how="left")
    comparisons_path = output_dir / f"nav15_{subset_name}_protocol_vs_vanilla.csv"
    comparisons.to_csv(comparisons_path, index=False)

    summary_rows = []
    for group_keys, group in profiles.groupby(["dataset", "sequence_condition", "protocol"], sort=True):
        dataset, condition, protocol = group_keys
        for region_name, region_mask in {
            "whole_raw_sequence": np.ones(len(group), dtype=bool),
            "alignment_core": group["alignment_core"].fillna(False).to_numpy(dtype=bool),
            "IFM_motif": group["ifm_motif"].fillna(False).to_numpy(dtype=bool),
        }.items():
            region = group.loc[region_mask]
            values = region["ensemble_rmsf_A"].dropna()
            summary_rows.append(
                {
                    "subset": subset_name,
                    "dataset": dataset,
                    "sequence_condition": condition,
                    "protocol": protocol,
                    "region": region_name,
                    "number_of_residues": len(region),
                    "number_with_reportable_rmsf": len(values),
                    "mean_rmsf_A": values.mean() if len(values) else math.nan,
                    "median_rmsf_A": values.median() if len(values) else math.nan,
                    "maximum_rmsf_A": values.max() if len(values) else math.nan,
                    "integrated_rmsf_A": values.sum() if len(values) else math.nan,
                    "mean_coverage": region["coverage_fraction"].mean() if len(region) else math.nan,
                }
            )
    summary_path = output_dir / f"nav15_{subset_name}_group_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    per_model_columns = [
        column for column in selected.columns
        if column.startswith("whole_matched_ca_rmsd_to_") or column.startswith("core_ca_rmsd_to_")
    ]
    per_model_path = output_dir / f"nav15_{subset_name}_per_model_reference_rmsd.csv"
    selected[[
        "manifest_index", "dataset", "sequence_condition", "protocol", "pdb_file", "trajectory_id",
        "recycle_label", "recycle_index", "is_final_model", "rank", "model_number", "seed",
        "core_ca_rmsd_to_anchor_A", "matched_core_ca", "core_coverage",
        *[column for column in per_model_columns if column not in {"core_ca_rmsd_to_anchor_A"}],
    ]].to_csv(per_model_path, index=False)

    return {
        "profiles": profiles_path,
        "comparisons": comparisons_path,
        "summary": summary_path,
        "per_model_reference_rmsd": per_model_path,
        "selected_models": selected_manifest_path,
    }
