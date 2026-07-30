#!/usr/bin/env python3
"""Recompute final-QC ensemble RMSF profiles from local aligned-coordinate arrays.

The aligned arrays are memory mapped.  Coordinates are streamed in bounded
chunks and accumulated in float64, so recomputation does not duplicate the full
selected ensemble in memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CHANNEL_CONFIG = {
    "kv21": {
        "profile": "kv21_all_ok_3_symmetry_averaged_profiles.csv",
        "chain_profile": "kv21_all_ok_3_chain_resolved_profiles.csv",
        "legacy": "kv21_all_models_symmetry_averaged_profiles.csv",
        "legacy_chain": "kv21_all_models_chain_resolved_profiles.csv",
        "selection": "all_ok_3_structural_interface_qc",
    },
    "nav15": {
        "profile": "nav15_all_ok_3_per_residue_profiles.csv",
        "legacy": "nav15_all_models_per_residue_profiles.csv",
        "selection": "all_ok_3",
    },
    "cav12": {
        "profile": "cav12_all_ok_3_per_residue_profiles.csv",
        "legacy": "cav12_all_models_per_residue_profiles.csv",
        "selection": "all_ok_3",
    },
}


def _basename(series: pd.Series) -> pd.Series:
    return series.astype(str).map(lambda value: Path(value).name)


def load_inputs(repo: Path, channel: str):
    root = repo / channel / "dataRMSF"
    merged = root / "merged"
    metadata = pd.read_csv(merged / f"{channel}_alignment_metadata.csv")
    if len(metadata) != metadata.manifest_index.nunique():
        raise ValueError(f"{channel}: manifest_index is not unique")
    expected = np.arange(len(metadata))
    if not np.array_equal(metadata.manifest_index.to_numpy(), expected):
        raise ValueError(f"{channel}: metadata row order does not equal manifest_index order")
    metadata["pdb_basename_join"] = _basename(metadata["pdb_file"])
    metadata["dataset_join"] = metadata.dataset.astype(str).str.lower()
    if metadata.duplicated(["dataset_join", "pdb_basename_join"]).any():
        examples = metadata.loc[
            metadata.duplicated(["dataset_join", "pdb_basename_join"], False),
            ["dataset", "pdb_basename_join"],
        ].head().values.tolist()
        raise ValueError(f"{channel}: non-unique dataset/basename keys: {examples}")
    coordinates = np.load(
        merged / f"{channel}_aligned_ca_coordinates.npy", mmap_mode="r"
    )
    present = np.load(merged / f"{channel}_aligned_ca_present.npy", mmap_mode="r")
    if coordinates.shape[:-1] != present.shape or coordinates.shape[0] != len(metadata):
        raise ValueError(
            f"{channel}: coordinate/presence/metadata shape mismatch: "
            f"{coordinates.shape}, {present.shape}, {len(metadata)}"
        )
    references = np.load(
        root / "references" / f"{channel}_aligned_references.npz", allow_pickle=True
    )
    return root, metadata, coordinates, present, references


def final_selection(repo: Path, channel: str, metadata: pd.DataFrame) -> pd.DataFrame:
    config = CHANNEL_CONFIG[channel]
    if channel == "kv21":
        paths = sorted(
            (repo / "kv21" / "dataDistances").glob(
                "*_all_ok_rmsd_3A_structural_interface_qc.csv"
            )
        )
        if len(paths) != 6:
            raise FileNotFoundError(
                f"Kv2.1 requires six structural/interface-QC allowlists; found {len(paths)}"
            )
        selected_parts = []
        for path in paths:
            values = pd.read_csv(path, usecols=["pdb_file"])
            values["pdb_basename_join"] = _basename(values.pdb_file)
            if values.pdb_basename_join.duplicated().any():
                raise ValueError(f"Duplicate PDB basename in {path}")
            mapped = values.merge(
                metadata[
                    [
                        "manifest_index", "pdb_basename_join", "dataset",
                        "sequence_condition", "protocol",
                    ]
                ],
                on="pdb_basename_join",
                how="left",
                validate="one_to_one",
            )
            if mapped.manifest_index.isna().any():
                missing = mapped.loc[mapped.manifest_index.isna(), "pdb_basename_join"].head()
                raise KeyError(f"{path.name}: models absent from aligned arrays: {missing.tolist()}")
            selected_parts.append(mapped)
        selected = pd.concat(selected_parts, ignore_index=True)
        if selected.pdb_basename_join.duplicated().any():
            raise ValueError("Kv2.1 structural/interface allowlists overlap")
        selected["selection_source"] = config["selection"]
    else:
        manifest_path = (
            repo / channel / "dataRMSF" / "qc"
            / f"{channel}_all_ok3_selection_manifest.csv"
        )
        manifest = pd.read_csv(manifest_path)
        manifest["pdb_basename_join"] = _basename(manifest.pdb_file)
        manifest["dataset_join"] = manifest.dataset.astype(str).str.lower()
        if manifest.duplicated(["dataset_join", "pdb_basename_join"]).any():
            raise ValueError(f"{channel}: duplicate dataset/basename in allOK3 manifest")
        selected = manifest.loc[manifest.all_ok_3.astype(bool)].merge(
            metadata[
                [
                    "manifest_index", "pdb_basename_join", "dataset_join", "dataset",
                    "sequence_condition", "protocol",
                ]
            ],
            on=["dataset_join", "pdb_basename_join"],
            how="left",
            suffixes=("_qc", ""),
            validate="one_to_one",
        )
        if selected.manifest_index.isna().any():
            missing = selected.loc[selected.manifest_index.isna(), "pdb_basename_join"].head()
            raise KeyError(f"{channel}: allOK3 models absent from arrays: {missing.tolist()}")
        selected["selection_source"] = config["selection"]
    selected["manifest_index"] = selected.manifest_index.astype(int)
    expected_datasets = set(metadata.dataset.unique())
    observed_datasets = set(selected.dataset.unique())
    if expected_datasets != observed_datasets:
        raise ValueError(
            f"{channel}: final selection dataset mismatch; "
            f"missing={sorted(expected_datasets-observed_datasets)}, "
            f"extra={sorted(observed_datasets-expected_datasets)}"
        )
    return selected.sort_values("manifest_index").reset_index(drop=True)


def streamed_moments(
    coordinates: np.ndarray,
    present: np.ndarray,
    selection: np.ndarray,
    chunk_size: int,
):
    """Return counts, coordinate sums, and squared-norm sums for selected rows."""
    if selection.dtype != bool or selection.shape != (coordinates.shape[0],):
        raise ValueError("Selection must be a boolean vector aligned to coordinate rows")
    spatial_shape = coordinates.shape[1:-1]
    counts = np.zeros(spatial_shape, dtype=np.int64)
    sums = np.zeros(spatial_shape + (3,), dtype=np.float64)
    sum_sq = np.zeros(spatial_shape, dtype=np.float64)
    for start in range(0, coordinates.shape[0], chunk_size):
        stop = min(start + chunk_size, coordinates.shape[0])
        local = selection[start:stop]
        if not local.any():
            continue
        xyz = np.asarray(coordinates[start:stop][local], dtype=np.float64)
        valid = np.asarray(present[start:stop][local], dtype=bool)
        finite = np.isfinite(xyz).all(axis=-1)
        valid &= finite
        safe = np.where(valid[..., None], xyz, 0.0)
        counts += valid.sum(axis=0, dtype=np.int64)
        sums += safe.sum(axis=0, dtype=np.float64)
        sum_sq += np.einsum("...j,...j->...", safe, safe).sum(axis=0, dtype=np.float64)
    mean = np.full_like(sums, np.nan)
    np.divide(sums, counts[..., None], out=mean, where=counts[..., None] > 0)
    mean_sq_norm = np.full(sum_sq.shape, np.nan, dtype=np.float64)
    np.divide(sum_sq, counts, out=mean_sq_norm, where=counts > 0)
    variance = mean_sq_norm - np.einsum("...j,...j->...", mean, mean)
    variance = np.maximum(variance, 0.0)
    rmsf = np.sqrt(variance)
    return counts, mean, rmsf, mean_sq_norm


def add_reference_metrics(
    frame: pd.DataFrame,
    mean: np.ndarray,
    mean_sq_norm: np.ndarray,
    counts: np.ndarray,
    references,
    channel: str,
    chain_index: int | None = None,
):
    ids = references["reference_ids"].astype(str)
    ref_coords = references["coords"]
    ref_present = references["present"]
    for ref_index, ref_id in enumerate(ids):
        if channel == "kv21":
            ref = np.asarray(ref_coords[ref_index, chain_index], dtype=np.float64)
            available = np.asarray(ref_present[ref_index, chain_index], dtype=bool)
            mean_name = f"mean_coordinate_distance_to_{ref_id}_A"
        else:
            ref = np.asarray(ref_coords[ref_index], dtype=np.float64)
            available = np.asarray(ref_present[ref_index], dtype=bool)
            mean_name = f"mean_distance_to_{ref_id}_A"
        usable = available & (counts > 0) & np.isfinite(mean).all(axis=-1)
        mean_distance = np.full(counts.shape, np.nan)
        mean_distance[usable] = np.linalg.norm(mean[usable] - ref[usable], axis=-1)
        rms_sq = (
            mean_sq_norm - 2.0 * np.einsum("...j,...j->...", mean, ref)
            + np.einsum("...j,...j->...", ref, ref)
        )
        rms_dev = np.full(counts.shape, np.nan)
        rms_dev[usable] = np.sqrt(np.maximum(rms_sq[usable], 0.0))
        frame[mean_name] = mean_distance
        frame[f"rms_deviation_to_{ref_id}_A"] = rms_dev
        if channel != "kv21":
            frame[f"number_comparable_to_{ref_id}"] = np.where(available, counts, 0)
            frame[f"coverage_comparable_to_{ref_id}"] = np.where(
                available, counts / frame.number_of_models.iloc[0], 0.0
            )
    return frame


def recompute_single_chain_channel(
    channel: str,
    legacy: pd.DataFrame,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    present: np.ndarray,
    references,
    selected: pd.DataFrame,
    chunk_size: int,
) -> pd.DataFrame:
    outputs = []
    for dataset, group in selected.groupby("dataset", sort=True):
        indices = group.manifest_index.to_numpy()
        mask = np.zeros(len(metadata), dtype=bool)
        mask[indices] = True
        counts, mean, rmsf, mean_sq = streamed_moments(
            coordinates, present, mask, chunk_size
        )
        template = legacy.loc[legacy.dataset.eq(dataset)].copy()
        if len(template) != coordinates.shape[1]:
            raise ValueError(
                f"{channel}/{dataset}: legacy template has {len(template)} residues, "
                f"expected {coordinates.shape[1]}"
            )
        template = template.sort_values("raw_residue_number").reset_index(drop=True)
        template["subset"] = "all_ok_3"
        template["number_of_models"] = len(indices)
        template["number_with_residue_resolved"] = counts
        template["coverage_fraction"] = counts / len(indices)
        template["ensemble_mean_x_A"] = mean[:, 0]
        template["ensemble_mean_y_A"] = mean[:, 1]
        template["ensemble_mean_z_A"] = mean[:, 2]
        template["ensemble_rmsf_A"] = rmsf
        template = add_reference_metrics(
            template, mean, mean_sq, counts, references, channel
        )
        outputs.append(template)
    return pd.concat(outputs, ignore_index=True)


def recompute_kv21(
    legacy_chain: pd.DataFrame,
    legacy_symmetry: pd.DataFrame,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    present: np.ndarray,
    references,
    selected: pd.DataFrame,
    chunk_size: int,
):
    chains = references["canonical_chains"].astype(str).tolist()
    chain_outputs = []
    for dataset, group in selected.groupby("dataset", sort=True):
        indices = group.manifest_index.to_numpy()
        mask = np.zeros(len(metadata), dtype=bool)
        mask[indices] = True
        counts, mean, rmsf, mean_sq = streamed_moments(
            coordinates, present, mask, chunk_size
        )
        for chain_index, chain in enumerate(chains):
            template = legacy_chain.loc[
                legacy_chain.dataset.eq(dataset) & legacy_chain.chain.eq(chain)
            ].copy()
            template = template.sort_values("raw_residue_number").reset_index(drop=True)
            if len(template) != coordinates.shape[2]:
                raise ValueError(f"kv21/{dataset}/{chain}: invalid template length")
            template["subset"] = "all_ok_3"
            template["number_of_models"] = len(indices)
            template["number_with_residue_resolved"] = counts[chain_index]
            template["coverage_fraction"] = counts[chain_index] / len(indices)
            if "passes_coverage_threshold" in template:
                template["passes_coverage_threshold"] = (
                    template.coverage_fraction >= template.coverage_threshold
                )
            template["ensemble_mean_x_A"] = mean[chain_index, :, 0]
            template["ensemble_mean_y_A"] = mean[chain_index, :, 1]
            template["ensemble_mean_z_A"] = mean[chain_index, :, 2]
            template["ensemble_rmsf_A"] = rmsf[chain_index]
            template = add_reference_metrics(
                template, mean[chain_index], mean_sq[chain_index],
                counts[chain_index], references, "kv21", chain_index
            )
            chain_outputs.append(template)
    chain_frame = pd.concat(chain_outputs, ignore_index=True)

    symmetry_outputs = []
    reference_ids = references["reference_ids"].astype(str).tolist()
    for dataset, group in chain_frame.groupby("dataset", sort=True):
        template = legacy_symmetry.loc[legacy_symmetry.dataset.eq(dataset)].copy()
        template = template.sort_values("raw_residue_number").reset_index(drop=True)
        pivot_rmsf = group.pivot(
            index="raw_residue_number", columns="chain", values="ensemble_rmsf_A"
        ).reindex(template.raw_residue_number)
        pivot_cov = group.pivot(
            index="raw_residue_number", columns="chain", values="coverage_fraction"
        ).reindex(template.raw_residue_number)
        template["subset"] = "all_ok_3"
        template["number_of_models"] = int(group.number_of_models.iloc[0])
        template["chains_with_valid_rmsf"] = pivot_rmsf.notna().sum(axis=1).to_numpy()
        template["symmetry_averaged_rmsf_A"] = pivot_rmsf.mean(axis=1).to_numpy()
        template["chain_to_chain_rmsf_std_A"] = pivot_rmsf.std(axis=1, ddof=0).to_numpy()
        template["chain_min_rmsf_A"] = pivot_rmsf.min(axis=1).to_numpy()
        template["chain_max_rmsf_A"] = pivot_rmsf.max(axis=1).to_numpy()
        template["mean_chain_coverage_fraction"] = pivot_cov.mean(axis=1).to_numpy()
        template["minimum_chain_coverage_fraction"] = pivot_cov.min(axis=1).to_numpy()
        for ref_id in reference_ids:
            for source, target_mean, target_std in (
                (
                    f"mean_coordinate_distance_to_{ref_id}_A",
                    f"symmetry_averaged_mean_coordinate_distance_to_{ref_id}_A",
                    f"chain_std_mean_coordinate_distance_to_{ref_id}_A",
                ),
                (
                    f"rms_deviation_to_{ref_id}_A",
                    f"symmetry_averaged_rms_deviation_to_{ref_id}_A",
                    f"chain_std_rms_deviation_to_{ref_id}_A",
                ),
            ):
                pivot = group.pivot(
                    index="raw_residue_number", columns="chain", values=source
                ).reindex(template.raw_residue_number)
                template[target_mean] = pivot.mean(axis=1).to_numpy()
                template[target_std] = pivot.std(axis=1, ddof=0).to_numpy()
        symmetry_outputs.append(template)
    return chain_frame, pd.concat(symmetry_outputs, ignore_index=True)


def validate_against_legacy(
    channel: str,
    legacy: pd.DataFrame,
    metadata: pd.DataFrame,
    coordinates: np.ndarray,
    present: np.ndarray,
    chunk_size: int,
    tolerance: float,
):
    """Validate the streamed RMSF formula on the first complete legacy dataset."""
    dataset = sorted(metadata.dataset.unique())[0]
    indices = metadata.index[metadata.dataset.eq(dataset)].to_numpy()
    mask = np.zeros(len(metadata), dtype=bool)
    mask[indices] = True
    _, _, rmsf, _ = streamed_moments(coordinates, present, mask, chunk_size)
    if channel == "kv21":
        observed = np.nanmean(rmsf, axis=0)
        expected = (
            legacy.loc[legacy.dataset.eq(dataset)]
            .sort_values("raw_residue_number")
            .symmetry_averaged_rmsf_A.to_numpy()
        )
    else:
        observed = rmsf
        expected = (
            legacy.loc[legacy.dataset.eq(dataset)]
            .sort_values("raw_residue_number")
            .ensemble_rmsf_A.to_numpy()
        )
    valid = np.isfinite(observed) & np.isfinite(expected)
    maximum = float(np.max(np.abs(observed[valid] - expected[valid])))
    median = float(np.median(np.abs(observed[valid] - expected[valid])))
    if maximum > tolerance:
        raise AssertionError(
            f"{channel}: streamed RMSF validation failed; max |delta|={maximum:.6g} Å"
        )
    return {"dataset": dataset, "max_abs_delta_A": maximum, "median_abs_delta_A": median}


def run_channel(repo: Path, channel: str, chunk_size: int, tolerance: float):
    config = CHANNEL_CONFIG[channel]
    root, metadata, coordinates, present, references = load_inputs(repo, channel)
    profiles = root / "profiles"
    legacy = pd.read_csv(profiles / config["legacy"])
    validation = validate_against_legacy(
        channel, legacy, metadata, coordinates, present, chunk_size, tolerance
    )
    selected = final_selection(repo, channel, metadata)
    selected.to_csv(profiles / f"{channel}_all_ok_3_selected_models.csv", index=False)
    if channel == "kv21":
        legacy_chain = pd.read_csv(profiles / config["legacy_chain"])
        chain_frame, profile = recompute_kv21(
            legacy_chain, legacy, metadata, coordinates, present, references,
            selected, chunk_size
        )
        chain_frame.to_csv(profiles / config["chain_profile"], index=False)
    else:
        profile = recompute_single_chain_channel(
            channel, legacy, metadata, coordinates, present, references,
            selected, chunk_size
        )
    output = profiles / config["profile"]
    profile.to_csv(output, index=False)
    summary = {
        "channel": channel,
        "selection": config["selection"],
        "aligned_models": len(metadata),
        "selected_models": len(selected),
        "output": str(output.relative_to(repo)),
        "validation": validation,
        "selected_by_dataset": {
            str(key): int(value)
            for key, value in selected.groupby("dataset").size().items()
        },
    }
    with (profiles / f"{channel}_all_ok_3_recompute_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--channel", choices=("kv21", "nav15", "cav12", "all"), default="all"
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--validation-tolerance", type=float, default=2e-4)
    args = parser.parse_args()
    channels = CHANNEL_CONFIG if args.channel == "all" else (args.channel,)
    for channel in channels:
        run_channel(args.repo_root.resolve(), channel, args.chunk_size, args.validation_tolerance)


if __name__ == "__main__":
    main()
