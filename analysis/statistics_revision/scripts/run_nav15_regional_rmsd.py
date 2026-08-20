#!/usr/bin/env python3
"""Regenerate traceable NaV1.5 regional C-alpha RMSD statistics.

The historical 3.88 GB regional RMSD table is unavailable because its gzip
payload is itself a Git-LFS pointer to an object that was never uploaded.  This
script deliberately does not tune selections to the persisted historical
medians.  It rebuilds a compact per-structure table from the authoritative
aligned-coordinate arrays, final-QC manifest, and explicit repository-native
region definitions.

Both models and experimental references are in the common 6UZ3 stable-core
alignment frame.  Primary inference first takes the median retained recycle in
each seed-model trajectory, gives available AF2 model parameterizations equal
mass within a seed, and bootstraps complete seeds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ensemble_rmsf_analysis.io import (  # noqa: E402
    read_csv_resolving_lfs,
    resolve_local_lfs_object,
)
from scripts.ensemble_rmsf_analysis.topology import TOPOLOGY  # noqa: E402
from shared.nav15_presentation import POCKET_RECEPTORS  # noqa: E402
from shared.plotting import NAV15_EXPERIMENTAL_STYLES  # noqa: E402
from shared.seed_block_statistics import (  # noqa: E402
    leave_one_model_out,
    make_seed_blocks,
    reduce_trajectory_values,
    seed_block_bootstrap,
    seed_distribution_metrics,
)


BASE_SEED = 20260820
DEFAULT_OUTPUT = ROOT / "analysis/statistics_revision/seed_block/nav15_regional_rmsd"
COORDINATE_PATH = ROOT / "nav15/dataRMSF/merged/nav15_aligned_ca_coordinates.npy"
PRESENT_PATH = ROOT / "nav15/dataRMSF/merged/nav15_aligned_ca_present.npy"
METADATA_PATH = ROOT / "nav15/dataRMSF/merged/nav15_alignment_metadata.csv"
REFERENCE_PATH = ROOT / "nav15/dataRMSF/references/nav15_aligned_references.npz"
SELECTION_PATH = ROOT / "nav15/dataRMSF/qc/nav15_all_ok3_selection_manifest.csv"


def _ranges_for_labels(*labels: str) -> tuple[tuple[int, int], ...]:
    wanted = set(labels)
    ranges = [
        (int(item["start"]), int(item["end"]))
        for item in TOPOLOGY["nav15"]
        if str(item["label"]) in wanted
    ]
    if len(ranges) != len(wanted):
        found = {
            str(item["label"])
            for item in TOPOLOGY["nav15"]
            if str(item["label"]) in wanted
        }
        raise ValueError(f"Missing NaV1.5 topology labels: {sorted(wanted - found)}")
    return tuple(ranges)


def region_definitions() -> dict[str, dict[str, object]]:
    """Return the four fixed regions used by the regenerated supplement."""
    pore_labels = tuple(
        f"D{domain} {helix}"
        for domain in ("I", "II", "III", "IV")
        for helix in ("S5", "S6")
    )
    pocket_residues = tuple(sorted(int(value[-4:]) for value in POCKET_RECEPTORS.values()))
    return {
        "pore_s5_s6_helices": {
            "label": "Pore helices (S5/S6)",
            "ranges": _ranges_for_labels(*pore_labels),
            "residues": (),
            "provenance": "scripts/ensemble_rmsf_analysis/topology.py:TOPOLOGY['nav15']",
            "rationale": "union of the reviewed/mapped S5 and S6 helix boundaries in all four domains",
        },
        "dii_s6": {
            "label": "DII S6",
            "ranges": _ranges_for_labels("DII S6"),
            "residues": (),
            "provenance": "scripts/ensemble_rmsf_analysis/topology.py:TOPOLOGY['nav15']",
            "rationale": "reviewed/mapped DII-S6 helix boundary",
        },
        "ifm_motif": {
            "label": "IFM/QQQ motif",
            "ranges": _ranges_for_labels("IFM"),
            "residues": (),
            "provenance": "scripts/ensemble_rmsf_analysis/topology.py:TOPOLOGY['nav15']",
            "rationale": "sequence-mapped IFM motif; the same positions contain QQQ in the mutant",
        },
        "ifm_receptor_set": {
            "label": "IFM receptor set",
            "ranges": (),
            "residues": pocket_residues,
            "provenance": "shared/nav15_presentation.py:POCKET_RECEPTORS",
            "rationale": "six prespecified receptor residues used by the independent IFM-pocket analysis",
        },
    }


def residues_for_region(definition: dict[str, object]) -> np.ndarray:
    residues: set[int] = set(int(value) for value in definition.get("residues", ()))
    for start, end in definition.get("ranges", ()):  # type: ignore[assignment]
        residues.update(range(int(start), int(end) + 1))
    return np.asarray(sorted(residues), dtype=np.int32)


def region_definition_table(raw_numbers: np.ndarray) -> pd.DataFrame:
    rows = []
    for region_id, definition in region_definitions().items():
        residues = residues_for_region(definition)
        if not np.isin(residues, raw_numbers).all():
            raise ValueError(f"{region_id} contains residues outside the aligned coordinate axis")
        ranges = definition["ranges"]
        rows.append({
            "region_id": region_id,
            "region_label": definition["label"],
            "raw_model_ranges": ";".join(f"{a}-{b}" for a, b in ranges),
            "raw_model_residues": ";".join(map(str, definition["residues"])),
            "display_sequence_ranges_or_residues": ";".join(
                [*(f"{a + 316}-{b + 316}" for a, b in ranges),
                 *(str(int(value) + 316) for value in definition["residues"])]
            ),
            "number_of_requested_CA": len(residues),
            "selection_provenance": definition["provenance"],
            "selection_rationale": definition["rationale"],
            "selection_frozen_before_regenerated_RMSD_calculation": True,
        })
    return pd.DataFrame(rows)


def regional_rmsd(
    model_coordinates: np.ndarray,
    model_present: np.ndarray,
    reference_coordinates: np.ndarray,
    reference_present: np.ndarray,
    region_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """C-alpha RMSD in an existing common alignment frame."""
    requested = np.asarray(region_mask, bool) & np.asarray(reference_present, bool)
    reference_count = int(requested.sum())
    valid = np.asarray(model_present, bool) & requested[None, :]
    counts = valid.sum(axis=1).astype(np.int32)
    delta_sq = np.sum(
        np.square(np.asarray(model_coordinates, dtype=np.float64)
                  - np.asarray(reference_coordinates, dtype=np.float64)[None, :, :]),
        axis=2,
    )
    numerator = np.sum(np.where(valid, delta_sq, 0.0), axis=1)
    rmsd = np.sqrt(np.divide(
        numerator,
        counts,
        out=np.full(len(counts), np.nan, dtype=float),
        where=counts > 0,
    ))
    coverage = np.divide(
        counts,
        reference_count,
        out=np.full(len(counts), np.nan, dtype=float),
        where=reference_count > 0,
    )
    return rmsd, counts, coverage, reference_count


def _pointer_oid(path: Path) -> str | None:
    prefix = path.read_text(errors="ignore")[:256]
    for line in prefix.splitlines():
        if line.startswith("oid sha256:"):
            return line.split(":", 1)[1]
    return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(path: Path) -> dict[str, object]:
    resolved = resolve_local_lfs_object(path, ROOT)
    return {
        "working_tree_path": str(path.relative_to(ROOT)),
        "resolved_local_path": str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(resolved),
        "git_lfs_oid": _pointer_oid(path),
        "resolved_size_bytes": resolved.stat().st_size,
        "sha256": _pointer_oid(path) or _sha256(resolved),
    }


def selected_metadata() -> pd.DataFrame:
    metadata = read_csv_resolving_lfs(METADATA_PATH, ROOT)
    selection = read_csv_resolving_lfs(
        SELECTION_PATH, ROOT, usecols=["pdb_basename", "dataset", "all_ok_3"]
    )
    selected_pairs = set(
        selection.loc[selection["all_ok_3"].fillna(False), ["dataset", "pdb_basename"]]
        .assign(dataset=lambda frame: frame["dataset"].str.lower())
        .itertuples(index=False, name=None)
    )
    basename = metadata["pdb_file"].map(lambda value: Path(str(value)).name)
    metadata_pairs = zip(metadata["dataset"].str.lower(), basename)
    selected_mask = np.fromiter(
        (pair in selected_pairs for pair in metadata_pairs), dtype=bool, count=len(metadata)
    )
    selected = metadata.loc[selected_mask].copy()
    selected["pdb_basename"] = basename.loc[selected.index]
    selected = selected.sort_values("manifest_index").reset_index(drop=True)
    if selected.duplicated(["dataset", "pdb_basename"]).any():
        raise ValueError("Aligned metadata contains duplicated selected dataset/PDB keys")
    if len(selected) != 34_998:
        raise ValueError(f"Expected 34,998 final-QC structures, observed {len(selected):,}")
    return selected


def write_per_structure_table(
    output: Path,
    *,
    chunk_size: int,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    coordinates = np.load(resolve_local_lfs_object(COORDINATE_PATH, ROOT), mmap_mode="r")
    present = np.load(resolve_local_lfs_object(PRESENT_PATH, ROOT), mmap_mode="r")
    references = np.load(REFERENCE_PATH)
    metadata = selected_metadata()
    raw_numbers = references["raw_residue_numbers"]
    definitions = region_definition_table(raw_numbers)
    definitions.to_csv(output / "nav15_regional_rmsd_region_definitions.csv", index=False)

    if coordinates.shape != (42_000, 1_572, 3) or present.shape != (42_000, 1_572):
        raise ValueError(f"Unexpected aligned-array shapes: {coordinates.shape}, {present.shape}")
    if not np.array_equal(raw_numbers, np.arange(1, 1_573)):
        raise ValueError("NaV1.5 aligned coordinate axis is not raw positions 1..1572")

    reference_ids = [str(value) for value in references["reference_ids"]]
    region_masks = {
        row.region_id: np.isin(raw_numbers, residues_for_region(region_definitions()[row.region_id]))
        for row in definitions.itertuples(index=False)
    }
    metadata_columns = [
        "manifest_index", "dataset", "sequence_condition", "protocol", "pdb_file",
        "pdb_basename", "trajectory_id", "recycle_label", "recycle_index",
        "model_number", "seed", "rank",
    ]
    table_path = output / "nav15_regional_rmsd_per_structure.csv.gz"
    validation_delta: dict[str, list[np.ndarray]] = {reference_id: [] for reference_id in reference_ids}

    with table_path.open("wb") as raw_handle:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_handle, mtime=0) as gzip_handle:
            with io.TextIOWrapper(gzip_handle, encoding="utf-8", newline="") as text_handle:
                first = True
                for start in range(0, len(metadata), chunk_size):
                    part = metadata.iloc[start:start + chunk_size]
                    indices = part["manifest_index"].to_numpy(int)
                    model_coordinates = np.asarray(coordinates[indices])
                    model_present = np.asarray(present[indices])
                    result = part[metadata_columns].reset_index(drop=True).copy()

                    for reference_index, reference_id in enumerate(reference_ids):
                        ref_coordinates = references["coords"][reference_index]
                        ref_present = references["present"][reference_index]
                        for row in definitions.itertuples(index=False):
                            rmsd, counts, coverage, reference_count = regional_rmsd(
                                model_coordinates, model_present, ref_coordinates,
                                ref_present, region_masks[row.region_id],
                            )
                            prefix = f"{reference_id}__{row.region_id}"
                            result[f"{prefix}__rmsd_A"] = rmsd
                            result[f"{prefix}__matched_CA"] = counts
                            result[f"{prefix}__coverage_fraction"] = coverage
                            result[f"{prefix}__reference_CA"] = reference_count

                        core_mask = references["core_mask"]
                        direct_core, _, _, _ = regional_rmsd(
                            model_coordinates, model_present, ref_coordinates,
                            ref_present, core_mask,
                        )
                        stored = pd.to_numeric(
                            part[f"core_ca_rmsd_to_{reference_id}_A"], errors="coerce"
                        ).to_numpy(float)
                        validation_delta[reference_id].append(np.abs(direct_core - stored))

                    result.to_csv(text_handle, index=False, header=first)
                    first = False

    validation_rows = []
    for reference_id, pieces in validation_delta.items():
        delta = np.concatenate(pieces)
        validation_rows.append({
            "validation": "common-anchor stable-core RMSD versus stored alignment metadata",
            "reference_id": reference_id,
            "n_structures": len(delta),
            "maximum_absolute_delta_A": float(np.nanmax(delta)),
            "median_absolute_delta_A": float(np.nanmedian(delta)),
            "passes_maximum_delta_le_1e-5_A": bool(np.nanmax(delta) <= 1e-5),
        })
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(output / "nav15_regional_rmsd_coordinate_validation.csv", index=False)
    return table_path, metadata, validation


def seed_block_analysis(
    table_path: Path,
    output: Path,
    definitions: pd.DataFrame,
    *,
    bootstrap: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    table = pd.read_csv(table_path)
    reference_ids = [str(value) for value in np.load(REFERENCE_PATH)["reference_ids"]]
    effects, sensitivities = [], []
    comparison_index = 0
    for sequence in ("wt", "qqq"):
        a = table[table["dataset"].eq(f"{sequence}_vanilla")]
        b = table[table["dataset"].eq(f"{sequence}_masked")]
        for reference_id in reference_ids:
            for region in definitions.itertuples(index=False):
                value_col = f"{reference_id}__{region.region_id}__rmsd_A"
                reduced_a = reduce_trajectory_values(a, value_col, reduction="median")
                reduced_b = reduce_trajectory_values(b, value_col, reduction="median")
                blocks_a = make_seed_blocks(reduced_a, value_col)
                blocks_b = make_seed_blocks(reduced_b, value_col)
                base = {
                    "sequence_condition": sequence,
                    "comparison": f"{sequence.upper()} original masked - vanilla",
                    "reference_id": reference_id,
                    "region_id": region.region_id,
                    "region_label": region.region_label,
                    "value_column": value_col,
                    "n_seeds_vanilla": len(blocks_a),
                    "n_seeds_masked": len(blocks_b),
                    "n_trajectories_vanilla": len(reduced_a),
                    "n_trajectories_masked": len(reduced_b),
                    "within_trajectory_reduction": "median retained final-QC recycle",
                    "within_seed_weighting": "equal available AF2 model parameterizations",
                    "resampling_unit": "whole seed",
                    "bootstrap_replicates": bootstrap,
                    "random_seed": BASE_SEED + comparison_index,
                    "p_value_policy": "not calculated; prespecified effect-size supplement with direct bootstrap CI",
                }
                if not blocks_a or not blocks_b:
                    effects.append({
                        **base,
                        "analysis_status": "unavailable_reference_region_no_CA",
                    })
                    comparison_index += 1
                    continue
                point = seed_distribution_metrics(blocks_a, blocks_b)
                interval = seed_block_bootstrap(
                    blocks_a, blocks_b,
                    n_bootstrap=bootstrap,
                    random_seed=BASE_SEED + comparison_index,
                )
                effects.append({
                    **base,
                    "analysis_status": "ok",
                    **point,
                    **interval,
                })
                sensitivity = leave_one_model_out(
                    a, b, value_col, within_trajectory_reduction="median"
                )
                sensitivity.insert(0, "region_label", region.region_label)
                sensitivity.insert(0, "region_id", region.region_id)
                sensitivity.insert(0, "reference_id", reference_id)
                sensitivity.insert(0, "sequence_condition", sequence)
                sensitivities.append(sensitivity)
                comparison_index += 1

    effects_frame = pd.DataFrame(effects)
    sensitivity_frame = pd.concat(sensitivities, ignore_index=True)
    effects_frame.to_csv(output / "nav15_regional_rmsd_seed_block_effects.csv", index=False)
    sensitivity_frame.to_csv(
        output / "nav15_regional_rmsd_leave_one_AF2_model_out.csv", index=False
    )
    return effects_frame, sensitivity_frame


def plot_effects(effects: pd.DataFrame, output: Path) -> None:
    reference_order = ["6UZ3", "7FBS", "8T6L", "7DTC", "8VYJ", "8VYK"]
    region_order = [
        "pore_s5_s6_helices", "dii_s6", "ifm_motif", "ifm_receptor_set"
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14.6, 7.8), sharey=True)
    for row_index, sequence in enumerate(("wt", "qqq")):
        for column_index, region_id in enumerate(region_order):
            axis = axes[row_index, column_index]
            part = effects[
                effects["sequence_condition"].eq(sequence)
                & effects["region_id"].eq(region_id)
            ].set_index("reference_id").loc[reference_order]
            y = np.arange(len(reference_order))
            for position, (reference_id, result) in enumerate(part.iterrows()):
                value = float(result["delta_weighted_median_A"])
                low = float(result["delta_weighted_median_CI_low_A"])
                high = float(result["delta_weighted_median_CI_high_A"])
                if not np.isfinite([value, low, high]).all():
                    axis.text(
                        .02, position, "no aligned C-alpha",
                        transform=axis.get_yaxis_transform(), ha="left", va="center",
                        fontsize=7.2, color="#777077", fontstyle="italic",
                    )
                    continue
                style = NAV15_EXPERIMENTAL_STYLES[reference_id]
                axis.errorbar(
                    value, position,
                    xerr=np.asarray([[value - low], [high - value]]),
                    fmt=style["marker"], markersize=6.2,
                    markerfacecolor="white", markeredgecolor=style["color"],
                    markeredgewidth=1.2, ecolor=style["color"],
                    elinewidth=1.0, capsize=2.5, zorder=3,
                )
            axis.axvline(0, color="#716A76", linestyle=":", linewidth=1.0)
            axis.set_yticks(y, reference_order)
            axis.invert_yaxis()
            axis.grid(axis="x", color="#EAE4ED", linestyle="--", linewidth=.55)
            axis.set_title(str(part["region_label"].iloc[0]), fontsize=10.5)
            if column_index == 0:
                axis.set_ylabel(f"{sequence.upper()} references", fontsize=10)
            if row_index == 1:
                axis.set_xlabel("Original masked - vanilla\nseed-balanced median RMSD (Angstrom)", fontsize=9)
            axis.spines[["top", "right"]].set_visible(False)
    figure.suptitle(
        "NaV1.5 regional C-alpha RMSD after common stable-core alignment",
        fontsize=14.5, fontweight="semibold", y=.985,
    )
    figure.text(
        .5, .012,
        "Points are seed-balanced effects after trajectory-median reduction; bars are 95% whole-seed bootstrap CIs.  Left indicates lower RMSD after masking.",
        ha="center", va="bottom", fontsize=9, color="#5F5666",
    )
    figure.tight_layout(rect=(0, .055, 1, .95))
    figure.savefig(output / "Figure_S5_Nav15_regional_RMSD_seed_block.png", dpi=300)
    figure.savefig(output / "Figure_S5_Nav15_regional_RMSD_seed_block.pdf")
    plt.close(figure)


def write_run_summary(
    output: Path,
    table_path: Path,
    metadata: pd.DataFrame,
    validation: pd.DataFrame,
    effects: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    bootstrap: int,
) -> None:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        commit = "unavailable"
    with table_path.open("rb") as handle:
        gzip_magic = handle.read(2).hex()
    report = {
        "status": "completed_traceable_nav15_regional_rmsd_regeneration",
        "historical_source_status": {
            "nested_git_lfs_oid": "e93fda439af7fb35a7cb8464485e164247b5ff64b784054df476f1349f12cbb4",
            "github_lfs_batch_query_result": "404 Object does not exist on the server",
            "policy": "historical regional medians retired; regenerated definitions and values are authoritative",
        },
        "git_commit_at_run": commit,
        "random_seed_base": BASE_SEED,
        "bootstrap_replicates": bootstrap,
        "per_structure_rows": len(metadata),
        "dataset_counts": metadata.groupby("dataset").size().astype(int).to_dict(),
        "reference_ids": sorted(effects["reference_id"].unique().tolist()),
        "regions": definitions[["region_id", "number_of_requested_CA"]].to_dict("records"),
        "effect_rows": len(effects),
        "primary_estimand": "median retained final-QC recycle per seed-model trajectory; equal AF2 models within seed; equal seeds",
        "alignment_frame": "models and references independently aligned to 6UZ3 using the fixed stable-core mask",
        "coordinate_validation_all_pass": bool(validation["passes_maximum_delta_le_1e-5_A"].all()),
        "maximum_coordinate_validation_delta_A": float(validation["maximum_absolute_delta_A"].max()),
        "compact_table": str(table_path.relative_to(ROOT)),
        "compact_table_gzip_magic_hex": gzip_magic,
        "compact_table_is_actual_gzip_not_lfs_pointer": gzip_magic == "1f8b",
        "source_files": [
            _source_record(COORDINATE_PATH), _source_record(PRESENT_PATH),
            _source_record(METADATA_PATH), _source_record(REFERENCE_PATH),
            _source_record(SELECTION_PATH),
        ],
        "limitations": [
            "the external historical region mappings were not recoverable and were not reverse-engineered from old medians",
            "the regenerated pore region is explicitly the reviewed/mapped S5/S6 helices, not an undocumented historical pore-domain selection",
            "Cav1.2/Nav1.5 masked WT-mutant comparisons remain nonfactorial because mask columns differ by condition",
        ],
    }
    (output / "nav15_regional_rmsd_run_summary.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("exploratory", "publication"), default="publication")
    parser.add_argument("--chunk-size", type=int, default=500)
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    bootstrap = 500 if args.mode == "exploratory" else 2_000

    table_path, metadata, validation = write_per_structure_table(
        output, chunk_size=args.chunk_size
    )
    definitions = pd.read_csv(output / "nav15_regional_rmsd_region_definitions.csv")
    effects, _ = seed_block_analysis(
        table_path, output, definitions, bootstrap=bootstrap
    )
    plot_effects(effects, output)
    write_run_summary(
        output, table_path, metadata, validation, effects, definitions,
        bootstrap=bootstrap,
    )
    print(f"NaV1.5 regional RMSD regeneration completed: {output}")


if __name__ == "__main__":
    main()
