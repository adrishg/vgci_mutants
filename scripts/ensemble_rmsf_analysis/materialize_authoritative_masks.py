#!/usr/bin/env python3
"""Materialize the user-supplied authoritative RMSF mask table."""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import yaml

from .masks import compact_ranges, parse_ranges


def _resolve_masks(definitions: dict) -> dict[str, set[int]]:
    resolved: dict[str, set[int]] = {}
    pending = dict(definitions)
    while pending:
        progressed = False
        for name, definition in list(pending.items()):
            base_name = definition.get("base")
            if base_name and base_name not in resolved:
                continue
            positions = set(resolved.get(base_name, set()))
            positions |= parse_ranges(definition.get("ranges"))
            positions |= parse_ranges(definition.get("add_ranges"))
            positions -= parse_ranges(definition.get("remove_ranges"))
            expected = int(definition["expected_count"])
            if len(positions) != expected:
                raise AssertionError(f"{name}: resolved {len(positions)} positions; expected {expected}")
            raw_length = int(definition["raw_length"])
            outside = {p for p in positions if p < 1 or p > raw_length}
            if outside:
                raise AssertionError(f"{name}: positions outside 1-{raw_length}: {compact_ranges(outside)}")
            resolved[name] = positions
            del pending[name]
            progressed = True
        if not progressed:
            raise ValueError(f"Unresolvable mask definitions: {sorted(pending)}")
    return resolved


def _assert_controlled_differences(masks: dict[str, set[int]]) -> None:
    checks = {
        "nav15 v2 minus noIFM": (
            masks["nav15_v2"] - masks["nav15_v2_noIFM"], parse_ranges("1164-1176")
        ),
        "nav15 standard_plus_IFM minus standard": (
            masks["nav15_standard_plus_IFM"] - masks["nav15_standard"], parse_ranges("1164-1176")
        ),
        "cav12 G402S minus WT": (
            masks["cav12_g402s"] - masks["cav12_wt_common"], parse_ranges("397-400, 402-406")
        ),
        "cav12 G406R minus WT": (
            masks["cav12_g406r"] - masks["cav12_wt_common"], parse_ranges("402-406")
        ),
        "cav12 G490R minus WT": (
            masks["cav12_g490r"] - masks["cav12_wt_common"], parse_ranges("485-495")
        ),
    }
    for label, (observed, expected) in checks.items():
        if observed != expected:
            raise AssertionError(
                f"{label}: observed {compact_ranges(observed)}; expected {compact_ranges(expected)}"
            )
    if masks["nav15_v2_noIFM"] & parse_ranges("1164-1176"):
        raise AssertionError("nav15_v2_noIFM unexpectedly masks the IFM block")
    for name, position in (("cav12_g402s", 402), ("cav12_g406r", 406), ("cav12_g490r", 490)):
        if position not in masks[name]:
            raise AssertionError(f"{name}: mutation position {position} is not directly masked")


def materialize(repo_root: Path) -> pd.DataFrame:
    config_path = Path(__file__).parent / "config" / "authoritative_mask_definitions.yaml"
    config = yaml.safe_load(config_path.read_text())
    masks = _resolve_masks(config["mask_definitions"])
    _assert_controlled_differences(masks)
    authority = config["authority"]
    all_summaries = []
    generated = {"authority": authority, "numbering": config["numbering"], "channels": {}}
    for channel, datasets in config["datasets"].items():
        table_dir = repo_root / channel / "dataRMSF" / "analysis" / "tables"
        table_dir.mkdir(parents=True, exist_ok=True)
        position_frames, summaries = [], []
        generated["channels"][channel] = {"datasets": {}}
        for dataset, dataset_config in datasets.items():
            mask_name = dataset_config["mask"]
            definition = config["mask_definitions"][mask_name]
            positions = masks[mask_name]
            raw_length = int(definition["raw_length"])
            frame = pd.DataFrame({
                "channel": channel,
                "dataset": dataset,
                "raw_residue_number": range(1, raw_length + 1),
            })
            frame["directly_masked"] = frame.raw_residue_number.isin(positions)
            frame["mask_definition"] = mask_name
            frame["mask_authority"] = authority
            position_frames.append(frame)
            summary = {
                "channel": channel,
                "dataset": dataset,
                "mask_definition": mask_name,
                "mask_authority": authority,
                "raw_query_length": raw_length,
                "number_of_directly_masked_residues": len(positions),
                "directly_masked_ranges": compact_ranges(positions),
                "validation_status": "passed",
            }
            summaries.append(summary)
            all_summaries.append(summary)
            generated["channels"][channel]["datasets"][dataset] = {
                "mask_definition": mask_name,
                "directly_masked_ranges": compact_ranges(positions),
                "number_of_directly_masked_residues": len(positions),
            }
        pd.concat(position_frames, ignore_index=True).to_csv(
            table_dir / f"{channel}_a3m_mask_positions.csv", index=False
        )
        pd.DataFrame(summaries).to_csv(
            table_dir / f"{channel}_a3m_mask_summary.csv", index=False
        )
    destination = Path(__file__).parent / "config" / "generated_mask_definitions.yaml"
    destination.write_text(yaml.safe_dump(generated, sort_keys=False))
    return pd.DataFrame(all_summaries)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    summary = materialize(repo_root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
