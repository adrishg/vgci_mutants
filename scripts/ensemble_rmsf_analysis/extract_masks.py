#!/usr/bin/env python3
"""Extract authoritative masks from production A3Ms and validate checkpoints."""

from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import yaml

from .masks import (
    compact_ranges, extract_a3m_mask, find_case_insensitive, parse_ranges, validate_mask,
)


def extract_all(repo_root: Path, a3m_root: Path) -> tuple[pd.DataFrame, dict]:
    config_path = Path(__file__).parent / "config" / "mask_definitions.yaml"
    config = yaml.safe_load(config_path.read_text())
    summary_rows, generated = [], {"authority": "A3M extraction", "channels": {}}
    for channel, channel_config in config["channels"].items():
        output = repo_root / channel / "dataRMSF" / "analysis" / "tables"
        output.mkdir(parents=True, exist_ok=True)
        channel_tables = []
        generated["channels"][channel] = {"raw_length": channel_config["raw_length"], "datasets": {}}
        for dataset, definition in channel_config["datasets"].items():
            matches, errors = [], []
            for pattern in definition["a3m_patterns"]:
                try:
                    matches.append(find_case_insensitive(a3m_root, pattern))
                except FileNotFoundError as error:
                    errors.append(str(error))
            matches = sorted(set(matches))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"{channel}/{dataset}: required one production A3M; found {len(matches)}. "
                    + " | ".join(errors)
                )
            table, summary = extract_a3m_mask(matches[0], channel_config["raw_length"])
            mask = set(table.loc[table.directly_masked, "raw_residue_number"].astype(int))
            validate_mask(
                mask, definition["expected_count"],
                parse_ranges(definition.get("required_ranges")), f"{channel}/{dataset}",
            )
            forbidden = parse_ranges(definition.get("forbidden_ranges"))
            if mask & forbidden:
                raise AssertionError(
                    f"{channel}/{dataset}: forbidden mask positions present: {compact_ranges(mask & forbidden)}"
                )
            mutation = definition.get("mutation_position")
            if mutation and mutation not in mask:
                raise AssertionError(f"{channel}/{dataset}: mutation position {mutation} is not masked")
            expected_sequence = definition.get("expected_local_sequence_485_495")
            if expected_sequence:
                observed = "".join(
                    table.query("485 <= raw_residue_number <= 495").query_residue_identity
                )
                if observed != expected_sequence:
                    raise AssertionError(
                        f"{channel}/{dataset}: raw 485-495 is {observed}, expected {expected_sequence}"
                    )
            table.insert(0, "dataset", dataset)
            table.insert(0, "channel", channel)
            channel_tables.append(table)
            summary_rows.append({"channel": channel, "dataset": dataset, **summary})
            generated["channels"][channel]["datasets"][dataset] = {
                "a3m_path": str(matches[0].resolve()),
                "directly_masked_ranges": compact_ranges(mask),
                "number_of_directly_masked_residues": len(mask),
            }
        pd.concat(channel_tables, ignore_index=True).to_csv(
            output / f"{channel}_a3m_mask_positions.csv", index=False
        )
    summary_frame = pd.DataFrame(summary_rows)
    for channel, part in summary_frame.groupby("channel"):
        part.to_csv(
            repo_root / channel / "dataRMSF" / "analysis" / "tables"
            / f"{channel}_a3m_mask_summary.csv", index=False
        )
    return summary_frame, generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--a3m-root", type=Path, default=None)
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    root = (args.a3m_root or repo).resolve()
    summary, generated = extract_all(repo, root)
    destination = Path(__file__).parent / "config" / "generated_mask_definitions.yaml"
    destination.write_text(yaml.safe_dump(generated, sort_keys=False))
    print(summary.to_string(index=False))
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
