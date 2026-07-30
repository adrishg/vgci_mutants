"""A3M-aware extraction and validation of directly masked query residues."""

from __future__ import annotations

from pathlib import Path
import re
import pandas as pd


def parse_ranges(specification: str | None) -> set[int]:
    positions: set[int] = set()
    if not specification:
        return positions
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = map(int, item.split("-", 1))
            if start > end:
                raise ValueError(f"Descending mask range: {item}")
            positions.update(range(start, end + 1))
        else:
            positions.add(int(item))
    return positions


def compact_ranges(positions: set[int]) -> str:
    if not positions:
        return ""
    values = sorted(positions)
    blocks, start, previous = [], values[0], values[0]
    for value in values[1:]:
        if value != previous + 1:
            blocks.append(str(start) if start == previous else f"{start}-{previous}")
            start = value
        previous = value
    blocks.append(str(start) if start == previous else f"{start}-{previous}")
    return ", ".join(blocks)


def read_fasta_records(path: str | Path) -> list[tuple[str, str]]:
    records, name, sequence = [], None, []
    with Path(path).open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append((name, "".join(sequence)))
                name, sequence = line[1:], []
            else:
                if name is None:
                    raise ValueError(f"Sequence encountered before FASTA header in {path}")
                sequence.append(line)
    if name is not None:
        records.append((name, "".join(sequence)))
    if len(records) < 2:
        raise ValueError(f"A masked A3M needs a query and at least one non-query sequence: {path}")
    return records


def _strip_insertions(sequence: str) -> str:
    return re.sub(r"[a-z]", "", sequence)


def extract_a3m_mask(path: str | Path, expected_query_length: int | None = None) -> tuple[pd.DataFrame, dict]:
    """Return residue-level X statistics using the first A3M record as query."""
    records = read_fasta_records(path)
    processed = [(name, _strip_insertions(sequence)) for name, sequence in records]
    lengths = {len(sequence) for _, sequence in processed}
    if len(lengths) != 1:
        details = {name: len(sequence) for name, sequence in processed[:10]}
        raise ValueError(f"Inconsistent processed A3M lengths in {path}: {details}")
    query_name, query = processed[0]
    query_length = sum(character != "-" for character in query)
    if expected_query_length is not None and query_length != expected_query_length:
        raise ValueError(
            f"Query length mismatch for {path}: observed {query_length}, expected {expected_query_length}"
        )
    rows, raw_position = [], 0
    homologs = [sequence for _, sequence in processed[1:]]
    for column, query_character in enumerate(query):
        if query_character == "-":
            continue
        raw_position += 1
        values = [sequence[column].upper() for sequence in homologs]
        usable = [value for value in values if value != "-"]
        x_count = sum(value == "X" for value in usable)
        fraction = x_count / len(usable) if usable else float("nan")
        rows.append({
            "raw_residue_number": raw_position,
            "query_residue_identity": query_character.upper(),
            "alignment_column_1based": column + 1,
            "number_of_non_query_sequences": len(homologs),
            "number_with_residue_at_column": len(usable),
            "number_containing_X": x_count,
            "fraction_containing_X": fraction,
            "directly_masked": bool(usable) and x_count == len(usable),
            "all_usable_homologs_masked": bool(usable) and x_count == len(usable),
        })
    table = pd.DataFrame(rows)
    masked = set(table.loc[table.directly_masked, "raw_residue_number"].astype(int))
    warnings = []
    partial = table.query("number_containing_X > 0 and not directly_masked")
    if len(partial):
        warnings.append(f"{len(partial)} positions have X in only a subset of usable homologs")
    no_usable = table.number_with_residue_at_column.eq(0).sum()
    if no_usable:
        warnings.append(f"{no_usable} query positions have no usable non-query residue")
    summary = {
        "a3m_path": str(Path(path).resolve()), "query_name": query_name,
        "query_length": query_length, "number_of_non_query_sequences": len(homologs),
        "number_of_directly_masked_residues": len(masked),
        "directly_masked_ranges": compact_ranges(masked),
        "parsing_warnings": "; ".join(warnings),
    }
    return table, summary


def find_case_insensitive(root: str | Path, relative_pattern: str) -> Path:
    """Resolve a slash-delimited glob case-insensitively, requiring one match."""
    root = Path(root)
    pattern_parts = Path(relative_pattern).parts
    candidates = [root]
    for part in pattern_parts:
        regex = re.compile("^" + re.escape(part).replace(r"\*", ".*") + "$", re.I)
        next_candidates = []
        for parent in candidates:
            if parent.is_dir():
                next_candidates.extend(child for child in parent.iterdir() if regex.match(child.name))
        candidates = next_candidates
    files = [candidate for candidate in candidates if candidate.is_file()]
    if len(files) != 1:
        raise FileNotFoundError(
            f"Required A3M pattern {relative_pattern!r} under {root} resolved to {len(files)} files: {files}"
        )
    return files[0]


def validate_mask(mask: set[int], expected_count: int, required: set[int], label: str) -> None:
    if len(mask) != expected_count:
        raise AssertionError(f"{label}: extracted {len(mask)} masked residues; expected {expected_count}")
    missing = required - mask
    if missing:
        raise AssertionError(f"{label}: required checkpoint residues absent: {compact_ranges(missing)}")
