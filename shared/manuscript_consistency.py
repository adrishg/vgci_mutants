"""Plain-text consistency checks for the paired-seed manuscript revision."""

from __future__ import annotations

import csv
from pathlib import Path
import re
import zipfile
import xml.etree.ElementTree as ET


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


FORBIDDEN = {
    "risk_ratio_terminology": r"\brisk ratio\b",
    "subset_ci_coverage_wording": r"subset CI covered|Subset CI covered",
    "unsupported_mutation_specific_response": r"\bmutation-specific response\b",
    "coordinate_count_wording": r"35,602 coordinates(?!-by-comparison)",
    "local_structural_validity": r"\blocal structural validity\b",
    "progressive_qqq_displacement": r"progressive QQQ displacement",
    "rmsf_flexibility": r"RMSF flexibility|flexibility[^.]{0,40}RMSF",
    "three_decimal_percentage": r"\b\d+\.\d{3}%",
}


def extract_docx_text(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        root = ET.fromstring(archive.read("word/document.xml"))
    paragraphs = []
    for paragraph in root.iter(f"{{{W}}}p"):
        paragraphs.append("".join(node.text or "" for node in paragraph.iter(f"{{{W}}}t")))
    return "\n".join(paragraphs)


def audit_text(text: str, stale_values: list[str] | None = None) -> list[dict[str, str]]:
    findings = []
    for check, pattern in FORBIDDEN.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            findings.append({"check": check, "matched_text": match.group(0)})
    for value in stale_values or []:
        if value and value in text:
            findings.append({"check": "stale_registered_value", "matched_text": value})
    return findings


def stale_values_from_crosswalk(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = csv.DictReader(handle, delimiter="\t")
        return [row.get("old_display_text", "") for row in rows if row.get("old_display_text")]
