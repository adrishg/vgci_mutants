"""Structural integrity checks for targeted OOXML manuscript edits."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import zipfile
import xml.etree.ElementTree as ET


REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def package_snapshot(path: Path) -> dict[str, object]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        document = archive.read("word/document.xml")
        media = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in names if name.startswith("word/media/")
        }
        unresolved = []
        for rel_name in (name for name in names if name.endswith(".rels")):
            root = ET.fromstring(archive.read(rel_name))
            if rel_name == "_rels/.rels":
                base = PurePosixPath("")
            else:
                rel_path = PurePosixPath(rel_name)
                base = rel_path.parent.parent
            for relationship in root.findall(f"{{{REL_NS}}}Relationship"):
                if relationship.get("TargetMode") == "External":
                    continue
                target = relationship.get("Target", "")
                resolved = str(base / target)
                normalized = str(PurePosixPath(resolved))
                while "/../" in f"/{normalized}":
                    normalized = re.sub(r"(^|/)[^/]+/\.\./", r"\1", normalized, count=1)
                normalized = normalized.lstrip("/")
                if normalized not in names:
                    unresolved.append(f"{rel_name}: {target}")
        return {
            "parts": tuple(sorted(names)),
            "media_sha256": media,
            "comment_parts": tuple(sorted(name for name in names if "comment" in name.lower())),
            "insertions": len(re.findall(b"<w:ins(?: |>)", document)),
            "deletions": len(re.findall(b"<w:del(?: |>)", document)),
            "comment_references": len(re.findall(b"commentReference", document)),
            "unresolved_relationships": tuple(unresolved),
        }


def compare_packages(
    original: Path,
    revised: Path,
    *,
    allowed_added_parts: set[str] | None = None,
    allowed_changed_parts: set[str] | None = None,
) -> list[str]:
    before, after = package_snapshot(original), package_snapshot(revised)
    issues = []
    allowed_added_parts = allowed_added_parts or set()
    allowed_changed_parts = allowed_changed_parts or set()
    before_parts, after_parts = set(before["parts"]), set(after["parts"])
    if before_parts - after_parts or after_parts - before_parts != allowed_added_parts:
        issues.append("parts changed")
    before_media, after_media = before["media_sha256"], after["media_sha256"]
    added_media = set(after_media) - set(before_media)
    allowed_added_media = {part for part in allowed_added_parts if part.startswith("word/media/")}
    allowed_changed_media = {part for part in allowed_changed_parts if part.startswith("word/media/")}
    changed_media = {
        key for key, digest in before_media.items()
        if after_media.get(key) != digest
    }
    if changed_media != allowed_changed_media or added_media != allowed_added_media:
        issues.append("media_sha256 changed")
    for key in ("comment_parts", "insertions", "deletions", "comment_references"):
        if before[key] != after[key]:
            issues.append(f"{key} changed")
    if after["unresolved_relationships"]:
        issues.append(f"unresolved relationships: {after['unresolved_relationships']}")
    return issues
