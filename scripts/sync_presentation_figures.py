"""Collect publication-resolution analysis figures in the writing repository."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image


CHANNELS = ("kv21", "nav15", "cav12")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--writing-root", type=Path)
    args = parser.parse_args()
    writing_root = args.writing_root or args.repo_root.parent / "vgci_mutants_writing"
    figure_root = writing_root / "figures"
    rows = []

    for channel in CHANNELS:
        patterns = (
            f"{channel}/dataRMSD/analysis/**/figures/*.png",
            f"{channel}/dataRMSF/analysis/figures/*.png",
            f"{channel}/dataDistances/analysis/figures/*.png",
        )
        sources = sorted({path for pattern in patterns for path in args.repo_root.glob(pattern)})
        for source in sources:
            relative = source.relative_to(args.repo_root / channel)
            destination = figure_root / channel / "analysis_exports" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            with Image.open(source) as image:
                dpi = image.info.get("dpi", ("", ""))
                rows.append({
                    "channel": {"kv21": "Kv2.1", "nav15": "Nav1.5", "cav12": "CaV1.2"}[channel],
                    "condition": next(
                        (part for part in relative.parts if part.lower() in
                         {"wt", "l403a", "f412l", "qqq", "g402s", "g406r", "g490r"}),
                        "multiple or not condition-specific",
                    ),
                    "source": str(source.relative_to(args.repo_root)),
                    "writing_copy": str(destination.relative_to(writing_root)),
                    "width_px": image.width,
                    "height_px": image.height,
                    "dpi_x": dpi[0] if dpi else "",
                    "dpi_y": dpi[1] if dpi else "",
                })

    manifest = figure_root / "figure_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Copied {len(rows)} analysis figures; manifest: {manifest}")


if __name__ == "__main__":
    main()
