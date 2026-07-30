"""Automatic, descriptive 300-dpi exports for plot-focused notebooks."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt


def _slug(text: str) -> str:
    text = re.sub(r"\$[^$]*\$", "", str(text))
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text[:110] or "untitled_figure"


def install_notebook_figure_export(
    output_dir: str | Path, notebook_label: str
) -> Path:
    """Save each figure at its first ``plt.show()`` call.

    The hook is intentionally small and notebook-local. Existing plotting code
    remains readable, while every displayed comparison receives a descriptive
    300-dpi PNG in the writing figure bundle.
    """
    output_dir = Path(output_dir) / _slug(notebook_label)
    output_dir.mkdir(parents=True, exist_ok=True)
    if getattr(plt.show, "_vgic_export_hook", False):
        return output_dir

    original_show = plt.show
    counter = {"value": 0}

    def show_and_export(*args, **kwargs):
        for number in plt.get_fignums():
            figure = plt.figure(number)
            if getattr(figure, "_vgic_exported", False):
                continue
            counter["value"] += 1
            titles = []
            if figure._suptitle is not None:
                titles.append(figure._suptitle.get_text())
            titles.extend(axis.get_title() for axis in figure.axes if axis.get_title())
            title = next((item for item in titles if item.strip()), "figure")
            filename = f"{counter['value']:02d}_{_slug(title)}.png"
            figure.savefig(
                output_dir / filename, dpi=300, bbox_inches="tight",
                facecolor="white",
            )
            figure._vgic_exported = True
        return original_show(*args, **kwargs)

    show_and_export._vgic_export_hook = True
    plt.show = show_and_export
    return output_dir
