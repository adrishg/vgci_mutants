"""Sampling-depth sensitivity analysis for final-QC distance ensembles."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from shared.experimental_overlays import experimental_rows
from shared.plotting import (
    ACCENT_PALETTE,
    experimental_reference_style,
    format_channel_title,
)


SUBSETS = ("Complete final QC", "100 retained trajectories")
SUBSET_COLORS = {
    "Complete final QC": ACCENT_PALETTE["PINK"],
    "100 retained trajectories": ACCENT_PALETTE["CORAL"],
}
_SEED = re.compile(r"_seed_(\d+)", re.I)
_MODEL = re.compile(r"_model_(\d+)", re.I)


def _trajectory_keys(frame: pd.DataFrame) -> pd.DataFrame:
    names = frame["pdb_file"].astype(str)
    return pd.DataFrame({
        "seed": pd.to_numeric(
            names.str.extract(_SEED, expand=False), errors="coerce"
        ),
        "model": pd.to_numeric(
            names.str.extract(_MODEL, expand=False), errors="coerce"
        ),
    }, index=frame.index)


def first_retained_trajectories(
    frame: pd.DataFrame, number: int = 100
) -> pd.DataFrame:
    """Select complete trajectories by deterministic seed/model order."""
    keys = _trajectory_keys(frame)
    if keys.isna().any(axis=None):
        bad = frame.loc[keys.isna().any(axis=1), "pdb_file"].head().tolist()
        raise ValueError(f"Could not parse trajectory identity: {bad}")
    key_table = keys.assign(_row_index=frame.index)
    chosen = (
        keys.drop_duplicates()
        .sort_values(["seed", "model"])
        .head(number)
    )
    selected_index = key_table.merge(
        chosen, on=["seed", "model"], how="inner"
    )["_row_index"]
    return frame.loc[selected_index].copy()


def load_final_qc_pair(paths: Mapping[str, Path]) -> dict[str, dict[str, pd.DataFrame]]:
    full = {
        protocol: pd.read_csv(path)
        for protocol, path in paths.items()
    }
    return {
        "Complete final QC": full,
        "100 retained trajectories": {
            protocol: first_retained_trajectories(frame)
            for protocol, frame in full.items()
        },
    }


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def rank_variability(
    frames: Mapping[str, Mapping[str, pd.DataFrame]],
    candidate_columns,
) -> pd.DataFrame:
    shared = set(candidate_columns)
    for subset in SUBSETS:
        for protocol in ("vanilla", "masked"):
            shared &= set(frames[subset][protocol].columns)
    rows = []
    for column in sorted(shared):
        row = {"distance": column}
        scores = []
        valid = True
        for subset in SUBSETS:
            vanilla = _numeric(frames[subset]["vanilla"], column)
            masked = _numeric(frames[subset]["masked"], column)
            if vanilla.empty or masked.empty:
                valid = False
                break
            vanilla_iqr = vanilla.quantile(0.75) - vanilla.quantile(0.25)
            masked_iqr = masked.quantile(0.75) - masked.quantile(0.25)
            ratio = (masked_iqr + 1e-9) / (vanilla_iqr + 1e-9)
            row[f"{subset} | vanilla IQR"] = vanilla_iqr
            row[f"{subset} | masked IQR"] = masked_iqr
            row[f"{subset} | IQR ratio"] = ratio
            scores.append(abs(np.log2(ratio)))
        if valid:
            row["ranking score"] = max(scores)
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["distance", "ranking score"])
    return pd.DataFrame(rows).sort_values(
        "ranking score", ascending=False
    )


def plot_sampling_depth_condition(
    repo_root: Path,
    condition: str,
    channel: str,
    region: str,
    frames,
    aliases: Mapping[str, str],
    protocol_colors: Mapping[str, str],
    top_n: int = 8,
):
    """Plot full final-QC and 100-trajectory protocol comparisons."""
    ranking = rank_variability(frames, aliases.values())
    if ranking.empty:
        return ranking, None
    columns = ranking.head(top_n)["distance"].tolist()
    visible = {column: alias for alias, column in aliases.items()}
    order = [visible[column] for column in columns]
    references = experimental_rows(repo_root, channel, region, order)

    figure, axes = plt.subplots(
        2, 1, figsize=(12.8, 10.5), sharex=True,
    )
    for axis, subset in zip(axes, SUBSETS):
        records = []
        for protocol in ("vanilla", "masked"):
            for column in columns:
                records.extend({
                    "Distance": value,
                    "Alias": visible[column],
                    "Protocol": protocol,
                } for value in _numeric(frames[subset][protocol], column))
        sns.violinplot(
            data=pd.DataFrame(records), x="Alias", y="Distance",
            hue="Protocol", order=order,
            hue_order=["vanilla", "masked"], split=True,
            inner="quartile", cut=0, linewidth=0.65,
            palette=protocol_colors, ax=axis,
        )
        used = set()
        structure_order = list(dict.fromkeys(
            row["Structure"] for row in references
        ))
        for row in references:
            if row["Alias"] not in order:
                continue
            structure = row["Structure"]
            style = experimental_reference_style(
                structure, structure_order.index(structure)
            )
            axis.scatter(
                order.index(row["Alias"]), row["Distance"],
                marker=style["marker"], s=30,
                facecolors="white", edgecolors=style["color"],
                linewidths=0.85, zorder=8,
                label=structure if structure not in used else None,
            )
            used.add(structure)
        axis.set_title(subset, fontsize=13, fontweight="semibold")
        axis.set_ylabel("Cα distance (Å)")
        axis.grid(axis="x", visible=False)
        sns.despine(ax=axis)
        handles, labels = axis.get_legend_handles_labels()
        if axis.get_legend() is not None:
            axis.get_legend().remove()
        axis.legend(
            handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1),
            title="Protocols and references", fontsize=8, frameon=True,
        )
    axes[-1].set_xlabel("Residue-pair alias")
    axes[-1].tick_params(axis="x", rotation=45)
    figure.suptitle(
        format_channel_title(
            f"{channel} | {condition} | {region.replace('_', ' ')} | sampling-depth sensitivity"
        ),
        fontsize=17, fontweight="semibold", y=0.995,
    )
    figure.subplots_adjust(right=0.79, bottom=0.14, top=0.92, hspace=0.16)
    return ranking, figure


def summarize_sampling_depth(condition, region, frames, ranking):
    rows = []
    for subset in SUBSETS:
        ratios = pd.to_numeric(
            ranking[f"{subset} | IQR ratio"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "Condition": condition,
            "Region": region,
            "Subset": subset,
            "Shared distances": len(ranking),
            "Median masked/vanilla IQR ratio": ratios.median(),
            "Fraction broader under masking": (ratios > 1).mean(),
            "Vanilla rows": len(frames[subset]["vanilla"]),
            "Masked rows": len(frames[subset]["masked"]),
        })
    return rows
