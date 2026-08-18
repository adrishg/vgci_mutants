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
    add_s6_cross_pore_columns,
    experimental_reference_style,
    format_channel_title,
)


EARLY_SUBSET = "QC survivors from nominal first 100"
SUBSETS = ("Complete final QC", EARLY_SUBSET)
SUBSET_COLORS = {
    "Complete final QC": ACCENT_PALETTE["PINK"],
    EARLY_SUBSET: ACCENT_PALETTE["CORAL"],
}
_SEED = re.compile(r"_seed_(\d+)", re.I)
_MODEL = re.compile(r"_model_(\d+)", re.I)
_RECYCLE = re.compile(r"\.r(\d+)\.pdb$", re.I)


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


def first_nominal_trajectory_cohort(
    frame: pd.DataFrame, number_seeds: int = 20
) -> pd.DataFrame:
    """Select QC survivors from the first N ordered seed cohorts.

    With five AlphaFold model trajectories per seed, 20 seeds define the
    nominal first 100 generated trajectories before QC attrition.
    """
    keys = _trajectory_keys(frame)
    if keys.isna().any(axis=None):
        bad = frame.loc[keys.isna().any(axis=1), "pdb_file"].head().tolist()
        raise ValueError(f"Could not parse trajectory identity: {bad}")
    seeds = keys["seed"].drop_duplicates().sort_values().head(number_seeds)
    return frame.loc[keys["seed"].isin(seeds)].copy()


def latest_qc_trajectory_representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the latest final-QC recycle for each seed/model trajectory."""
    keys = _trajectory_keys(frame)
    recycle = pd.to_numeric(
        frame["pdb_file"].astype(str).str.extract(_RECYCLE, expand=False),
        errors="coerce",
    )
    if keys.isna().any(axis=None) or recycle.isna().any():
        bad = frame.loc[keys.isna().any(axis=1) | recycle.isna(), "pdb_file"].head().tolist()
        raise ValueError(f"Could not parse trajectory/recycle identity: {bad}")
    order = keys.assign(recycle=recycle, _row_index=frame.index)
    selected = (
        order.sort_values(["seed", "model", "recycle", "_row_index"])
        .groupby(["seed", "model"], sort=False)
        .tail(1)["_row_index"]
    )
    return frame.loc[selected].copy()


def _kv21_ranked_ring_columns(
    frame: pd.DataFrame, residue_name: str = "GLY", residue_number: int = 377
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Add chain-label-invariant sorted distances for a homotetrameric ring."""
    result = frame.copy()
    pairs = ("A-B", "A-C", "A-D", "B-C", "B-D", "C-D")
    columns = [
        f"CA_CA_{pair[0]}_{residue_name}{residue_number}_CA-{pair[2]}_{residue_name}{residue_number}_CA"
        for pair in pairs
    ]
    missing = [column for column in columns if column not in result]
    if missing:
        raise KeyError(f"Missing Kv2.1 ring columns: {missing}")
    ranked = np.sort(result[columns].apply(pd.to_numeric, errors="coerce").to_numpy(), axis=1)
    aliases = {}
    for rank in range(6):
        output = f"Kv21_SF_ring_rank_{rank + 1}"
        result[output] = ranked[:, rank]
        aliases[f"G375 ring rank {rank + 1}"] = output
    return result, aliases


def load_final_qc_pair(paths: Mapping[str, Path]) -> dict[str, dict[str, pd.DataFrame]]:
    raw = {
        protocol: pd.read_csv(path)
        for protocol, path in paths.items()
    }
    full = {
        protocol: latest_qc_trajectory_representatives(frame)
        for protocol, frame in raw.items()
    }
    return {
        "Complete final QC": full,
        EARLY_SUBSET: {
            protocol: latest_qc_trajectory_representatives(
                first_nominal_trajectory_cohort(frame)
            )
            for protocol, frame in raw.items()
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
    """Plot full final-QC and fixed nominal first-100 cohort comparisons."""
    # Kv2.1 is homotetrameric, so raw A-B/A-C labels are not stable identities
    # when masking changes subunit ordering. Replace those panels with
    # chain-label-invariant summaries before ranking or plotting.
    if channel == "Kv2.1" and region in {"selectivity_filter", "intracellular_gate"}:
        prepared = {subset: {} for subset in SUBSETS}
        invariant_aliases = None
        for subset in SUBSETS:
            for protocol in ("vanilla", "masked"):
                if region == "selectivity_filter":
                    prepared[subset][protocol], current = _kv21_ranked_ring_columns(frames[subset][protocol])
                else:
                    prepared[subset][protocol], current = add_s6_cross_pore_columns(frames[subset][protocol])
                invariant_aliases = current
        frames = prepared
        aliases = invariant_aliases
    ranking = rank_variability(frames, aliases.values())
    if ranking.empty:
        return ranking, None
    columns = ranking.head(top_n)["distance"].tolist()
    visible = {column: alias for alias, column in aliases.items()}
    order = [visible[column] for column in columns]
    references = experimental_rows(repo_root, channel, region, order)
    # The 9O10–9O13 series is used here as a WT state ladder. Keeping those
    # markers off the mutant panels avoids implying mutation-matched validation.
    if channel == "Kv2.1" and condition != "WT":
        references = [
            row for row in references
            if not row["Structure"].startswith(("9O10:", "9O11:", "9O12:", "9O13:"))
        ]

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
