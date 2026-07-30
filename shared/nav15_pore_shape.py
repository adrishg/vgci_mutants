"""Nav1.5 intracellular-gate shape analysis.

The four pore-forming domains are treated as a cyclic quadrilateral:
DI (M415), DII (A742), DIII (I1154), and DIV (I1455).  The six pairwise
Cα distances already stored in the Nav1.5 distance tables are sufficient to
separate pore size from square-versus-rectangular shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import seaborn as sns

from shared.plotting import NAV15_EXPERIMENTAL_STYLES

from shared.structure_distances import read_ca_atoms


NAV15_GATE_COLUMNS = {
    "DI–DII": "CA_MET415_CA-ALA742_CA",
    "DI–DIII": "CA_MET415_CA-ILE1154_CA",
    "DI–DIV": "CA_MET415_CA-ILE1455_CA",
    "DII–DIII": "CA_ALA742_CA-ILE1154_CA",
    "DII–DIV": "CA_ALA742_CA-ILE1455_CA",
    "DIII–DIV": "CA_ILE1154_CA-ILE1455_CA",
}

NAV15_GATE_PAIR_INDEX = {
    "DI–DII": (0, 1),
    "DI–DIII": (0, 2),
    "DI–DIV": (0, 3),
    "DII–DIII": (1, 2),
    "DII–DIV": (1, 3),
    "DIII–DIV": (2, 3),
}

NAV15_GATE_LABELS = ("DI", "DII", "DIII", "DIV")

# Model label -> experimental PDB residue number.  7FBS/6UZ3/8T6L use the
# older rat construct numbering; 8VYJ/8VYK/7DTC use the full-length human map.
NAV15_EXPERIMENTAL_GATE_MAPS = {
    "6UZ3": {"DI": 415, "DII": 939, "DIII": 1472, "DIV": 1773},
    "7FBS": {"DI": 415, "DII": 939, "DIII": 1472, "DIV": 1773},
    "8T6L": {"DI": 415, "DII": 939, "DIII": 1472, "DIV": 1773},
    "7DTC": {"DI": 414, "DII": 936, "DIII": 1470, "DIV": 1771},
    "8VYJ": {"DI": 414, "DII": 936, "DIII": 1470, "DIV": 1771},
    "8VYK": {"DI": 414, "DII": 936, "DIII": 1470, "DIV": 1771},
}

NAV15_EXPERIMENTAL_STATES = {
    "6UZ3": "WT inactivated",
    "7FBS": "QQQ engineered open",
    "7DTC": "E1784K intermediate-inactivated",
    "8VYJ": "full-length open, Model I",
    "8VYK": "full-length expanded open, Model II",
    "8T6L": "BTX-B-bound comparator",
}


def _shape_metrics(distances: pd.DataFrame) -> pd.DataFrame:
    """Calculate size and quadrilateral-shape metrics from six distances."""
    edge_a = (
        distances["DI–DII"] + distances["DIII–DIV"]
    ) / 2.0
    edge_b = (
        distances["DII–DIII"] + distances["DI–DIV"]
    ) / 2.0
    smaller = np.minimum(edge_a, edge_b)
    larger = np.maximum(edge_a, edge_b)
    diagonal_a = distances["DI–DIII"]
    diagonal_b = distances["DII–DIV"]

    result = distances.copy()
    result["short_side_mean_A"] = smaller
    result["long_side_mean_A"] = larger
    result["side_aspect_ratio"] = larger / smaller
    result["squareness_index"] = smaller / larger
    result["mean_diagonal_A"] = (diagonal_a + diagonal_b) / 2.0
    result["diagonal_mismatch"] = (
        (diagonal_a - diagonal_b).abs()
        / ((diagonal_a + diagonal_b) / 2.0)
    )
    mismatch_a = (
        (distances["DI–DII"] - distances["DIII–DIV"]).abs()
        / ((distances["DI–DII"] + distances["DIII–DIV"]) / 2.0)
    )
    mismatch_b = (
        (distances["DII–DIII"] - distances["DI–DIV"]).abs()
        / ((distances["DII–DIII"] + distances["DI–DIV"]) / 2.0)
    )
    result["opposite_side_mismatch"] = (mismatch_a + mismatch_b) / 2.0
    return result


def ensemble_gate_shape(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Return one shape-metric row per AF2 model."""
    missing = [
        column for column in NAV15_GATE_COLUMNS.values() if column not in frame
    ]
    if missing:
        raise KeyError(f"Missing Nav1.5 gate distance columns: {missing}")
    distances = pd.DataFrame(
        {
            alias: pd.to_numeric(frame[column], errors="coerce")
            for alias, column in NAV15_GATE_COLUMNS.items()
        }
    ).dropna()
    result = _shape_metrics(distances)
    result.insert(0, "model", frame.loc[result.index, "pdb_file"].astype(str).values)
    result.insert(0, "dataset", label)
    return result.reset_index(drop=True)


def experimental_gate_shape(
    experimental_dir: str | Path,
    *,
    chain: str = "A",
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Measure all six gate distances and shape metrics for experimental PDBs."""
    experimental_dir = Path(experimental_dir)
    rows = []
    distance_audit: dict[str, dict[str, float]] = {}
    for pdb_id, residue_map in NAV15_EXPERIMENTAL_GATE_MAPS.items():
        atoms = read_ca_atoms(experimental_dir / f"{pdb_id}.pdb")
        coordinates = {}
        for domain, residue_number in residue_map.items():
            coordinate = atoms.get((chain, residue_number))
            if coordinate is None:
                raise KeyError(
                    f"{pdb_id}: Cα for chain {chain} residue {residue_number} is missing"
                )
            coordinates[domain] = np.asarray(coordinate, dtype=float)

        distances = {}
        for alias, (first, second) in NAV15_GATE_PAIR_INDEX.items():
            distance = float(
                np.linalg.norm(
                    coordinates[NAV15_GATE_LABELS[first]]
                    - coordinates[NAV15_GATE_LABELS[second]]
                )
            )
            distances[alias] = distance
        distance_audit[pdb_id] = distances
        metric_row = _shape_metrics(pd.DataFrame([distances])).iloc[0].to_dict()
        metric_row.update(
            {
                "dataset": pdb_id,
                "state": NAV15_EXPERIMENTAL_STATES[pdb_id],
                "model": pdb_id,
            }
        )
        rows.append(metric_row)
    return pd.DataFrame(rows), distance_audit


def embed_gate_distances(distances: Mapping[str, float]) -> np.ndarray:
    """Classical-MDS embedding of the six distances in the best-fit 2D plane."""
    matrix = np.zeros((4, 4), dtype=float)
    for alias, (first, second) in NAV15_GATE_PAIR_INDEX.items():
        matrix[first, second] = matrix[second, first] = float(distances[alias])
    centering = np.eye(4) - np.ones((4, 4)) / 4.0
    gram = -0.5 * centering @ (matrix ** 2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    coordinates = eigenvectors[:, order] * np.sqrt(
        np.clip(eigenvalues[order], 0, None)
    )

    # Use the DI–DIII diagonal as a stable horizontal reference.
    vector = coordinates[2] - coordinates[0]
    angle = np.arctan2(vector[1], vector[0])
    rotation = np.array(
        [[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]]
    )
    coordinates = coordinates @ rotation.T
    if coordinates[1, 1] < coordinates[3, 1]:
        coordinates[:, 1] *= -1
    return coordinates


def plot_shape_landscape(
    ensemble_metrics: pd.DataFrame,
    experimental_metrics: pd.DataFrame,
    palette: Mapping[str, str],
    *,
    sample_per_dataset: int = 1200,
):
    """Plot pore size against square-versus-rectangular shape."""
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for label, part in ensemble_metrics.groupby("dataset", sort=False):
        sample = part.sample(min(len(part), sample_per_dataset), random_state=27)
        ax.scatter(
            sample["mean_diagonal_A"],
            sample["side_aspect_ratio"],
            s=12,
            alpha=0.24,
            linewidths=0,
            color=palette[label],
            label=label,
        )
        ax.scatter(
            part["mean_diagonal_A"].median(),
            part["side_aspect_ratio"].median(),
            s=70,
            color=palette[label],
            edgecolor="white",
            linewidth=1.1,
            zorder=4,
        )

    markers = ("o", "s", "D", "^", "v", "P")
    for marker, (_, row) in zip(markers, experimental_metrics.iterrows()):
        toxin = row["dataset"] == "8T6L"
        ax.scatter(
            row["mean_diagonal_A"],
            row["side_aspect_ratio"],
            s=65 if not toxin else 55,
            marker=marker,
            facecolor="white" if not toxin else "#F1D8A8",
            edgecolor="#4A3657" if not toxin else "#8C6D31",
            linewidth=1.3,
            zorder=6,
            label=f"{row['dataset']} | {row['state']}",
        )

    ax.axhline(1.0, color="#8C7A96", linestyle=":", linewidth=1)
    ax.text(
        0.99, 1.01, "square limit",
        transform=ax.get_yaxis_transform(), ha="right", va="bottom",
        color="#76687F", fontsize=9,
    )
    ax.set(
        xlabel="Mean cross-pore diagonal (Å)  |  opening-size proxy",
        ylabel="Side aspect ratio  |  1 = square; larger = more rectangular",
        title=r"$\mathrm{Na}_{\mathrm{V}}1.5$ intracellular gate | pore size and four-domain shape",
    )
    ax.legend(
        title="AF2 ensembles and experimental structures",
        bbox_to_anchor=(0.5, -0.18),
        loc="upper center",
        ncol=3,
        frameon=True,
    )
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.14, bottom=0.30)
    return fig, ax


def plot_shape_distributions(
    ensemble_metrics: pd.DataFrame,
    experimental_metrics: pd.DataFrame,
    palette: Mapping[str, str],
):
    """Show the full ensemble aspect-ratio distributions with references."""
    order = list(dict.fromkeys(ensemble_metrics["dataset"]))
    fig, ax = plt.subplots(figsize=(11.5, 7.0))
    sns.violinplot(
        data=ensemble_metrics,
        x="dataset",
        y="side_aspect_ratio",
        order=order,
        palette=[palette[label] for label in order],
        inner="quartile",
        cut=0,
        linewidth=0.7,
        density_norm="width",
        ax=ax,
    )
    offsets = np.linspace(-0.24, 0.24, len(experimental_metrics))
    for dataset_index in range(len(order)):
        for offset, (_, row) in zip(
            offsets, experimental_metrics.iterrows()
        ):
            style = NAV15_EXPERIMENTAL_STYLES[row["dataset"]]
            ax.scatter(
                dataset_index + offset,
                row["side_aspect_ratio"],
                s=35,
                marker=style["marker"],
                facecolor="white",
                edgecolor=style["color"],
                linewidth=1.0,
                zorder=5,
            )
    handles = [
        Line2D(
            [0], [0], marker=style["marker"], linestyle="none", markersize=6,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            label=f"{row['dataset']} | {row['state']}",
        )
        for _, row in experimental_metrics.iterrows()
        for style in [NAV15_EXPERIMENTAL_STYLES[row["dataset"]]]
    ]
    ax.legend(
        handles=handles,
        title="Experimental references",
        bbox_to_anchor=(0.5, -0.20),
        loc="upper center",
        ncol=3,
        frameon=True,
    )
    ax.axhline(1.0, color="#8C7A96", linestyle=":", linewidth=1)
    ax.set(
        xlabel="",
        ylabel="Side aspect ratio  |  1 = square; larger = more rectangular",
        title=r"$\mathrm{Na}_{\mathrm{V}}1.5$ intracellular gate | square-to-rectangle distributions",
    )
    ax.tick_params(axis="x", rotation=24)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    sns.despine(ax=ax)
    fig.subplots_adjust(left=0.14, bottom=0.32)
    return fig, ax


def plot_representative_shapes(
    ensemble_metrics: pd.DataFrame,
    experimental_metrics: pd.DataFrame,
    palette: Mapping[str, str],
):
    """Draw quadrilaterals reconstructed from each median six-distance set."""
    datasets = list(dict.fromkeys(ensemble_metrics["dataset"]))
    references = ["6UZ3", "7DTC", "8VYJ", "8VYK", "7FBS", "8T6L"]
    entries = []
    for label in datasets:
        median = ensemble_metrics.loc[
            ensemble_metrics["dataset"].eq(label), NAV15_GATE_COLUMNS.keys()
        ].median()
        entries.append((label, median.to_dict(), palette[label], False))
    for pdb_id in references:
        row = experimental_metrics.loc[
            experimental_metrics["dataset"].eq(pdb_id)
        ].iloc[0]
        entries.append(
            (
                f"{pdb_id}\n{row['state']}",
                {alias: row[alias] for alias in NAV15_GATE_COLUMNS},
                NAV15_EXPERIMENTAL_STYLES[pdb_id]["color"],
                True,
            )
        )

    columns = 3
    rows = int(np.ceil(len(entries) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(10.5, 3.35 * rows))
    axes = np.asarray(axes).ravel()
    for ax, (label, distances, color, experimental) in zip(axes, entries):
        coordinates = embed_gate_distances(distances)
        closed = np.vstack([coordinates, coordinates[0]])
        ax.fill(
            closed[:, 0], closed[:, 1],
            color=color, alpha=0.10 if experimental else 0.20,
        )
        ax.plot(
            closed[:, 0], closed[:, 1],
            color=color, linewidth=1.7,
        )
        ax.scatter(
            coordinates[:, 0], coordinates[:, 1],
            color="white" if experimental else color,
            edgecolor=color, s=42, zorder=3,
        )
        for (x, y), domain in zip(coordinates, NAV15_GATE_LABELS):
            ax.annotate(domain, (x, y), xytext=(4, 4), textcoords="offset points",
                        fontsize=8, color="#3C3340")
        ax.set_title(label, fontsize=10)
        ax.set_aspect("equal", adjustable="datalim")
        ax.axis("off")
    for ax in axes[len(entries):]:
        ax.axis("off")
    fig.suptitle(
        r"$\mathrm{Na}_{\mathrm{V}}1.5$ intracellular gate | median/reconstructed four-domain outlines",
        fontsize=15,
        fontweight="semibold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    return fig, axes
