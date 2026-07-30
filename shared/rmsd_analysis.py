"""Shared summaries, effect sizes and restrained plots for RMSD notebooks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE_ORDER = {
    "kv21": ["8SD3", "8SDA"],
    "nav15": ["8VYJ", "8VYK", "7DTC", "6UZ3", "7FBS", "8T6L"],
    "cav12": ["8HLP", "8WE6", "8FD7"],
}
PROTOCOL_PALETTE = {
    "Vanilla": "#D9D7F2", "Masked": "#6764B8",
    "Masked v2": "#3E3D91", "Masked v2 no IFM": "#292A70",
}


def apply_kv21_rmsd_qc(
    frame: pd.DataFrame,
    repo_root: str | Path,
    *,
    stable_core_failure_cutoff_A: float = 4.0,
) -> pd.DataFrame:
    """Apply the independent Kv2.1 structural and v2 alignment QC.

    The allOK3 filter establishes convergence within a prediction trajectory;
    it does not guarantee a physically assembled tetramer. This filter first
    joins the trajectory-level selectivity-filter/pore–VSD interface allowlists
    used by the distance notebooks. It then removes complete trajectories in
    the clearly separated v2 alignment-failure population.

    The latter begins above 5.7 Å in the current table, with no observations
    between 4 and 5.7 Å and concurrent 11–16 Å S6-bundle RMSD. A conservative
    4 Å threshold therefore separates mapping/topology failures without
    trimming the continuous conformational distribution below it.
    """
    required = {
        "sequence_condition", "protocol", "pdb_file",
        "selected_core_postfit_rmsd_A",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Kv2.1 RMSD QC is missing columns: {sorted(missing)}")

    root = Path(repo_root)
    work = frame.copy()
    work["_pdb_basename"] = work["pdb_file"].astype(str).map(lambda x: Path(x).name)
    work["_trajectory"] = work["_pdb_basename"].str.replace(
        r"\.r\d+\.pdb$", ".pdb", regex=True
    )

    allowlists: dict[tuple[str, str], set[str]] = {}
    for condition in ("wt", "l403a", "f412l"):
        for protocol in ("vanilla", "masked"):
            matches = sorted(
                (root / "kv21/dataDistances").glob(
                    f"*{condition}_{protocol}*structural_interface_qc.csv"
                )
            )
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one Kv2.1 structural-interface allowlist for "
                    f"{condition}/{protocol}; found {matches}"
                )
            values = pd.read_csv(matches[0], usecols=["pdb_file"])["pdb_file"]
            allowlists[(condition, protocol)] = set(
                values.astype(str).map(lambda x: Path(x).name)
            )

    interface_ok = work.apply(
        lambda row: row["_pdb_basename"]
        in allowlists[
            (str(row["sequence_condition"]).lower(), str(row["protocol"]).lower())
        ],
        axis=1,
    )
    after_interface = work.loc[interface_ok].copy()

    core = pd.to_numeric(
        after_interface["selected_core_postfit_rmsd_A"], errors="coerce"
    )
    failed_trajectories = set(
        after_interface.loc[
            core.ge(stable_core_failure_cutoff_A) | core.isna(), "_trajectory"
        ]
    )
    result = after_interface.loc[
        ~after_interface["_trajectory"].isin(failed_trajectories)
    ].copy()
    result = result.drop(columns=["_pdb_basename", "_trajectory"])
    result.attrs.update(frame.attrs)
    result.attrs["kv21_interface_qc_rejected_rows"] = int((~interface_ok).sum())
    result.attrs["kv21_alignment_qc_rejected_trajectories"] = len(failed_trajectories)
    result.attrs["kv21_alignment_qc_rejected_rows"] = int(
        len(after_interface) - len(result)
    )
    result.attrs["kv21_stable_core_failure_cutoff_A"] = float(
        stable_core_failure_cutoff_A
    )
    print(
        f"Kv2.1 structural RMSD QC: {len(frame):,} rows → "
        f"{len(after_interface):,} after tetramer/interface QC → {len(result):,} "
        f"after excluding {len(failed_trajectories)} separated alignment-failure "
        f"trajectories (stable-core cutoff {stable_core_failure_cutoff_A:g} Å)"
    )
    return result


def rmsd_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame if c.lower().endswith("rmsd_a")]


def protocol_label(value: object) -> str:
    low = str(value).lower()
    if "noifm" in low or "no_ifm" in low:
        return "Masked v2 no IFM"
    if "masked_v2" in low or "maskedv2" in low:
        return "Masked v2"
    return "Masked" if "mask" in low else "Vanilla"


def humanize_measurement(column: str) -> str:
    """Publication-facing label for a pipeline RMSD column."""
    region, *rest = column.split("__")
    region_labels = {
        "best_mapping_core_ca_rmsd_A": "Stable alignment core",
        "whole_matched_tetramer": "Whole matched tetramer",
        "whole_matched_structure": "Whole matched structure",
        "pore_domain": "Pore domain",
        "DI_s6": "DI S6",
        "DII_s6": "DII S6",
        "s6_bundle_working": "Four-chain S6 bundle",
        "distal_s6_working": "Distal S6",
        "l403_region": "L403 region",
        "f412_region": "F412 region",
        "hydrophobic_nexus": "Hydrophobic coupling nexus",
        "ifm_receptor_pocket": "IFM receptor pocket",
        "ifm_motif": "IFM motif",
        "iii_iv_linker": "III-IV linker",
        "iii_iv_linker_ctd_interface": "III-IV linker–CTD interface",
        "ctd": "CTD",
        "small_residue_nexus": "Small-residue nexus",
        "IS6_IVS6_interface": "IS6-IVS6 interface",
        "g402_region": "G402 region",
        "g406_region": "G406 region",
    }
    label = region_labels.get(region, region.replace("_", " "))
    label = label.replace("VSDIII", "VSD III").replace("VSDIV", "VSD IV")
    label = label.replace("VSDII", "VSD II").replace("VSDI", "VSD I")
    label = label.replace("IIS6", "DII S6").replace("IIIS6", "DIII S6")
    label = label.replace("IVS6", "DIV S6").replace("IS6", "DI S6")
    low = column.lower()
    atom = "Cα" if "__ca__" in low else ("backbone" if "__bb__" in low else "")
    frame = "core-aligned" if "core_aligned" in low else ("locally aligned" if "local_aligned" in low else "")
    qualifier = ", ".join(x for x in (atom, frame) if x)
    return f"{label} ({qualifier})" if qualifier else label


def measurement_companions(column: str) -> tuple[str | None, str | None]:
    stem = column.removesuffix("rmsd_A")
    atom = stem + "matched_atoms"
    coverage = stem + "atom_coverage"
    return atom, coverage


def summary_statistics(
    frame: pd.DataFrame, measurements: list[str], group_cols=None
) -> pd.DataFrame:
    group_cols = group_cols or ["sequence_condition", "protocol", "reference_id"]
    rows = []
    for keys, part in frame.groupby(group_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        base = dict(zip(group_cols, keys))
        for measurement in measurements:
            values = pd.to_numeric(part[measurement], errors="coerce")
            atom, coverage = measurement_companions(measurement)
            rows.append({
                **base, "measurement": measurement,
                "n_unique_models": part["pdb_file"].nunique(),
                "n_measured": int(values.notna().sum()),
                "median": values.median(), "q25": values.quantile(.25),
                "q75": values.quantile(.75), "iqr": values.quantile(.75) - values.quantile(.25),
                "p05": values.quantile(.05), "p95": values.quantile(.95),
                "missing_fraction": values.isna().mean(),
                "median_aligned_atoms": pd.to_numeric(part[atom], errors="coerce").median()
                if atom in part else np.nan,
                "median_coverage": pd.to_numeric(part[coverage], errors="coerce").median()
                if coverage in part else np.nan,
            })
    return pd.DataFrame(rows)


def cliffs_delta(x, y) -> float:
    x, y = pd.Series(x).dropna().to_numpy(), pd.Series(y).dropna().to_numpy()
    if not len(x) or not len(y):
        return np.nan
    # Rank-based form avoids an O(n*m) comparison matrix.
    pooled = pd.Series(np.r_[x, y]).rank(method="average").to_numpy()
    rank_x = pooled[:len(x)].sum()
    u = rank_x - len(x) * (len(x) + 1) / 2
    return float((2 * u) / (len(x) * len(y)) - 1)


def bootstrap_median_difference(x, y, iterations=2000, seed=10403):
    """Masked minus vanilla median and percentile bootstrap CI."""
    x, y = pd.Series(x).dropna().to_numpy(), pd.Series(y).dropna().to_numpy()
    if not len(x) or not len(y):
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    observed = np.median(y) - np.median(x)
    boot = np.empty(iterations)
    for i in range(iterations):
        boot[i] = np.median(rng.choice(y, len(y))) - np.median(rng.choice(x, len(x)))
    return observed, *np.quantile(boot, [.025, .975])


def protocol_effects(frame: pd.DataFrame, measurements: list[str]) -> pd.DataFrame:
    rows = []
    work = frame.copy()
    work["protocol_display"] = work["protocol"].map(protocol_label)
    for (condition, reference), part in work.groupby(["sequence_condition", "reference_id"]):
        for measurement in measurements:
            vanilla = pd.to_numeric(
                part.loc[part.protocol_display.eq("Vanilla"), measurement], errors="coerce"
            ).dropna()
            alternatives = [x for x in part.protocol_display.dropna().unique() if x != "Vanilla"]
            for alternative in alternatives:
                masked = pd.to_numeric(
                    part.loc[part.protocol_display.eq(alternative), measurement], errors="coerce"
                ).dropna()
                difference, low, high = bootstrap_median_difference(vanilla, masked)
                rows.append({
                    "sequence_condition": condition, "reference_id": reference,
                    "comparison_protocol": alternative, "measurement": measurement,
                    "n_vanilla": len(vanilla), "n_masked": len(masked),
                    "vanilla_median": vanilla.median(), "masked_median": masked.median(),
                    "masked_minus_vanilla_median": difference,
                    "bootstrap_ci_low": low, "bootstrap_ci_high": high,
                    "cliffs_delta_masked_vs_vanilla": cliffs_delta(masked, vanilla),
                    "recycle_dependence_note": "Recycle snapshots pooled; effect size is descriptive.",
                })
    return pd.DataFrame(rows)


def reference_preference(
    frame: pd.DataFrame,
    measurement: str,
    reference_a: str,
    reference_b: str,
    *,
    tie_tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Pair each model's RMSD to two references and quantify reference preference.

    The signed score is RMSD(reference B) - RMSD(reference A). Positive values
    are therefore closer to A, while negative values are closer to B.
    """
    required = {"pdb_file", "reference_id", measurement}
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Reference-preference analysis is missing columns: {sorted(missing)}")

    work = frame.loc[
        frame["reference_id"].astype(str).isin([reference_a, reference_b])
    ].copy()
    work[measurement] = pd.to_numeric(work[measurement], errors="coerce")
    identity = [
        c for c in ("dataset", "sequence_condition", "protocol", "Protocol", "pdb_file")
        if c in work.columns
    ]
    paired = work.pivot_table(
        index=identity, columns="reference_id", values=measurement, aggfunc="first"
    ).reset_index()
    paired.columns.name = None
    if reference_a not in paired or reference_b not in paired:
        return pd.DataFrame()
    paired = paired.dropna(subset=[reference_a, reference_b]).copy()
    paired = paired.rename(columns={
        reference_a: "rmsd_reference_a_A",
        reference_b: "rmsd_reference_b_A",
    })
    paired["reference_a"] = reference_a
    paired["reference_b"] = reference_b
    paired["measurement"] = measurement
    paired["delta_b_minus_a_A"] = (
        paired["rmsd_reference_b_A"] - paired["rmsd_reference_a_A"]
    )
    paired["closer_reference"] = np.select(
        [
            paired["delta_b_minus_a_A"] > tie_tolerance,
            paired["delta_b_minus_a_A"] < -tie_tolerance,
        ],
        [reference_a, reference_b],
        default="Tie",
    )
    return paired


def summarize_reference_preference(paired: pd.DataFrame) -> pd.DataFrame:
    """Summarize paired reference-preference scores by protocol."""
    if paired.empty:
        return pd.DataFrame()
    protocol_col = "Protocol" if "Protocol" in paired else "protocol"
    rows = []
    for protocol, part in paired.groupby(protocol_col, dropna=False):
        counts = part["closer_reference"].value_counts()
        n = len(part)
        rows.append({
            "protocol": protocol,
            "reference_a": part["reference_a"].iloc[0],
            "reference_b": part["reference_b"].iloc[0],
            "measurement": part["measurement"].iloc[0],
            "n_paired_models": n,
            "median_delta_b_minus_a_A": part["delta_b_minus_a_A"].median(),
            "fraction_closer_to_a": counts.get(part["reference_a"].iloc[0], 0) / n,
            "fraction_closer_to_b": counts.get(part["reference_b"].iloc[0], 0) / n,
            "fraction_tied": counts.get("Tie", 0) / n,
        })
    return pd.DataFrame(rows)


def principal_measurements(frame: pd.DataFrame, channel: str, limit=12) -> list[str]:
    priorities = {
        "kv21": ["best_mapping_core", "whole_matched", "s6_bundle", "distal_s6", "l403_region", "f412_region", "hydrophobic_nexus"],
        "nav15": ["stable_core", "pore", "s6", "ifm", "linker", "ctd", "vsd"],
        "cav12": ["stable_core", "pore", "s6", "g402", "g406", "is6", "ivs6", "nexus"],
    }[channel]
    columns = rmsd_columns(frame)
    chosen = []
    for token in priorities:
        candidates = [c for c in columns if token in c.lower() and "chain_" not in c.lower()]
        candidates.sort(key=lambda c: (
            0 if "__ca__core_aligned" in c else
            1 if "__ca__local_aligned" in c else
            2 if "__bb__core_aligned" in c else 3,
            len(c),
        ))
        chosen.extend(c for c in candidates[:2] if c not in chosen)
    return chosen[:limit] or columns[:limit]


def save_basic_distribution(frame, measurement, title, output: Path):
    import matplotlib.pyplot as plt
    import seaborn as sns
    work = frame.copy()
    work["Protocol"] = work.protocol.map(protocol_label)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    sns.violinplot(
        data=work, x="reference_id", y=measurement, hue="Protocol",
        order=[x for x in REFERENCE_ORDER.get("kv21", []) if x in set(work.reference_id)],
        palette=PROTOCOL_PALETTE, cut=0, inner="quart", linewidth=.8, ax=ax,
    )
    ax.set(title=title, xlabel="Experimental reference", ylabel="Cα RMSD (Å)")
    ax.grid(axis="y", alpha=.16)
    sns.despine(ax=ax)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
