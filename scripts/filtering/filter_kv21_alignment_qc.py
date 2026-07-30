#!/usr/bin/env python3
"""Synchronize Kv2.1 distance tables with the final v2 RMSD alignment QC.

The distance tables already contain allOK3 convergence, selectivity-filter,
and pore–VSD interface checks. This final pass removes complete prediction
trajectories when a retained recycle either failed the v2 RMSD analysis or
entered the separated stable-core alignment-failure population.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "kv21" / "dataDistances"
RMSD_SOURCE = DATA_DIR.parent / "dataRMSD" / "Kv21_all_models_vs_8SD3_8SDA_RMSD_v2.csv"
SOURCE_SUFFIX = "_all_ok_rmsd_3A_structural_interface_qc.csv"
OUTPUT_SUFFIX = "_all_ok_rmsd_3A_structural_interface_alignment_qc.csv"
AUDIT_OUTPUT = DATA_DIR.parent / "dataRMSD" / "analysis" / "kv21_distance_alignment_qc_summary.csv"
STABLE_CORE_FAILURE_CUTOFF_A = 4.0


def basename(value: object) -> str:
    return Path(str(value).replace("\\", "/")).name


def trajectory(value: object) -> str:
    return pd.Series([basename(value)]).str.replace(
        r"\.r\d+\.pdb$", ".pdb", regex=True
    ).iat[0]


def main() -> None:
    rmsd = pd.read_csv(
        RMSD_SOURCE,
        low_memory=False,
        usecols=[
            "analysis_status", "pdb_file", "selected_core_postfit_rmsd_A",
        ],
    )
    rmsd["_model"] = rmsd["pdb_file"].map(basename)
    rmsd["_core"] = pd.to_numeric(
        rmsd["selected_core_postfit_rmsd_A"], errors="coerce"
    )
    ok = rmsd.loc[rmsd["analysis_status"].eq("ok")].copy()
    # The mapping/core diagnostic should agree across the two references.
    core_by_model = ok.groupby("_model")["_core"].max()

    sources = sorted(DATA_DIR.glob(f"*{SOURCE_SUFFIX}"))
    if len(sources) != 6:
        raise RuntimeError(f"Expected six Kv2.1 distance inputs; found {len(sources)}")

    audit_rows: list[dict[str, object]] = []
    for source in sources:
        frame = pd.read_csv(source, low_memory=False)
        model = frame["pdb_file"].map(basename)
        traj = model.map(trajectory)
        core = model.map(core_by_model)
        direct_missing = core.isna()
        direct_failed = core.ge(STABLE_CORE_FAILURE_CUTOFF_A)
        failed_trajectories = set(traj[direct_missing | direct_failed])
        rejected = traj.isin(failed_trajectories)
        retained = frame.loc[~rejected].copy()

        if retained.empty:
            raise RuntimeError(f"Alignment QC removed every row from {source}")
        retained_core = core.loc[~rejected]
        if retained_core.isna().any() or retained_core.ge(
            STABLE_CORE_FAILURE_CUTOFF_A
        ).any():
            raise AssertionError(f"Invalid RMSD mapping survived in {source}")

        output = source.with_name(source.name.replace(SOURCE_SUFFIX, OUTPUT_SUFFIX))
        retained.to_csv(output, index=False)
        audit_rows.append(
            {
                "source": str(source.relative_to(REPO_ROOT)),
                "output": str(output.relative_to(REPO_ROOT)),
                "stable_core_failure_cutoff_A": STABLE_CORE_FAILURE_CUTOFF_A,
                "input_rows": len(frame),
                "retained_rows": len(retained),
                "rejected_rows": int(rejected.sum()),
                "input_trajectories": int(traj.nunique()),
                "retained_trajectories": int(traj.loc[~rejected].nunique()),
                "rejected_trajectories": len(failed_trajectories),
                "direct_failed_rows": int(direct_failed.sum()),
                "rows_missing_successful_rmsd_analysis": int(direct_missing.sum()),
                "maximum_retained_stable_core_rmsd_A": float(retained_core.max()),
            }
        )
        print(
            f"{source.name}: {len(frame):,} → {len(retained):,} rows; "
            f"removed {len(failed_trajectories)} trajectories"
        )

    AUDIT_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(AUDIT_OUTPUT, index=False)
    print(f"Wrote audit: {AUDIT_OUTPUT}")


if __name__ == "__main__":
    main()
