#!/usr/bin/env python3

from pathlib import Path

import pandas as pd


DATA_DIR = Path("kv21/dataDistances")

FILES = {
    "vanilla": DATA_DIR
    / "26-02-11_Kv2.1_l403a_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_qc.csv",

    "masked": DATA_DIR
    / "26-02-11_Kv2.1_l403a_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_qc.csv",
}

# E423-N179 in the figure corresponds to GLU425-ASN181 in these CSVs.
DISTANCE_COLUMNS = {
    "A": "CA_CA_A_GLU425_CA-A_ASN181_CA",
    "B": "CA_CA_B_GLU425_CA-B_ASN181_CA",
    "C": "CA_CA_C_GLU425_CA-C_ASN181_CA",
    "D": "CA_CA_D_GLU425_CA-D_ASN181_CA",
}

THRESHOLDS = {
    "shifted threshold": 12.84,
    "8SDA shorter elongated distance": 14.17,
    "8SDA longer elongated distance": 16.24,
}

OUTPUT_DIR = DATA_DIR / "analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def check_not_lfs_pointer(path: Path) -> None:
    with path.open("rb") as handle:
        first_line = handle.readline()

    if first_line.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(
            f"{path} is still a Git LFS pointer. Run git lfs pull first."
        )


def analyze(condition: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    check_not_lfs_pointer(path)

    print("\n" + "=" * 90)
    print(condition.upper())
    print("=" * 90)
    print(f"Reading: {path}")

    df = pd.read_csv(path)

    required = ["pdb_file", *DISTANCE_COLUMNS.values()]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise KeyError(
            "The following required columns were not found:\n"
            + "\n".join(f"  {column}" for column in missing)
        )

    work = df[required].copy()

    for column in DISTANCE_COLUMNS.values():
        work[column] = pd.to_numeric(work[column], errors="coerce")

    work = work.dropna(
        subset=list(DISTANCE_COLUMNS.values()),
        how="all",
    ).copy()

    # Add simple chain names.
    for chain, original_column in DISTANCE_COLUMNS.items():
        work[f"chain_{chain}"] = work[original_column]

    chain_columns = [f"chain_{chain}" for chain in "ABCD"]

    # Per-structure values.
    work["maximum_distance_A"] = work[chain_columns].max(axis=1)
    work["minimum_distance_A"] = work[chain_columns].min(axis=1)
    work["mean_distance_A"] = work[chain_columns].mean(axis=1)

    work["chain_with_maximum"] = (
        work[chain_columns]
        .idxmax(axis=1)
        .str.replace("chain_", "", regex=False)
    )

    work["number_shifted_subunits"] = (
        work[chain_columns] >= THRESHOLDS["shifted threshold"]
    ).sum(axis=1)

    # Flatten all four chains to locate the exact global maximum.
    long_df = work.melt(
        id_vars=["pdb_file"],
        value_vars=chain_columns,
        var_name="chain",
        value_name="distance_A",
    )

    long_df["chain"] = long_df["chain"].str.replace(
        "chain_",
        "",
        regex=False,
    )

    long_df = long_df.dropna(subset=["distance_A"]).copy()

    maximum_row = long_df.loc[long_df["distance_A"].idxmax()]

    print(f"\nRetained structures: {len(work):,}")
    print(f"Total chain-resolved distances: {len(long_df):,}")

    print("\nEXACT MAXIMUM OBSERVED")
    print("-" * 90)
    print(f"Maximum distance: {maximum_row['distance_A']:.6f} Å")
    print(f"Chain: {maximum_row['chain']}")
    print(f"Structure: {maximum_row['pdb_file']}")

    max_structure = work.loc[
        work["pdb_file"] == maximum_row["pdb_file"]
    ].iloc[0]

    print("\nAll four distances in that structure:")
    for chain in "ABCD":
        print(
            f"  Chain {chain}: "
            f"{max_structure[f'chain_{chain}']:.6f} Å"
        )

    print("\nPER-CHAIN MAXIMA")
    print("-" * 90)

    per_chain_rows = []

    for chain in "ABCD":
        column = f"chain_{chain}"
        index = work[column].idxmax()

        value = work.loc[index, column]
        structure = work.loc[index, "pdb_file"]

        per_chain_rows.append(
            {
                "condition": condition,
                "chain": chain,
                "maximum_distance_A": value,
                "pdb_file": structure,
            }
        )

        print(
            f"Chain {chain}: {value:.6f} Å | {structure}"
        )

    print("\nMAXIMUM DISTANCE PER TETRAMER")
    print("-" * 90)

    quantiles = work["maximum_distance_A"].quantile(
        [0.50, 0.90, 0.95, 0.99, 0.999, 1.00]
    )

    for quantile, value in quantiles.items():
        print(f"q{quantile:.3f}: {value:.6f} Å")

    print("\nFRACTION OF STRUCTURES REACHING EACH LANDMARK")
    print("-" * 90)

    threshold_summary = {}

    for label, threshold in THRESHOLDS.items():
        any_subunit = (
            work[chain_columns].ge(threshold).any(axis=1)
        )
        all_subunits = (
            work[chain_columns].ge(threshold).all(axis=1)
        )

        percent_any = any_subunit.mean() * 100
        percent_all = all_subunits.mean() * 100

        threshold_summary[label] = {
            "threshold_A": threshold,
            "percent_any_subunit": percent_any,
            "percent_all_subunits": percent_all,
        }

        print(
            f"≥ {threshold:.2f} Å | {label}\n"
            f"  At least one subunit: {percent_any:.6f}%\n"
            f"  All four subunits:    {percent_all:.6f}%"
        )

    print("\nNUMBER OF SHIFTED SUBUNITS PER STRUCTURE")
    print("-" * 90)
    print("Threshold: 12.84 Å")

    occupancy = (
        work["number_shifted_subunits"]
        .value_counts()
        .reindex(range(5), fill_value=0)
        .sort_index()
    )

    occupancy_rows = []

    for number_shifted, count in occupancy.items():
        percentage = count / len(work) * 100

        occupancy_rows.append(
            {
                "condition": condition,
                "number_shifted_subunits": number_shifted,
                "count": count,
                "percentage": percentage,
            }
        )

        print(
            f"{number_shifted} shifted: "
            f"{count:,} structures ({percentage:.6f}%)"
        )

    # Save top structures.
    top50 = work.sort_values(
        "maximum_distance_A",
        ascending=False,
    ).head(50)

    top50_path = (
        OUTPUT_DIR
        / f"L403A_{condition}_E423_N179_top50.csv"
    )
    top50.to_csv(top50_path, index=False)

    print(f"\nTop 50 structures saved to:\n{top50_path}")

    summary = {
        "condition": condition,
        "n_structures": len(work),
        "global_maximum_A": maximum_row["distance_A"],
        "global_maximum_chain": maximum_row["chain"],
        "global_maximum_pdb_file": maximum_row["pdb_file"],
        "median_maximum_per_tetramer_A": quantiles.loc[0.50],
        "q95_maximum_per_tetramer_A": quantiles.loc[0.95],
        "q99_maximum_per_tetramer_A": quantiles.loc[0.99],
        "q999_maximum_per_tetramer_A": quantiles.loc[0.999],
    }

    for label, values in threshold_summary.items():
        safe_label = (
            label.lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        summary[
            f"percent_any_{safe_label}"
        ] = values["percent_any_subunit"]

        summary[
            f"percent_all_{safe_label}"
        ] = values["percent_all_subunits"]

    return (
        work,
        summary,
        pd.DataFrame(per_chain_rows),
        pd.DataFrame(occupancy_rows),
    )


def main():
    all_structures = []
    summaries = []
    per_chain_maxima = []
    occupancies = []

    for condition, path in FILES.items():
        (
            work,
            summary,
            chain_summary,
            occupancy,
        ) = analyze(condition, path)

        work.insert(0, "condition", condition)

        all_structures.append(work)
        summaries.append(summary)
        per_chain_maxima.append(chain_summary)
        occupancies.append(occupancy)

    combined_structures = pd.concat(
        all_structures,
        ignore_index=True,
    )

    summary_df = pd.DataFrame(summaries)

    chain_df = pd.concat(
        per_chain_maxima,
        ignore_index=True,
    )

    occupancy_df = pd.concat(
        occupancies,
        ignore_index=True,
    )

    combined_structures.to_csv(
        OUTPUT_DIR
        / "L403A_E423_N179_all_structure_distances.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR
        / "L403A_E423_N179_summary.csv",
        index=False,
    )

    chain_df.to_csv(
        OUTPUT_DIR
        / "L403A_E423_N179_per_chain_maxima.csv",
        index=False,
    )

    occupancy_df.to_csv(
        OUTPUT_DIR
        / "L403A_E423_N179_shifted_subunit_occupancy.csv",
        index=False,
    )

    print("\n" + "=" * 90)
    print("FINAL VANILLA VERSUS MASKED SUMMARY")
    print("=" * 90)

    columns_to_print = [
        "condition",
        "n_structures",
        "global_maximum_A",
        "global_maximum_chain",
        "q95_maximum_per_tetramer_A",
        "q99_maximum_per_tetramer_A",
        "q999_maximum_per_tetramer_A",
    ]

    print(
        summary_df[columns_to_print].to_string(
            index=False,
        )
    )

    print("\nResults saved under:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
