# Distance-distribution statistics

> This notebook contains the paired common-survivor model-seed sensitivity
> analysis. The primary all-survivor seed-block analysis is documented in
> `analysis/statistics_revision/README.md`, and the complete all-distance panel
> is under `analysis/statistics_revision/seed_block/full_panel/`.

`01_distance_distribution_statistics.ipynb` analyzes existing distance CSVs
without reading PDB/CIF coordinates or recalculating structural distances.

This notebook pairs common `(seed, model_number)` survivors. Retained recycle
snapshots remain within each trajectory. Its W1, weighted median, weighted IQR,
pooled weighted IQR, normalized W1, and weighted KS now use the same
equal-trajectory weights. Because QC survival depends on protocol, these
complete-case results are interpreted as a sensitivity analysis alongside the
primary all-survivor seed-block analysis.

Run the notebook from the repository root only when producing the paired
sensitivity. Its configuration cell selects
`exploratory` (999 permutations, 500 bootstrap samples) or `publication`
(9,999 permutations, 2,000 bootstrap samples). Publication mode may be
computationally expensive because every exact shared distance is tested for
every registered biological comparison.

The workflow uses the dataset selectors in `shared.dataset_selection` with
`fallback_to_all=False`. Current final cohorts are Cav1.2 and Nav1.5
`all_ok_3`, and Kv2.1 `all_ok_3_structural_interface_alignment_qc`.

Outputs are written under `outputs/`; per-comparison caches include the input
paths, comparison ID, run mode, permutation/bootstrap counts, and random seed.
Removing a cache CSV causes that comparison to be recomputed on the next run.

For a quick execution check, set the environment variable
`DISTANCE_STATS_SMOKE_MAX_DISTANCES` to a small integer. Smoke outputs are
written to `outputs/smoke/` so they cannot overwrite complete exploratory or
publication results.
