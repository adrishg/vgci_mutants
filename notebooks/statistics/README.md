# Distance-distribution statistics

`01_distance_distribution_statistics.ipynb` analyzes existing distance CSVs
without reading PDB/CIF coordinates or recalculating structural distances.

The independent sampling block is an AlphaFold trajectory identified by
`(seed, model_number)`. Retained recycle snapshots remain within that block.
The primary effect size is trajectory-balanced 1-Wasserstein distance in Å;
paired permutation tests and paired bootstraps operate on whole trajectory
pairs. A trajectory-median sensitivity analysis is reported beside the
weighted full-snapshot analysis.

Run the notebook from the repository root. Its configuration cell selects
`exploratory` (999 permutations, 500 bootstrap samples) or `publication`
(9,999 permutations, 2,000 bootstrap samples). Publication mode may be
computationally expensive because every exact shared distance is tested for
every registered biological comparison.

The workflow uses the dataset selectors in `shared.dataset_selection` with
`fallback_to_all=False`. Current final cohorts are Cav1.2 and Nav1.5
`all_ok_3`, and Kv2.1 `all_ok_3_structural_interface_alignment_qc`.

Outputs are written under `outputs/`; per-comparison caches include the input
paths, comparison ID, run mode, permutation/bootstrap counts, and random seed.
Delete a cache CSV only when intentionally forcing that comparison to rerun.

For a quick execution check, set the environment variable
`DISTANCE_STATS_SMOKE_MAX_DISTANCES` to a small integer. Smoke outputs are
written to `outputs/smoke/` so they cannot overwrite complete exploratory or
publication results.
