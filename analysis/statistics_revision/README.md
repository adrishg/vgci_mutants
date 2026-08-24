# Seed-block analysis of voltage-gated ion-channel ensembles

## Paired-seed v2 revision (authoritative manuscript analysis)

The current manuscript-facing revision is in `paired_seed_v2/`; the earlier
`seed_block/` directory remains unchanged for provenance. The v2 primary
estimand jointly resamples recorded nominal seed labels while preserving each
condition's QC-qualified survivor set. Common-contributing-seed and common
model-seed survivor analyses are reported separately as sensitivities. Actual
AlphaFold RNG values could not be recovered from the available run metadata,
so overlap of numeric seed labels is explicitly marked as unverified rather
than asserted to be confirmed random-seed pairing.

Continuous outcomes use trajectory medians, equal surviving AF2-model weights
within seed, and equal seed weights. Categorical outcomes use named trajectory
fractions or earliest/latest summaries. QC retention uses all five nominal
model strata per condition-specific seed set, with missing/failed trajectories
contributing zero. Geometry among survivors and QC-adjusted target yield are
kept separate. Sampling frequencies are protocol frequencies, not equilibrium
occupancies.

Publication commands:

```bash
python analysis/statistics_revision/scripts/run_paired_seed_v2.py --mode publication
python analysis/statistics_revision/scripts/run_paired_seed_full_panel.py --mode publication
MPLCONFIGDIR=/tmp/vgci_mpl python analysis/statistics_revision/scripts/run_paired_seed_breadth_rmsf.py --mode publication
python analysis/statistics_revision/scripts/run_paired_seed_reduced_depth.py
python analysis/statistics_revision/scripts/run_qc_adjusted_target_yields.py --mode publication
```

Exact mask IDs and statuses are registered in `docs/MASK_REGISTRY.tsv`.
Authoritative narrative values are in
`paired_seed_v2/manuscript_numbers.csv`; unresolved FASTA/A3M and seed
provenance is documented in `paired_seed_v2/UNRESOLVED_BLOCKERS.md`.

This directory contains the publication-scale statistical analysis of the
Kv2.1, Nav1.5, and Cav1.2 AlphaFold2 ensembles. The analysis separates
prediction-run variability, quality-control retention, and structural geometry
so that recycle snapshots and AlphaFold2 model parameterizations are not
treated as independent biological observations.

## Statistical design

The input random seed is the resampling unit. Retained recycles are reduced
within each seed-model trajectory, available AlphaFold2 model
parameterizations receive equal weight within a seed, and seeds receive equal
weight between conditions. Primary continuous outcomes use the median retained
recycle. Categorical outcomes use a named within-trajectory fraction or an
earliest/latest retained-recycle summary.

The publication run used 2,000 whole-seed bootstrap replicates. The focal
L403A Wasserstein test used 9,999 whole-seed permutations. Leave-one-AF2-model-
out estimates accompany the focal continuous result and the complete distance
panel.

Geometry is estimated from all analysis-qualified survivors in each condition.
Paired common-survivor calculations are retained as sensitivity analyses
because survival through quality control depends on prediction protocol.
Snapshot-level percentages and percentiles describe structures sampled by the
protocol; they are not thermodynamic occupancies.

## Cohort and quality control

`seed_block/master_structure_cohort.csv` contains one row per generated
structure with channel, sequence, protocol, seed, AF2 model, recycle, mapping,
convergence, final-QC, analysis membership, and exclusion information. All
125,994 manifest structures have an alignment-mapping record and pass the
recorded mapping checks.

Mapping, convergence, final structural/interface QC, and analysis-specific
availability are separate cohort stages. This distinction is most visible in
Kv2.1, where the interface and coordinate-availability filters reduce the
analysis cohort after final QC. Masked-minus-vanilla analysis-final trajectory
retention differences were -13.6 percentage points for WT (95% CI -16.8 to
-10.4), -10.6 for L403A (-13.8 to -7.2), and -12.2 for F412L (-15.6 to -8.6).

## Kv2.1 L403A

The primary coordinate is the maximum E423-N179 Cα distance across the four
subunits. Targeted masking increased the seed-balanced mean of trajectory
medians from 10.237 to 11.298 Å, a difference of 1.061 Å (95% CI 0.960 to
1.166 Å). Seed-balanced W1 was 1.061 Å (95% CI 0.961 to 1.161 Å;
permutation p = 0.0001).

The secondary shifted-interface threshold is 12.8413 Å, the midpoint between
the maximum experimental WT 8SD3 distance (11.5083 Å) and the lower shifted
experimental L403A 8SDA distance (14.1744 Å). The seed-balanced shifted
fraction increased from 0.605% to 12.951%, an absolute difference of 12.346
percentage points (95% CI 10.040 to 14.892). The legacy ratio of seed-balanced protocol sampling fractions was 21.4, with a
wide 95% CI of 10.0 to 209.5 because the vanilla frequency was close to zero.

At this threshold, 10.755% of masked observations had all four interfaces
shifted, whereas 0.836% had two shifted interfaces. The threshold-positive
masked ensemble therefore predominantly sampled a symmetric four-interface
extension rather than the two-shifted/two-WT-like pattern in 8SDA. The ordered
four-distance RMSE to 8SDA decreased by 0.740 Å (95% CI -0.795 to -0.684 Å),
supporting partial geometric movement without recovery of the complete
experimental conformational program.

The formal masking-by-L403A interaction for the maximum distance was +0.106 Å
(95% CI -0.035 to +0.248 Å).

## Kv2.1 F412L

Masking redistributed the three L412-centered contacts in different
directions. Seed-balanced within-4-Å fractions changed from 0.513% to 13.929%
for L412-L316 (difference +13.416 percentage points; 95% CI +11.564 to
+15.321), remained 0% for L412-L329, and changed from 95.774% to 84.097% for
L412-L403 (difference -11.677 percentage points; 95% CI -14.026 to -9.346).
The ensemble therefore supports contact-specific redistribution rather than a
uniformly expanded hydrophobic pocket.

The objective representative recorded in
`seed_block/f412l_objective_representative_selection.csv` is
`kv21_f412l_masked_unrelaxed_rank_376_alphafold2_multimer_v3_model_3_seed_093.r10.pdb`.
Its three audited distances are 4.515, 6.711, and 4.038 Å, and its minimum
L412-centered local distance is 4.4 Å. Selection used the latest final-QC
snapshot per trajectory, a 2-Å local-overlap screen, three-contact weakening,
and robust distance to the eligible subpopulation median. The coordinate file
itself is not present in the local repository, so the record is numerical
rather than a regenerated structure rendering.

## Nav1.5

For vanilla QQQ, the snapshot-level association between QQQ-receptor
separation and maximum gate span was weak (Spearman rho = 0.134) and changed
across trajectory reductions: earliest rho = -0.157 (95% CI -0.239 to
-0.081), latest rho = 0.014 (-0.055 to 0.079), and trajectory-median rho =
0.147 (0.068 to 0.229). Motif displacement was therefore not consistently
coupled to progressive gate opening.

The regional RMSD supplement was regenerated from the intact aligned Cα arrays
and final-QC manifest using four fixed regions: 206 S5/S6 pore-helix Cα atoms,
29 DII-S6 atoms, the three-residue IFM/QQQ motif, and a six-residue receptor
set. The compact table contains 34,998 structures across six experimental
references. Stable-core validation agrees with the stored alignment RMSDs to a
maximum absolute difference of 5.213e-7 Å.

Relative to 8VYJ, the original WT mask changed DII-S6 RMSD by -0.443 Å (95% CI
-0.976 to -0.247), IFM-motif RMSD by +16.376 Å (16.102 to 16.895), and
receptor-set RMSD by +0.353 Å (0.320 to 0.399). Relative to 7FBS, the original
QQQ mask changed pore-helix RMSD by -0.101 Å (-0.193 to -0.016), DII-S6 RMSD
by -0.862 Å (-1.197 to -0.750), and receptor-set RMSD by +0.241 Å (0.228 to
0.257). The 7FBS motif cell is unavailable because the corresponding linker
coordinates are unresolved in that reference.

The original historical regional table is represented by a nested Git-LFS
pointer whose object is absent from the server. The regenerated region
definitions and values are the traceable analysis source.

## Cav1.2

Within G402S, masking altered the categorical nearest DIV-S6 partner to S402.
I1523 decreased from 60.760% to 47.691%, M1524 increased from 4.840% to
35.062%, and V1520 increased from 0% to 12.240%. These values describe the
masking response within the G402S sequence background. The persisted
categorical table does not include a WT-versus-G402S sequence contrast.

For G406R, the seed-balanced fraction of snapshots passing the R406-centered
local-overlap criterion decreased from 67.420% to 21.887%, a difference of
-45.533 percentage points (95% CI -47.400 to -43.720). The fraction of
trajectories with any locally valid snapshot decreased from 95.0% to 51.6%
(difference -43.4 percentage points; 95% CI -47.0 to -39.6). Among locally
valid survivors, R406-D1528 proximity changed from 21.625% to 4.236%, and
R406-D1533 proximity changed from 33.070% to 25.200%. These conditional
frequencies describe the surviving local geometries rather than unconditional
state populations.

## Reduced-depth sensitivity

The repeated common-seed analysis concerns the L403A maximum-distance and
shifted-interface outcomes. Across 1,000 draws of 20 shared seeds, equivalent
to 100 nominal seed-model trajectories per protocol, both effects retained
their complete-ensemble direction in every draw. Median relative error was
6.9% for the continuous distance and 14.1% for the shifted fraction. Subset
intervals covered the complete-ensemble effects in 94.4% and 95.1% of draws,
respectively, and the masked rare geometry was observed in every rare-state
draw. These results characterize retrospective stability of the two L403A
outcomes rather than a general sampling-depth rule for all channels.

## Complete structural-distance panel

`seed_block/full_panel/` contains 35,602 point estimates across 37 registered
comparisons. W1, weighted medians, weighted IQRs, pooled weighted IQRs, and
normalized W1 use the same seed/model weights. The panel is an effect-size
discovery analysis without mass-univariate p/q values. Prespecified focal
coordinates carry their own whole-seed intervals and permutation tests.

Kv2.1 uses a shared mask and supports sequence-by-masking contrasts. Nav1.5
and Cav1.2 use condition-specific mask designs, so their vanilla WT-mutant
comparisons are the primary sequence contrasts and masked WT-mutant cells are
protocol-specific comparisons.

## Principal artifacts

- `seed_block/master_structure_cohort.csv`: per-structure cohort and exclusions
- `seed_block/master_cohort_flow_summary.csv`: cohort-stage counts
- `seed_block/qc_retention_seed_block.csv`: seed-block retention estimates
- `seed_block/l403a_seed_block_contrasts.csv`: focal L403A effects
- `seed_block/f412l_direct_seed_block_contrasts.csv`: F412L contact effects
- `seed_block/nav15_qqq_seed_block_correlation_sensitivity.csv`: QQQ correlation sensitivities
- `seed_block/nav15_regional_rmsd/`: regenerated regional RMSD tables and Figure S5
- `seed_block/cav12_g402s_nearest_partner_seed_block.csv`: G402S partner categories
- `seed_block/cav12_g406r_local_validity_seed_block.csv`: G406R local validity
- `seed_block/first100_repeated_common_seed_summary.csv`: L403A reduced-depth summary
- `seed_block/full_panel/`: complete all-distance effect panel and overview figure

## Reproduction

```bash
PYTHONPATH=. python analysis/statistics_revision/scripts/run_seed_block_revision.py --mode publication
PYTHONPATH=. python analysis/statistics_revision/scripts/run_seed_block_distance_panel.py
PYTHONPATH=. python analysis/statistics_revision/scripts/run_nav15_regional_rmsd.py --mode publication
PYTHONPATH=. pytest -q tests/test_seed_block_statistics.py tests/test_distribution_statistics.py tests/test_nav15_regional_rmsd.py
```

The scripts write additive analysis products beneath
`analysis/statistics_revision/seed_block/`; source coordinates, distance
tables, notebooks, and manuscript files are not overwritten.
