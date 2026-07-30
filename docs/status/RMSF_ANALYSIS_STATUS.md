# Ensemble RMSF analysis status

Final quantitative RMSF profiles are recomputed from the allOK3-selected ensembles. Direct-mask
annotations use the supplied authoritative 1-based raw AlphaFold query/model positions; production
A3M provenance remains documented separately in [A3M_PROVENANCE.md](A3M_PROVENANCE.md).

## Kv2.1

The primary symmetry-averaged profile is:

`kv21/dataRMSF/profiles/kv21_all_ok_3_symmetry_averaged_profiles.csv`

WT, L403A, and F412L each have paired vanilla and masked profiles. Chain-resolved profiles remain
available for subunit-asymmetry checks. The expected direct mask contains 73 positions spanning
288–328, 370–384, and 401–417.

## Nav1.5

The primary profile is:

`nav15/dataRMSF/profiles/nav15_all_ok_3_per_residue_profiles.csv`

The analysis includes WT vanilla, standard masked, masked v2, masked v2 no-IFM, QQQ vanilla,
standard masked, and masked v2. The direct-mask definition explicitly distinguishes protocols that
mask the 1164–1176 IFM region from `masked_v2_noIFM`.

## CaV1.2

The primary profile is:

`cav12/dataRMSF/profiles/cav12_all_ok_3_per_residue_profiles.csv`

WT, G402S, G406R, and G490R each have paired vanilla and masked profiles. G490R remains
exploratory/supplemental in the paper narrative, but it is available for RMSF analysis. The expected
mask counts are 263, 272, 268, and 274 positions, respectively.

## Reproducibility

The recomputation procedure is documented in
[`scripts/ensemble_rmsf_analysis/ALL_OK3_RECOMPUTE.md`](../../scripts/ensemble_rmsf_analysis/ALL_OK3_RECOMPUTE.md).
RMSF is calculated across aligned independent predictions, not molecular-dynamics frames. Increased
RMSF means broader positional variability and is not automatically evidence of improved modeling.
