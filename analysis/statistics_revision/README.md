# Trajectory-aware statistics revision

## Scope completed

- Read and visually inspected the complete 26-page manuscript draft.
- Tested a canonical parser for rank, AlphaFold model, seed, and recycle naming patterns.
- Treated model-seed trajectories as independent units.
- Generated trajectory-bootstrap confidence intervals, equal-trajectory estimates, and earliest/latest retained-snapshot sensitivities for the primary L403A, F412L, NaV1.5, G402S, and G406R coordinates available with exact source mappings.
- Generated raw and severe-overlap-filtered G406R analyses.
- Added repeated trajectory subsampling for L403A and all three F412L contacts.

## Important discrepancies and unresolved items

1. The 12.84 A L403A threshold is hard-coded in `kv21/check_L403A_E423_N179_extremes.py`; no independent derivation was found. Figure S3 therefore includes threshold sensitivity from 11.5 to 15.0 A.
2. The original WT masked NaV1.5 IFM CSV lacks the six gate-span and complete-motif terminal-distance columns. Those metrics were marked unavailable rather than borrowed from a different mask design.
3. The exact trajectory-resolved input behind the proposed NaV1.5 regional-RMSD Figure S5 could not be resolved locally: the nominal compressed OK3 table contains a nested Git LFS pointer. Figure S5 was not fabricated from snapshot-level summary tables.
4. Exact mapping-QC counts are not present in the convergence manifests; Table S2 marks this stage unresolved instead of equating convergence with mapping.
5. The manuscript F412L representative (rank 037, model 4, seed 103, r1) has `all_ok=True` and is the `earliest_converged_selected` row, but `all_ok_3=False`; it is absent from the corrected v5 final-QC contact table, which retains later snapshots from that trajectory. The draft must not state that r1 passed the corrected final-QC set. No pre-registered repository rule defining all equivalent candidates was found.
6. The traced L403A analysis table contains 3,910 masked and 4,403 vanilla snapshots, versus 4,073 and 4,521 in Table S1. On the traced table, the >=12.84 A any-shifted fractions are 12.69% masked and 0.50% vanilla, not the draft's 9.94% and 0.02%. This denominator/source mismatch must be resolved before submission.
7. Repeated trajectory-subsampling was completed for L403A, all three F412L contacts, NaV1.5 QQQ motif-receptor separation, G402S S402-I1523 proximity, and clash-filtered G406R acidic-partner proximity.

## Interpretation

Snapshot-level percentages remain useful descriptions of geometries sampled by each prediction protocol. They are not independent-replicate estimates or thermodynamic populations. Main-text claims should pair those descriptive values with model-seed trajectory counts and trajectory-bootstrap confidence intervals.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/vgci-matplotlib conda run -n bioadri python analysis/statistics_revision/scripts/run_all_statistics.py --repo-root <REPO_ROOT> --output-dir analysis/statistics_revision --seed 20260803
MPLCONFIGDIR=/tmp/vgci-matplotlib conda run -n bioadri python analysis/statistics_revision/scripts/complete_remaining_outputs.py --repo-root <REPO_ROOT> --output-dir analysis/statistics_revision --seed 20260803
```

The full execution record is in `logs/run_all_statistics.log`. No original notebook, source CSV, figure, or manuscript file was overwritten.
