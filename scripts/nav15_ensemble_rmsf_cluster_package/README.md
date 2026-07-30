# Nav1.5 AlphaFold Ensemble RMSF — Hive Cluster Package

This package performs the expensive, reusable part of the Nav1.5 structural-ensemble analysis on Hive. It aligns every AlphaFold model once, preserves full raw residue coverage, and calculates full-protein ensemble RMSF and experimental-reference deviations. Plotting and final structural-region/masking annotations can be added locally without reopening or realigning the PDB files.

## Scientific interpretation

The primary quantity is **ensemble RMSF**, calculated across aligned independent AlphaFold structures or recycle snapshots:

```text
RMSF_i = sqrt(mean(||r_i,k - mean(r_i)||^2))
```

It measures structural variability across an aligned prediction ensemble. It is not a time-dependent MD fluctuation.

Experimental structures are not included as extra ensemble members. Instead, the package separately reports:

- Distance between the ensemble-mean residue position and each experimental reference.
- RMS deviation of all ensemble coordinates from each experimental reference.
- Whole-model and alignment-core RMSD of every model to every experimental reference.

All structures are placed in one common coordinate frame defined by 6UZ3. No model is locally refitted to individual references after that transformation.

## Default production datasets

The configuration includes seven recursively searched ensembles:

```text
WT vanilla
WT masked
WT masked_v2
WT masked_v2_noIFM
QQQ vanilla
QQQ masked
QQQ masked_v2
```

Recursive discovery supports both layouts:

```text
condition/protocol/*.pdb
condition/protocol/models/*.pdb
```

The legacy/test directories `wt/masked_ifm` and `wt/masked_test` are intentionally excluded. Add them to `config/nav15_hive.yaml` only if they represent distinct production ensembles that should be analyzed.

## Experimental references

| PDB | Package role | Notes |
|---|---|---|
| 6UZ3 | Common coordinate anchor; primary WT reference | Rat Nav1.5 WT construct; sequence-matched to the WT AlphaFold construct over resolved residues. |
| 7FBS | Primary QQQ reference | Rat IFM→QQQ construct. The engineered IFM residues themselves are not resolved in the uploaded coordinates, but the structure is retained as the matched QQQ-state reference. |
| 8T6L | Secondary WT-state reference | Rat BTX-B-bound structure with engineered A33T/G214D substitutions. |
| 7DTC | Cross-species mutant reference | Human Nav1.5 E1784K. It is not treated as a WT- or QQQ-matched experimental structure. |
| 8VYJ | Cross-species WT reference | Full-length human Nav1.5 class I. |
| 8VYK | Cross-species WT reference | Full-length human Nav1.5 class II. |

The package calculates every ensemble against every reference. Interpretation should respect the relationship labels above.

## Raw numbering and alignment core

The uploaded AlphaFold example contains chain A with raw residues `1–1572`. All outputs retain this numbering.

The default rigid-body alignment uses the established Nav1.5 stable-core selection from the recycle-convergence workflow:

```text
132–282
359–426
505–665
690–749
870–1112
1128–1164
1200–1405
1429–1466
```

The complete model is transformed once using this core. All raw residues remain available for RMSF, including loops, termini, and the IFM region.

The raw IFM motif is annotated as:

```text
1169–1171
```

Exact targeted-masking ranges are deliberately deferred. They are not needed for the expensive alignment or full-protein RMSF calculations and can be merged into the resulting CSV files later.

## Package layout

```text
config/nav15_hive.yaml
inspect_nav15_inputs.py
prepare_nav15_references.py
align_nav15_models.py
merge_nav15_alignment_shards.py
calculate_nav15_ensemble_profiles.py
make_pilot_manifest.py
submit_nav15_hive_workflow.sh
scripts/hive/
src/nav15_ensemble_rmsf/
examples/recalculate_qc_subset.sh
```

## Cluster installation

Move and extract the package under the repository scripts directory, for example:

```bash
cd /quobyte/yarovoygrp/ahgz/vgic_mutants/scripts
unzip nav15_ensemble_rmsf_cluster_package.zip
cd nav15_ensemble_rmsf_cluster_package
```

Activate the existing environment and expose the local package:

```bash
micromamba activate bioadri
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

An editable pip installation is optional:

```bash
python -m pip install -e .
```

## 1. Inspect real cluster paths and counts

```bash
python inspect_nav15_inputs.py \
  --config config/nav15_hive.yaml \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/inspection \
  --max-models-per-dataset 3
```

Review:

```bash
RESULTS=/quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/inspection
column -s, -t < "$RESULTS/nav15_dataset_path_status.csv"
column -s, -t < "$RESULTS/nav15_dataset_counts.csv"
column -s, -t < "$RESULTS/nav15_fasta_status.csv"
column -s, -t < "$RESULTS/nav15_reference_inventory.csv"
```

Expected FASTA length for WT and QQQ is `1572`. Correct the YAML before submitting if a dataset path has zero structures or a FASTA has an unexpected length.

## 2. Prepare references interactively

This is the exact preflight operation that validates experimental mapping and the alignment core:

```bash
python prepare_nav15_references.py \
  --config config/nav15_hive.yaml \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references
```

Review:

```bash
column -s, -t < \
/quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references/nav15_reference_alignment_report.csv
```

## 3. Recommended balanced pilot

```bash
python make_pilot_manifest.py \
  --input /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/inspection/nav15_model_manifest.csv \
  --output /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/inspection/nav15_pilot_manifest.csv \
  --models-per-dataset 2
```

```bash
PILOT=/quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/pilot_parts
rm -rf "$PILOT"
mkdir -p "$PILOT"

python align_nav15_models.py \
  --config config/nav15_hive.yaml \
  --manifest /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/inspection/nav15_pilot_manifest.csv \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references/nav15_aligned_references.npz \
  --output-dir "$PILOT" \
  --task-id 0 \
  --task-count 1 \
  --overwrite
```

Check pilot success:

```bash
python - <<'PY'
import pandas as pd
path = "/quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/pilot_parts/aligned_part_0000_metadata.csv"
df = pd.read_csv(path)
print(df["alignment_success"].value_counts(dropna=False))
print(df.groupby("dataset")["alignment_success"].agg(total="size", successful="sum"))
print(df.loc[df["alignment_success"] != True, ["dataset", "pdb_file", "alignment_error"]])
PY
```

## 4. Submit the dependent Hive workflow

The included SLURM files use:

```text
partition: high
account: genome-center-grp
environment: bioadri
```

Submit:

```bash
mkdir -p logs
chmod +x submit_nav15_hive_workflow.sh
./submit_nav15_hive_workflow.sh \
  | tee "logs/submission_$(date +%Y%m%d_%H%M%S).txt"
```

The wrapper submits:

```text
preflight/reference preparation
  ↓ afterok
100-task alignment array, at most 20 simultaneous tasks
  ↓ afterok
merge aligned-coordinate shards
  ↓ afterok
full all-model profiles
```

## Main outputs

Reusable aligned data:

```text
ensemble_rmsf_results/merged/nav15_aligned_ca_coordinates.npy
ensemble_rmsf_results/merged/nav15_aligned_ca_present.npy
ensemble_rmsf_results/merged/nav15_aligned_residue_identities.npy
ensemble_rmsf_results/merged/nav15_alignment_metadata.csv
ensemble_rmsf_results/merged/nav15_failed_models.csv
```

Full-protein profile outputs:

```text
ensemble_rmsf_results/profiles/nav15_all_models_per_residue_profiles.csv
ensemble_rmsf_results/profiles/nav15_all_models_protocol_vs_vanilla.csv
ensemble_rmsf_results/profiles/nav15_all_models_group_summary.csv
ensemble_rmsf_results/profiles/nav15_all_models_per_model_reference_rmsd.csv
ensemble_rmsf_results/profiles/nav15_all_models_selected_models.csv
```

The per-residue table contains one row per dataset and raw residue and includes:

```text
ensemble RMSF
coverage
ensemble mean x/y/z
mean distance to 6UZ3, 7FBS, 8T6L, 7DTC, 8VYJ, and 8VYK
RMS deviation to every experimental reference
WT and QQQ residue identities
alignment-core flag
IFM-motif flag
```

## Recalculate filtered subsets without realigning PDB files

After the expensive array and merge complete, use the existing QC manifests to recalculate profiles locally or on Hive:

```bash
python calculate_nav15_ensemble_profiles.py \
  --config config/nav15_hive.yaml \
  --merged-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/merged \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references/nav15_aligned_references.npz \
  --annotations /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references/nav15_residue_annotations.csv \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/profiles \
  --subset-name earliest_converged \
  --subset-manifest /path/to/qc_manifest.csv \
  --subset-column earliest_converged_selected \
  --subset-key pdb_file
```

This supports `all_ok`, `earliest_converged_selected`, `first100`, final-only queries, or revised thresholds without rerunning structural alignment.
