# Kv2.1 tetramer ensemble-RMSF cluster package

This package calculates **per-residue structural variability across aligned AlphaFold2 ensembles** for Kv2.1. The structures are independent predictions and recycle snapshots, not molecular-dynamics frames. The primary measurement is therefore called **ensemble RMSF** or **per-residue structural variability**.

The package is focused on these six groups:

- WT vanilla
- WT masked
- L403A vanilla
- L403A masked
- F412L vanilla
- F412L masked

Every group is analyzed across the complete raw AlphaFold sequence, residues **1–600**, for all four subunits. Both experimental structures, **8SD3** and **8SDA**, are mapped into the same coordinate frame and evaluated against every group.

## Scientific design

### Ensemble RMSF

For each chain and raw residue position, the package calculates

```text
RMSF_i = sqrt(mean(||r_i,k - mean(r_i)||^2))
```

where `k` indexes the aligned structures included in the selected group or QC subset. RMSF is measured relative to the **ensemble mean coordinate**, not relative to an experimental structure.

### Experimental references

An experimental structure is not inserted as one extra member of an AlphaFold ensemble. Doing so would give it an arbitrary weight determined by the number of predictions. Instead, the package calculates separate quantities:

- distance from the ensemble mean coordinate to 8SD3 or 8SDA;
- RMS deviation of all ensemble coordinates from 8SD3 or 8SDA;
- whole-model matched-Cα RMSD to both references.

This makes it possible to distinguish a tightly clustered ensemble that is far from experiment from a broader ensemble that samples coordinates closer to an experimental state.

### Common coordinate frame

1. 8SD3 is the canonical coordinate anchor.
2. 8SDA is aligned once to 8SD3.
3. Every AlphaFold tetramer is aligned once to the 8SD3 frame.
4. The geometric pore-ring order is inferred from the four subunit centroids because alphabetical PDB chain order is not reliable.
5. Four cyclic shifts are tested in both ring traversal directions, giving eight adjacency-preserving mappings without permitting arbitrary chain scrambling.
6. One rigid transformation is calculated from all four subunits together.
7. The same transform is applied to the complete tetramer.
8. Original model chain labels are canonicalized to the matched 8SD3 chain labels.

Chains are never aligned independently for the principal analysis.

### Alignment core

The default alignment core uses raw model positions:

```text
184–208
231–247
262–277
```

These are conservative, experimentally covered S1–S3 transmembrane segments. The selection excludes all three directly masked blocks, the pore, S6, and unresolved experimental loops. It is deliberately narrower than the earlier 184–422 convergence region so that pore and S6 variability are not fitted away.

### Mask annotations

The supplied configuration records the mask blocks inferred from the provided masked query sequence:

```text
288–328
370–384
401–417
```

The output keeps raw model numbering. Biological aliases and more detailed region labels can be added later without rerunning the expensive alignment.

## Important correction from the previous RMSD code

The earlier code title-cased residue names before looking them up in Biopython's uppercase three-letter dictionary. Ordinary residues such as `ALA`, `GLY`, and `LEU` were therefore converted to `X`. This package fixes the conversion and performs real sequence verification.

## Package layout

```text
kv21_ensemble_rmsf_cluster_package/
├── config/kv21_hive.yaml
├── src/kv21_ensemble_rmsf/
├── scripts/hive/
├── inspect_kv21_inputs.py
├── prepare_kv21_references.py
├── align_kv21_models.py
├── merge_kv21_alignment_shards.py
├── calculate_kv21_ensemble_profiles.py
├── make_pilot_manifest.py
├── submit_kv21_hive_workflow.sh
└── tests/
```

## Dependencies

The workflow requires:

```text
Python 3.10+
NumPy
pandas
Biopython
PyYAML
```

The Slurm scripts activate the existing `bioadri` micromamba environment. Test it with:

```bash
micromamba activate bioadri
python - <<'PY'
import numpy, pandas, yaml, Bio
print("NumPy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("PyYAML:", yaml.__version__)
print("Biopython:", Bio.__version__)
PY
```

No internet connection is required.

## 1. Copy and unpack on Hive

From the directory where the ZIP was uploaded:

```bash
cd /quobyte/yarovoygrp/ahgz/vgic_mutants
unzip kv21_ensemble_rmsf_cluster_package.zip
cd kv21_ensemble_rmsf_cluster_package
```

The default configuration assumes the package is separate from the `Kv2.1` model directory and writes results to:

```text
/quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results
```

## 2. Verify the Hive account once

The supplied Slurm files use:

```bash
#SBATCH --partition=high
#SBATCH --account=genome-center-grp
```

Confirm the exact account spelling on Hive before submission:

```bash
sacctmgr show assoc user="$USER" format=User,Account,Partition
```

Change the `--account` line in the four files under `scripts/hive/` only if Hive reports a different spelling.

## 3. Inspect the configuration

Open:

```bash
nano config/kv21_hive.yaml
```

The only path that may require immediate adjustment is WT vanilla. The current default is:

```text
/quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/wt/models
```

Your directory listing also showed `wt/vanilla`. The preflight report will count both only if you inspect them manually; it will never merge them silently. Before production, confirm which contains the intended WT vanilla structures:

```bash
find /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/wt/models -type f -name '*.pdb' | wc -l
find /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/wt/vanilla -type f -name '*.pdb' | wc -l
```

Edit the one `wt_vanilla.path` line if needed.

## 4. Run preflight interactively first

```bash
cd /quobyte/yarovoygrp/ahgz/vgic_mutants/kv21_ensemble_rmsf_cluster_package

micromamba activate bioadri
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

python inspect_kv21_inputs.py \
  --config config/kv21_hive.yaml \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/inspection \
  --max-models-per-dataset 3

python prepare_kv21_references.py \
  --config config/kv21_hive.yaml \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/references
```

Review these files before the full array:

```text
inspection/kv21_dataset_counts.csv
inspection/kv21_dataset_path_status.csv
inspection/kv21_fasta_status.csv
inspection/kv21_structure_inventory.csv
inspection/kv21_reference_inventory.csv
references/kv21_reference_alignment_report.csv
references/kv21_residue_annotations.csv
```

The inventory should show four mapped chains and approximately 600 mapped Cα positions per model chain. The reference report should show a valid 8SDA-to-8SD3 fit.

## 5. Recommended pilot

Create a 2-model-per-dataset pilot manifest:

```bash
python make_pilot_manifest.py \
  --input /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/inspection/kv21_model_manifest.csv \
  --output /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/inspection/kv21_pilot_manifest.csv \
  --models-per-dataset 2
```

Run the 12 pilot structures in one process:

```bash
mkdir -p /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/pilot_parts

python align_kv21_models.py \
  --config config/kv21_hive.yaml \
  --manifest /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/inspection/kv21_pilot_manifest.csv \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/references/kv21_aligned_references.npz \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/pilot_parts \
  --task-id 0 \
  --task-count 1 \
  --overwrite
```

Inspect:

```text
pilot_parts/aligned_part_0000_metadata.csv
```

Important columns include:

```text
alignment_success
alignment_error
best_cyclic_shift
chain_mapping
best_mapping_core_ca_rmsd_A
mapping_rmsd_gap_A
core_coverage
whole_matched_ca_rmsd_to_8SD3_A
whole_matched_ca_rmsd_to_8SDA_A
```

## 6. Submit the complete workflow

After preflight and pilot look correct:

```bash
./submit_kv21_hive_workflow.sh
```

This submits four dependent jobs:

1. preflight and reference preparation;
2. 100-task model-alignment array;
3. coordinate-shard merge;
4. full-protein all-model RMSF and experimental-reference profiles.

The alignment array uses a maximum of 20 simultaneous tasks by default:

```text
#SBATCH --array=0-99%20
```

Each structure is assigned deterministically by `manifest_index % 100`. Completed shard files are skipped safely when the array is resubmitted.

## 7. Main outputs

### Model inventory and alignment metadata

```text
inspection/kv21_model_manifest.csv
merged/kv21_alignment_metadata.csv
merged/kv21_failed_models.csv
merged/kv21_merge_summary.json
```

The metadata retains dataset, seed, model number, rank, recycle index, final/recycle status, trajectory ID, cyclic mapping, alignment RMSD, coverage, and model-level reference RMSDs.

### Reusable aligned coordinates

```text
merged/kv21_aligned_ca_coordinates.npy
merged/kv21_aligned_ca_present.npy
merged/kv21_aligned_residue_identities.npy
merged/kv21_raw_residue_numbers.npy
merged/kv21_canonical_chains.npy
```

The coordinate array shape is:

```text
models × 4 canonical chains × 600 raw residues × xyz
```

These arrays are the expensive reusable result. New QC subsets can be evaluated without reopening or realigning PDB files.

### Full-protein profiles

```text
profiles/kv21_all_models_chain_resolved_profiles.csv
profiles/kv21_all_models_chain_resolved_masked_vs_vanilla.csv
profiles/kv21_all_models_symmetry_averaged_profiles.csv
profiles/kv21_all_models_masked_vs_vanilla_comparisons.csv
profiles/kv21_all_models_per_model_reference_rmsd.csv
profiles/kv21_all_models_whole_protein_and_mask_summary.csv
profiles/kv21_all_models_selected_models.csv
```

The chain-resolved profile contains one row per:

```text
dataset × chain × raw residue
```

Key columns include:

```text
ensemble_rmsf_A
coverage_fraction
ensemble_mean_x_A
ensemble_mean_y_A
ensemble_mean_z_A
mean_coordinate_distance_to_8SD3_A
rms_deviation_to_8SD3_A
mean_coordinate_distance_to_8SDA_A
rms_deviation_to_8SDA_A
reference_8SD3_pdb_residue_number
reference_8SDA_pdb_residue_number
reference_8SD3_source_chain
reference_8SDA_source_chain
directly_masked
mask_names
mask_category
```

The symmetry-averaged profile reports the mean of the four chain-resolved RMSFs and the chain-to-chain RMSF standard deviation. It does not pool coordinates from physically different subunit positions.

The masked-versus-vanilla file includes:

```text
masked_minus_vanilla_rmsf_A
masked_divided_by_vanilla_rmsf
masked_minus_vanilla_mean_coordinate_distance_to_8SD3_A
masked_minus_vanilla_mean_coordinate_distance_to_8SDA_A
```

For experimental-distance differences, a negative value means the masked ensemble mean is closer to that experimental reference than the vanilla ensemble mean.

## 8. Coverage rule

The default minimum per-residue coverage is 0.80. RMSF is written as missing when fewer than 80% of selected aligned models contain that Cα. Counts and coverage remain in the CSV, so the threshold can be audited or changed later.

Change it in the YAML or rerun only the profile step:

```bash
python calculate_kv21_ensemble_profiles.py \
  ... \
  --minimum-residue-coverage 0.90
```

No PDB realignment is required.

## 9. Recalculate profiles for QC subsets later

The package intentionally aligns all structures before applying convergence filters.

### Example: `all_ok`

```bash
python calculate_kv21_ensemble_profiles.py \
  --config config/kv21_hive.yaml \
  --merged-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/merged \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/references/kv21_aligned_references.npz \
  --annotations /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/references/kv21_residue_annotations.csv \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/profiles/all_ok \
  --subset-name all_ok \
  --subset-manifest /path/to/qc_manifest.csv \
  --subset-column all_ok \
  --subset-key pdb_file
```

### Example: `earliest_converged_selected`

Use the same command with:

```text
--subset-name earliest_converged
--subset-column earliest_converged_selected
```

### Example: final structures only

```text
--subset-name final_models
--subset-query "is_final_model == True"
```

### Example: first 100 generated acceptable models

Use the corresponding Boolean column from the previously generated QC manifest. The profile calculation changes the ensemble mean and RMSF after selection; therefore filtering cannot be performed by deleting rows from a previously aggregated RMSF CSV. It must be rerun from the saved aligned-coordinate arrays, which is fast compared with PDB alignment.

## 10. Copy compact outputs to the Mac later

The plotting stage is intentionally not included yet. The compact CSV outputs can be copied locally with:

```bash
rsync -avh \
  hive:/quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/profiles/ \
  ./Kv21_ensemble_rmsf_profiles/
```

Copy the `.npy` aligned-coordinate arrays only if local recalculation of new subsets is desired; the profile CSV files are sufficient for plotting existing subsets.

## 11. Tests

From the package root:

```bash
micromamba activate bioadri
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
pytest -q
```

The tests cover the corrected amino-acid conversion, rigid-body fitting, cyclic chain mappings, and mask selection.

## Interpretation cautions

- Higher RMSF means broader positional variability, not automatically a better prediction.
- Lower distance to experiment means greater similarity to that reference, not proof that the full structure is correct.
- Recycle snapshots from one seed are correlated. The `all_models` profile intentionally captures both between-seed variability and within-trajectory recycle evolution.
- Independent-seed interpretation should use one selected structure per trajectory, such as `earliest_converged_selected`.
- F412L has no matched experimental structure in this package. Its comparisons to 8SD3 and 8SDA are exploratory references, not an F412L experimental validation.
