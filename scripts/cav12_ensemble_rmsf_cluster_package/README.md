# CaV1.2 AlphaFold Ensemble RMSF — Hive Cluster Package

This package performs the expensive, reusable portion of the CaV1.2 structural-ensemble analysis on Hive. It aligns every AlphaFold model once, preserves the complete raw 1–1685 residue axis, and calculates full-protein ensemble RMSF and deviations from three experimental CaV1.2 structures. Plotting, exact masking annotations, structural-region labels, and QC subset selection can be added locally without reopening or realigning the original PDB files.

## Scientific distinction

The primary quantity is **ensemble RMSF**, calculated for each Cα relative to that residue's mean coordinate within one aligned AlphaFold ensemble. These are independent predicted structures, not molecular-dynamics frames.

Experimental structures are not inserted into the ensemble RMSF calculation. They are aligned into the same coordinate frame and used for separate metrics:

- distance between the ensemble mean and each experimental Cα;
- RMS deviation of the full ensemble from each experimental Cα;
- per-model whole-structure and stable-core RMSD to each experiment.

## Configured ensembles

Eight independent production ensembles are configured:

```text
WT vanilla
WT masked
G402S vanilla
G402S masked
G406R vanilla
G406R masked
G490R vanilla
G490R masked
```

Discovery is recursive, so PDB files may be directly inside the protocol directory or inside a nested `models/` directory.

## Experimental references

All references are human CaV1.2 α1 subunits and are mapped to the shortened AlphaFold raw sequence by sequence alignment.

| Reference | CaV1.2 chain | Role |
|---|---:|---|
| 8WE6 | A | Common coordinate anchor and primary high-resolution WT reference; 2.9 Å reported resolution. |
| 8HLP | A | Secondary apo-state WT reference. |
| 8FD7 | K | Secondary gabapentin-complex WT reference. Chain K is the CaV1.2 α1 subunit; chains D and C are auxiliary subunits. |

All mutant ensembles are compared with these WT experimental states. None is labeled as a matched mutant experimental structure.

## Raw numbering and mutation sites

The uploaded example AlphaFold model contains:

```text
chain A
raw residues 1–1685
```

The WT sequence has glycine at raw positions:

```text
G402
G406
G490
```

The package records residue identities for WT, G402S, G406R, and G490R FASTAs in the annotation table.

## Alignment strategy

All models and references are placed in a common 8WE6 coordinate frame. A single rigid transformation is applied to the complete single-chain model.

The alignment core is derived from the established CaV1.2 recycle-convergence stable core, with the mutation-centered IS6 segment removed:

```text
112–381
524–759
901–1145
1160–1197
1239–1323
1370–1534
```

The original convergence core extended through residue 410 in the first block. For this RMSF analysis, residues 382–426 are excluded so motion around G402 and G406 is not fitted away. G490 is already outside the established core.

All raw residues remain in the output, including loops, termini, S4–S5 linkers, pore segments, S6 segments, and mutation-centered regions.

## Package structure

```text
config/cav12_hive.yaml
inspect_cav12_inputs.py
prepare_cav12_references.py
align_cav12_models.py
merge_cav12_alignment_shards.py
calculate_cav12_ensemble_profiles.py
make_pilot_manifest.py
submit_cav12_hive_workflow.sh
scripts/hive/
src/cav12_ensemble_rmsf/
examples/recalculate_qc_subset.sh
tests/
```

## Install or activate the environment

From the package root on Hive:

```bash
micromamba activate bioadri
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
```

An editable installation is optional:

```bash
python -m pip install -e .
```

## 1. Inspect all inputs

```bash
python inspect_cav12_inputs.py \
  --config config/cav12_hive.yaml \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/inspection \
  --max-models-per-dataset 3
```

Review:

```bash
RESULTS=/quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/inspection

column -s, -t < "$RESULTS/cav12_dataset_path_status.csv"
column -s, -t < "$RESULTS/cav12_dataset_counts.csv"
column -s, -t < "$RESULTS/cav12_fasta_status.csv"
column -s, -t < "$RESULTS/cav12_reference_inventory.csv"
```

Expected FASTA length for all four sequence conditions is 1685. Verify that every dataset has the intended number of structures before submitting the full array.

## 2. Prepare references interactively

```bash
python prepare_cav12_references.py \
  --config config/cav12_hive.yaml \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/references
```

Review:

```bash
column -s, -t < \
/quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/references/cav12_reference_alignment_report.csv
```

## 3. Run a balanced pilot

Create two structures from each of the eight datasets:

```bash
python make_pilot_manifest.py \
  --input /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/inspection/cav12_model_manifest.csv \
  --output /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/inspection/cav12_pilot_manifest.csv \
  --models-per-dataset 2
```

Run the pilot:

```bash
PILOT=/quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/pilot_parts
rm -rf "$PILOT"
mkdir -p "$PILOT"

python align_cav12_models.py \
  --config config/cav12_hive.yaml \
  --manifest /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/inspection/cav12_pilot_manifest.csv \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/references/cav12_aligned_references.npz \
  --output-dir "$PILOT" \
  --task-id 0 \
  --task-count 1 \
  --overwrite
```

Check that all 16 models report `alignment_success=True`.

## 4. Submit the complete Hive workflow

The SLURM scripts use:

```text
partition = high
account = genome-center-grp
```

Submit from the package root:

```bash
mkdir -p logs
chmod +x submit_cav12_hive_workflow.sh

./submit_cav12_hive_workflow.sh \
  | tee "logs/submission_$(date +%Y%m%d_%H%M%S).txt"
```

The dependency chain is:

```text
preflight and reference preparation
        ↓ afterok
100-task alignment array, at most 20 simultaneous tasks
        ↓ afterok
merge aligned-coordinate shards
        ↓ afterok
calculate all-model full-protein profiles
```

Monitor with:

```bash
squeue -u "$USER" \
  -o "%.22i %.30j %.10T %.10M %.12l %.8C %.10m %.30R"
```

## Reusable merged outputs

```text
ensemble_rmsf_results/merged/cav12_aligned_ca_coordinates.npy
ensemble_rmsf_results/merged/cav12_aligned_ca_present.npy
ensemble_rmsf_results/merged/cav12_aligned_residue_identities.npy
ensemble_rmsf_results/merged/cav12_alignment_metadata.csv
ensemble_rmsf_results/merged/cav12_failed_models.csv
```

The coordinate matrix has shape:

```text
models × 1685 residues × xyz
```

These files allow RMSF to be recalculated for new QC subsets without repeating structural alignment.

## Full-profile outputs

```text
ensemble_rmsf_results/profiles/cav12_all_models_per_residue_profiles.csv
ensemble_rmsf_results/profiles/cav12_all_models_protocol_vs_vanilla.csv
ensemble_rmsf_results/profiles/cav12_all_models_group_summary.csv
ensemble_rmsf_results/profiles/cav12_all_models_per_model_reference_rmsd.csv
ensemble_rmsf_results/profiles/cav12_all_models_selected_models.csv
```

The per-residue profile contains:

- full-protein ensemble RMSF;
- model count and residue coverage;
- ensemble mean coordinates;
- distance and RMS deviation to 8WE6, 8HLP, and 8FD7;
- WT and mutant FASTA identities;
- mutation-site and mutation-window annotations;
- alignment-core flag.

The protocol comparison file reports masked versus vanilla within each sequence condition, including differences and ratios for RMSF and every experimental-reference metric.

## Recalculate a filtered QC subset later

Use `examples/recalculate_qc_subset.sh` as a template. The merged coordinates are indexed by the original `manifest_index`, so a QC manifest or a pandas query can select `all_ok`, `earliest_converged_selected`, `first100`, final-only models, or any later threshold without reopening the PDB files.

## Important interpretation notes

- A positive masked-minus-vanilla RMSF difference means broader positional variability, not automatically improved prediction.
- Experimental-distance metrics and ensemble RMSF answer different questions and remain separate.
- 8WE6, 8HLP, and 8FD7 are WT experimental states. Comparisons for G402S, G406R, and G490R are therefore state-similarity analyses, not matched mutant validation.
- Exact masking ranges are deliberately deferred and can be merged locally after the expensive cluster stage.
