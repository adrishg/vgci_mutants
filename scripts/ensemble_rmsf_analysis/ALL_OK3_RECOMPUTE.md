# All-OK-3 RMSF recomputation

Final-QC RMSF profiles can be recomputed locally from the aligned, memory-mapped
coordinate arrays without regenerating structural alignments:

```bash
python scripts/ensemble_rmsf_analysis/recompute_all_ok3_local.py --channel all
```

The local calculation streams bounded coordinate chunks, joins selection keys
strictly by dataset and PDB basename, and validates its RMSF formula against an
existing complete all-model profile before writing any final profile. Kv2.1
uses the stricter `all_ok_3_structural_interface_qc` distance allowlists;
Nav1.5 and CaV1.2 use their `all_ok_3` selection manifests.

The notebooks automatically prefer these final profiles:

- `kv21_all_ok_3_symmetry_averaged_profiles.csv`
- `nav15_all_ok_3_per_residue_profiles.csv`
- `cav12_all_ok_3_per_residue_profiles.csv`

The Hive commands below remain useful when local aligned arrays are unavailable.

Selection manifests have been generated locally:

- `kv21/dataRMSF/qc/kv21_all_ok3_selection_manifest.csv`
- `nav15/dataRMSF/qc/nav15_all_ok3_selection_manifest.csv`
- `cav12/dataRMSF/qc/cav12_all_ok3_selection_manifest.csv`

Copy the relevant manifest and current cluster package to Hive, then rerun only
the profile calculation. The aligned coordinate arrays do not need to be
regenerated.

## Kv2.1

```bash
python calculate_kv21_ensemble_profiles.py \
  --config config/kv21_hive.yaml \
  --merged-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/merged \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/references/kv21_aligned_references.npz \
  --annotations /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/references/kv21_residue_annotations.csv \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Kv2.1/ensemble_rmsf_results/profiles/all_ok_3 \
  --subset-name all_ok_3 \
  --subset-manifest /path/to/kv21_all_ok3_selection_manifest.csv \
  --subset-column all_ok_3 \
  --subset-key pdb_file
```

## Nav1.5

```bash
python calculate_nav15_ensemble_profiles.py \
  --config config/nav15_hive.yaml \
  --merged-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/merged \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references/nav15_aligned_references.npz \
  --annotations /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/references/nav15_residue_annotations.csv \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Nav1.5/ensemble_rmsf_results/profiles/all_ok_3 \
  --subset-name all_ok_3 \
  --subset-manifest /path/to/nav15_all_ok3_selection_manifest.csv \
  --subset-column all_ok_3 \
  --subset-key pdb_file
```

## CaV1.2

```bash
python calculate_cav12_ensemble_profiles.py \
  --config config/cav12_hive.yaml \
  --merged-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/merged \
  --references /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/references/cav12_aligned_references.npz \
  --annotations /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/references/cav12_residue_annotations.csv \
  --output-dir /quobyte/yarovoygrp/ahgz/vgic_mutants/Cav1.2/ensemble_rmsf_results/profiles/all_ok_3 \
  --subset-name all_ok_3 \
  --subset-manifest /path/to/cav12_all_ok3_selection_manifest.csv \
  --subset-column all_ok_3 \
  --subset-key pdb_file
```

Copy the generated primary profile CSVs into the local channel profile folders.
The expected filenames are:

- `kv21_all_ok_3_symmetry_averaged_profiles.csv`
- `nav15_all_ok_3_per_residue_profiles.csv`
- `cav12_all_ok_3_per_residue_profiles.csv`
