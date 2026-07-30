# Experimental-comparison RMSD workflow

This workflow audits merged model-to-reference RMSD tables and applies the same
model-level convergence selections used by the distance analyses. Experimental
RMSD is never used as a quality-control criterion.

## Current input status

| Channel | RMSD input | Status |
|---|---|---|
| Kv2.1 | `kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD.csv` | Audited and analyzed |
| Nav1.5 | `nav15/dataRMSD/Nav15_all_models_all_references_RMSD_distances.csv` | Audited and analyzed |
| Cav1.2 | `cav12/dataRMSD/Cav12_all_models_all_references_RMSD_contacts.csv` | Audited and analyzed |

The Nav1.5 and Cav1.2 source files currently sit one directory above
`dataRMSD/`. They remain unchanged there; filtered copies and audit products
are written under each channel's `dataRMSD/` directory.

## Filtering

The principal `_OK3.csv` selection exactly reproduces
`shared.dataset_selection.select_manifest_rows(..., "all_ok_3")`:

- exclude base-model rows;
- within each seed/model-number trajectory, identify the first recycle after
  which every remaining transition has successive stable-core RMSD ≤ 3.0 Å
  and aligned coverage ≥ 0.90;
- retain that recycle and every later recycle;
- retain every experimental-reference row belonging to an accepted model.

The model join key is the exact basename of `pdb_file`, falling back to
`model_path`. Rank, seed, model number, and `.r0`–`.r10` suffixes are preserved.

Kv2.1 also receives `_OK3_QC.csv`. Its allowlists come from the six existing
`*_all_ok_rmsd_3A_structural_interface_qc.csv` distance tables. These already
encode the established G377 tetramer-integrity and trajectory-level pore–VSD
interface checks. No new RMSD cutoff is introduced.

## Run

Use the repository's analysis environment:

```bash
/Users/ahernandezgonzalez/.local/share/mamba/envs/bioadri/bin/python \
  scripts/filter_rmsd_by_existing_qc.py \
  --channel kv21 \
  --rmsd-csv kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD.csv
```

Audit without writing a filtered RMSD table:

```bash
/Users/ahernandezgonzalez/.local/share/mamba/envs/bioadri/bin/python \
  scripts/filter_rmsd_by_existing_qc.py \
  --channel kv21 \
  --rmsd-csv kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD.csv \
  --dry-run
```

Custom destinations are available through `--output` and
`--diagnostics-output`. The script refuses to overwrite the source.

Generate per-condition notebooks:

```bash
/Users/ahernandezgonzalez/.local/share/mamba/envs/bioadri/bin/python \
  scripts/generate_rmsd_notebooks.py \
  --channel kv21 \
  --source kv21/dataRMSD/Kv21_all_models_vs_8SD3_8SDA_RMSD_OK3.csv
```

The notebooks write figures and tables under
`<channel>/dataRMSD/analysis/<condition>/`.

## Interpretation

Comparisons are performed within sequence condition, reference, QC subset, and
measurement. The summaries report complete distributions, medians, IQRs,
5th–95th percentiles, missingness, coverage, bootstrap confidence intervals,
and Cliff's delta. Recycle snapshots are not independent; their effect sizes
describe the sampled ensembles and should not be interpreted as independent
replicate statistics.

Core-aligned and locally aligned regional RMSDs answer different questions:

- high core-aligned plus low local RMSD suggests largely rigid displacement;
- high values in both suggest displacement plus internal deformation.

No numerical threshold is imposed on this interpretation. A lower RMSD means
greater structural similarity to that reference, not proof of the correct
functional state.
