# Figure index

The presentation notebooks live directly inside each channel directory. Generated figures and
tables are stored below the corresponding `dataDistances/`, `dataRMSD/`, or `dataRMSF/` analysis
directory.

## CaV1.2 experimental RMSD

Primary notebooks:

- `cav12/Cav12_WT_experimental_RMSD.ipynb`
- `cav12/Cav12_G402S_experimental_RMSD.ipynb`
- `cav12/Cav12_G406R_experimental_RMSD.ipynb`

The notebooks report retained-model counts, measurement completeness, overall and stable-core RMSD,
regional pore/S6/interface distributions, core-aligned versus locally aligned RMSD, protocol median
differences, and robust effect-size summaries. Generated outputs are under
`cav12/dataRMSD/analysis/<condition>/`.

Experimental references are 8HLP, 8WE6, and 8FD7. These are WT/state references rather than
mutation-matched G402S or G406R structures.

## Nav1.5 experimental RMSD

Primary notebooks:

- `nav15/Nav15_WT_experimental_RMSD.ipynb`
- `nav15/Nav15_QQQ_experimental_RMSD.ipynb`

The notebooks report retained-model counts, completeness, overall/core and regional RMSD,
core-aligned versus locally aligned measurements, masked-minus-vanilla median differences, and
effect-size summaries. WT and QQQ outputs are under `nav15/dataRMSD/analysis/wt/` and
`nav15/dataRMSD/analysis/qqq/`.

The main-text comparison emphasizes 8VYJ as the native full-length open reference and 7FBS as the
engineered QQQ-open reference. The broader experimental panel remains available as supplemental
state context.

## Kv2.1 experimental RMSD

Primary notebooks:

- `kv21/Kv21_WT_experimental_RMSD.ipynb`
- `kv21/Kv21_L403A_experimental_RMSD.ipynb`
- `kv21/Kv21_F412L_experimental_RMSD.ipynb`
- `kv21/Kv21_WT_mutants_RMSD_comparison.ipynb`

8SD3 is the mutation-matched WT reference and 8SDA is the L403A reference. For F412L, 8SDA is used
only as an L403A-like state comparator because no mutation-matched experimental F412L structure is
available.

## High-resolution exports

Notebook figures are also mirrored outside the repository under
`vgci_mutants_writing/figures/<channel>/`. These exports are presentation products; the notebooks
and shared plotting functions remain the reproducible source.
