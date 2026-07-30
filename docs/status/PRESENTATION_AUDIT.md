# Presentation audit

Updated 29 July 2026 after the final allOK3 RMSF recomputation.

The final figure-by-figure refresh on the same date re-executed the three
distance-distribution notebooks and all eight condition-specific RMSD
notebooks. All 24 channel-root notebooks currently contain no saved Python
error outputs. The focused Kv2.1 WT/mutant RMSD comparison was also regenerated
after its ensemble legend was standardized.

## Visual identity

The canonical palettes are defined in `shared/plotting.py`.

| Channel | Visual family | Conditions |
|---|---|---|
| K\(_\mathrm{V}\)2.1 | green | WT, L403A, F412L |
| Na\(_\mathrm{V}\)1.5 | purple/lilac | WT, QQQ; v2 masks use darker supplemental purples |
| Ca\(_\mathrm{V}\)1.2 | blue | WT, G402S, G406R, G490R |

Within each condition, vanilla is the light member and masked is the
high-contrast dark member. Experimental structures use consistent color–marker
pairs within a channel. Categorical mask-distance classes use the agreed warm
accent sequence (orange, peach, yellow, cream). Continuous signed quantities
use an appropriate centered gradient rather than categorical channel colors.
Neutral gray is reserved for axes, annotations, and QC.

## Comparison coverage

| Channel | Main comparisons | Supplemental or exploratory comparisons |
|---|---|---|
| K\(_\mathrm{V}\)2.1 | WT versus L403A and F412L; vanilla versus masked within each sequence; experimental distances; gate, distal S6, pore–VSD interface, hydrophobic nexus; RMSD reference resemblance; final allOK3 RMSF | chain-resolved asymmetry and full regional/QC panels |
| Na\(_\mathrm{V}\)1.5 | WT versus QQQ; vanilla versus original mask v1; IFM/QQQ placement; pore opening/shape; experimental distances; RMSD reference resemblance; final allOK3 RMSF | mask v2 and v2-noIFM design controls and full regional/QC panels |
| Ca\(_\mathrm{V}\)1.2 | WT versus G402S and G406R; vanilla versus masked; mutation-centered contacts; experimental distances; channel-state and RMSD comparisons; final allOK3 RMSF | G490R RMSF is exploratory/supplemental until clustering and independent structural support |

The distance and mutation notebooks include both sequence comparisons
(WT versus mutant) and protocol comparisons (vanilla versus masked). Experimental
overlays are retained wherever the reference contains the required residues.

## Tables

Every CSV under `*/data*/analysis/**/tables/` contains explicit `channel`,
`condition`, and `protocol` columns. Tables that combine conditions or are not
protocol-specific state that explicitly. Run:

```bash
python scripts/standardize_analysis_tables.py --repo-root .
```

after regenerating analysis tables.

## High-resolution figure bundle

All plot-heavy notebooks export displayed figures as descriptive 300-dpi PNGs
under:

```text
../vgci_mutants_writing/figures/<channel>/<notebook>/
```

Existing RMSD, RMSF, and distance-analysis exports are mirrored under:

```text
../vgci_mutants_writing/figures/<channel>/analysis_exports/
```

`../vgci_mutants_writing/figures/figure_manifest.csv` records the channel,
condition, source file, writing-folder copy, pixel dimensions, and embedded DPI
for the mirrored analysis figures.

The final mirror contains 159 analysis figures. Every mirrored PNG reports at
least 300 dpi. The eight short `08_experimental_baseline_status.png` canvases
are QC status cards rather than manuscript figures; they record that no
independent experimental reference-to-reference baseline was available.

## Remaining manuscript-level caution

The presentation set is complete for the current preliminary claims, but this
does not replace the planned clustering and trajectory/seed-level uncertainty
analysis. G490R should remain supplemental, and the provisional topology
boundaries used in RMSF panels should be checked before final residue-level
manuscript annotation.
