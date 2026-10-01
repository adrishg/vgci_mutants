"""Regenerate the three dedicated notebooks; execute them with Jupyter/nbclient."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]

METHODS = """## Statistical design and interpretation

The complete downloaded table is audited before filtering. `status != ok`, nonzero exit codes,
nonfinite/negative diagnostics and malformed identities are reported explicitly. Ambiguous identities
or mismatched manifests stop execution. Empty anomaly CSVs still retain their headers.

Every ensemble has up to 100 seeds × 5 AF models × 11 recycle snapshots plus a final/base file.
The final/base file has identical recorded MolProbity metrics to r10 for every available trajectory;
that is **metric equality, not a coordinate checksum**. To avoid double weighting those observations,
`all_recycles` excludes final copies. `final_only` is a separate sensitivity analysis.

The primary scientific comparison uses the repository's existing QC subset, independently of clashscore:
3 Å convergence for Cav1.2/Nav1.5 and the established structural/interface/alignment-QC distance
selection for Kv2.1. All raw distributions and excluded counts remain visible. QC-selected results
describe survivors, not all generated models. No new clashscore cutoff is used.

For each metric, average snapshots within seed/AF-model trajectory, then average AF models within seed.
This gives each retained seed equal weight and avoids treating recycles or the five AF models as
independent replicates. Point estimates are seed-balanced mean differences B−A; 95% percentile
bootstrap intervals use 3,000 seed draws. When seed-label sets match completely, nominal paired-seed
t tests and the existing `paired_common_seed_bootstrap` are used. Otherwise Welch tests compare all
retained seed summaries. Both independent-seed and common-label paired sensitivities are exported.
Shared numeric labels do **not** establish shared random-number streams; no seeds are renumbered,
and rank is never a pairing key. Common-label sensitivity may change the population when ranges differ.

BH correction covers all planned pairwise comparisons × all three metrics within each channel/cohort;
primary, independent, and common-label p-values form separate families. Supplemental cohort families
are exploratory. The seed rank-biserial effect is P(B>A)−P(B<A) across seed summaries, with ties zero;
it describes marginal ordering even for nominally paired tests. Model medians/IQRs are descriptive
snapshot summaries and are a different estimand from the seed-balanced mean contrast.

AF-model/snapshot-adjusted regressions provide a sensitivity check with standard errors clustered by
recorded seed. Rank is an output ordering variable, is absent for Kv2.1 L403A vanilla, and is inspected
descriptively rather than controlled as if it were a randomized covariate. These uncertainty estimates
describe computational sampling, not biological replication or a validated clinical effect.

Clashscore is a normalized global clash statistic ([Phenix glossary](https://www.phenix-online.org/version_docs/1.8.2-1296/glossary.htm)).
`n_bad_clashes` is an absolute count and `max_overlap` is a single extreme; they are complementary
diagnostics, not interchangeable biological measures. The local export does not record Phenix version,
hydrogen options, the atom denominator, or atom-level clash identities. Absolute validation grades,
cross-channel biological comparisons, and residue-specific attribution require additional provenance.
"""

INTRO = {
'cav12': """All six WT/G402S/G406R × vanilla/masked ensembles are shown together, emphasizing G406R.
The nearby short distances in `Cav12_G406R_mutationSite_analysis.ipynb` motivate the question; they
do not establish that any global clashscore change originates at introduced R406.""",
'kv21': """All six WT/L403A/F412L × vanilla/masked ensembles are compared. The conformational extension
uses the exact structures and existing chain-resolved coordinates from
`Kv21_L403A_conformational_validation.ipynb`. Established tetramer/interface/alignment QC is retained.""",
'nav15': """Following `Nav15_main_and_supplemental_figures.ipynb`, the main comparison is **WT vanilla →
QQQ vanilla → QQQ original masked**. All seven downloaded ensembles enter the audit, tables, and
supplementary mask-design plot. WT original masked and masked-v2/no-IFM variants are controls.
`no-IFM` means exclusion of the IFM region from the mask, not deletion of the native IFM sequence.
Original and v2 masks are joined to their explicitly named manifests; a few filenames overlap
between masks, so basename without ensemble identity is not a safe global key.""",
}


def build(channel, prefix, title):
    cells=[]
    md=lambda s:cells.append(nbf.v4.new_markdown_cell(s))
    code=lambda s:cells.append(nbf.v4.new_code_cell(s))
    md(f'# {title} MolProbity / clashscore analysis\n\n'+INTRO[channel])
    md('## Reproducible setup\nRun from the repository root or channel directory. Dependencies: NumPy, pandas, SciPy, statsmodels, Matplotlib, seaborn, and Jupyter. MolProbity tables remain in the local downloaded directory. No upload or move is performed. Shared functions live in `shared/molprobity_analysis.py`, as in the existing analyses.')
    code(f"""from pathlib import Path
import sys, os, json, platform, importlib.metadata
from IPython.display import display, Markdown
import matplotlib.pyplot as plt
import pandas as pd

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'shared/data_access.py').is_file())
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from shared import molprobity_analysis as mp
CHANNEL = {channel!r}
OUT = mp.output_dir(CHANNEL)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 24)
mp.apply_kv21_style()
""")
    md('## Data-access boundary\nSet `VGIC_MOLPROBITY_ROOT` or `DATA_ROOT` to the downloaded directory. Future Hugging Face migration only needs to replace the loader with explicit paths through the existing `shared.data_access.resolve`; downstream analysis accepts the same DataFrame. Existing structural-QC and conformational inputs already use that resolver, including local/cache/offline behavior.')
    code("""DATA_ROOT = os.environ.get('VGIC_MOLPROBITY_ROOT', str(ROOT/'molprobity_download'))
raw, inventory, total_csv_files = mp.load_molprobity(CHANNEL, DATA_ROOT)
inventory.to_csv(OUT/'tables/input_inventory.csv', index=False)
print(f'{total_csv_files} CSVs in downloaded directory; {len(inventory)} for this channel; {len(raw):,} rows')
display(inventory)
""")
    md('## Audit before plotting\nThe expected grid uses each ensemble’s recorded seed labels, AF models 1–5, r0–r10, and final. Missing whole trajectories/seeds are additionally checked against the existing model manifest. Raw input hashes are saved above. NaN recycle is expected only for final/base models; missing rank is allowed and reported.')
    code("""data, audit = mp.audit_data(raw, CHANNEL, OUT)
display(audit)
display(pd.read_csv(OUT/'tables/missing_snapshots.csv'))
display(pd.read_csv(OUT/'tables/manifest_discrepancies.csv'))
display(pd.read_csv(OUT/'tables/row_anomalies.csv'))
cohorts = mp.select_cohorts(data, CHANNEL, OUT)
summary = pd.concat([mp.describe(frame, name) for name, frame in cohorts.items()], ignore_index=True)
summary.to_csv(OUT/'tables/descriptive_statistics.csv', index=False)
display(summary[(summary.cohort=='all_recorded') & summary.metric.eq('clashscore')].round(3))
display(pd.read_csv(OUT/'tables/selection_audit.csv'))
""")
    md(METHODS)
    code("""contrasts = mp.contrast_statistics(cohorts, CHANNEL, OUT)
primary = contrasts[(contrasts.cohort=='repository_QC') & contrasts.metric.eq('clashscore')]
display(primary[['A','B','N_A','N_B','n_seeds_A','n_seeds_B','common_seeds',
                 'common_model_snapshots','only_A_snapshots','only_B_snapshots',
                 'paired_seed_correlation','primary_test']])
display(primary[['A','B','median_difference','seed_mean_difference','CI_low','CI_high',
                 'seed_rank_biserial','q_BH','independent_q_BH','common_seed_difference',
                 'common_seed_CI_low','common_seed_CI_high','common_seed_q_BH']].round(4))
""")
    md('## Clashscore distributions and effect sizes\nViolins retain every eligible individual snapshot with median and quartiles; no outliers are clipped. The paired raw/QC panels expose selection effects. The contrast figure shows seed-balanced differences and 95% bootstrap intervals; q-values are reported without significance stars. Colors, typography, thin outlines, light grids, and PDF/300-dpi PNG exports follow `shared.plotting` and existing conformational figures.')
    code("""mp.plot_distributions(cohorts, CHANNEL, OUT)
plt.show()
mp.plot_effects(contrasts, CHANNEL, OUT)
plt.show()
display(Markdown(mp.interpretation(summary, contrasts, CHANNEL)))
""")
    md('## Complementary steric diagnostics\nThese panels use the repository QC subset. Counts depend on structure size and atom handling; maximum overlap reports one extreme contact. Full numerical results and corrected comparisons for both metrics are retained alongside clashscore.')
    code("""mp.plot_diagnostics(cohorts, CHANNEL, OUT)
plt.show()
display(summary[(summary.cohort=='repository_QC') & summary.metric.ne('clashscore')].round(3))
display(contrasts[(contrasts.cohort=='repository_QC') & contrasts.metric.ne('clashscore')][
    ['A','B','metric','median_difference','seed_mean_difference','CI_low','CI_high','seed_rank_biserial','q_BH']].round(4))
""")
    if channel=='nav15':
        md('## Supplementary mask-design controls\nAll seven ensembles are retained here. Compare the original mask with v2 within sequence, and WT masked v2 with WT masked v2 no-IFM. These controls test sensitivity to mask design.')
        code("mp.plot_nav_controls(cohorts, OUT)\nplt.show()")
    md('## Recycle, AF-model, seed, and rank sensitivity\nThe full snapshot trace includes final separately and all early-recycle tails. The final-only analysis does not substitute a last available snapshot for a missing final file. The adjusted regression controls AF-model and snapshot composition, with uncertainty clustered by seed; its observation-weighted estimand differs from the seed-balanced primary result.')
    code("""mp.stratification(cohorts, CHANNEL, OUT)
plt.show()
adjusted = mp.adjusted_contrasts(cohorts, CHANNEL, OUT)
display(adjusted.round(4))
af_sensitivity = mp.af_model_sensitivity(cohorts, CHANNEL, OUT)
display(af_sensitivity.pivot(index=['A','B','mode'], columns='AF_model', values='seed_mean_difference').round(3))
display(contrasts[contrasts.metric.eq('clashscore')][
    ['cohort','A','B','seed_mean_difference','CI_low','CI_high','q_BH']].round(3))
""")
    if channel=='kv21':
        md("""## S6 conformational association: join first, test second

The existing chain-resolved table contains all six ensembles. Join on **ensemble + exact PDB basename**,
require one record per canonical chain, and cross-check seed, AF model, and snapshot. No rank-based
matching or fuzzy filename normalization is permitted. The audit covers all recorded models; the
primary association uses the exact final structural-QC subset.

Two complementary existing-coordinate summaries are examined:

- Median canonical-chain F412-position Cα displacement from 8SD3 (Å), an **unsigned** magnitude.
  For F412L, the existing column refers to the homologous L412 Cα anchor.
- A directional projection of the four canonical-chain **whole-S6 rotations** onto the experimental
  8SD3→8SDA rotation vector: `dot(model − 8SD3, 8SDA − 8SD3) / norm(8SDA − 8SD3)^2`.
  Angular differences wrap to [−180°,180°). Zero is the WT rotation reference, one the mutant-reference
  projection. This is a derived summary of the existing coordinate, not a state classifier, a complete
  3-D conformation match, or a claim that projection one reproduces all four chains. Canonical subunit
  mapping is inherited from the validated analysis, not inferred from MolProbity.

Spearman correlation uses seed summaries (median per trajectory, equal AF-model mean within seed),
with 1,000 seed bootstrap draws. Descriptive model-level rho is also exported. Regressions adjust for
AF model/snapshot and cluster uncertainty by seed; slope × coordinate IQR expresses a local magnitude.
Direct interaction tests compare slopes between conditions rather than comparing significance labels.
The 12 association tests per method/cohort and the 18 interaction tests form separate BH families.
Raw-recycle, final-only, and r10 within the same structural-QC population are sensitivity analyses. Correlation cannot establish that
steric strain causes or limits S6 movement; early-recycle quality and survivor selection can influence it.
""")
        code("""joined, associations, interactions = mp.kv21_conformation(cohorts, OUT)
display(pd.read_csv(OUT/'tables/conformation_join_audit.csv'))
display(pd.read_csv(OUT/'tables/S6_rotation_reference.csv'))
display(associations[associations.cohort.eq('repository_QC')].round(4))
display(interactions.round(4))
mp.plot_conformation(joined, associations, OUT)
plt.show()
mp.plot_conformation_adjusted(joined, associations, OUT)
plt.show()
display(associations[associations.cohort.eq('r10_repository_QC')].round(4))
""")
    md('## Focused interpretation\nThe following text is generated from the executed result tables so it stays synchronized with reruns.')
    code("display(Markdown(mp.focused_interpretation(summary, contrasts, CHANNEL, OUT)))")
    md('## Reproducibility record\nAll generated outputs are confined to this channel’s `dataExtra/molprobity/{tables,figures}`. Existing figures and analyses are not overwritten. Input hashes, missing-model lists, full descriptive/contrast tables, pairing diagnostics and sensitivity results accompany the figures.')
    code("""record = {'channel': CHANNEL, 'total_downloaded_csvs': total_csv_files,
          'channel_csvs': len(inventory), 'rows': len(raw), 'python': platform.python_version(),
          'bootstrap_replicates': mp.BOOTSTRAPS, 'random_seed': mp.RANDOM_SEED,
          'packages': {p: importlib.metadata.version(p) for p in ['numpy','pandas','scipy','statsmodels','matplotlib','seaborn']}}
(OUT/'tables/run_metadata.json').write_text(json.dumps(record, indent=2)+chr(10))
print('Completed:', OUT.relative_to(ROOT))
""")
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}})
    path=ROOT/channel/f'{prefix}_MolProbity_clashscore_analysis.ipynb'
    nbf.write(nb,path)
    print(path)


if __name__=='__main__':
    for args in [('cav12','Cav12','Cav1.2'),('kv21','Kv21','Kv2.1'),('nav15','Nav15','Nav1.5')]:
        build(*args)
