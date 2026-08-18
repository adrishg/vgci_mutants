"""Append the all-distance L403A extension to the full sampling-depth notebook."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
path=root/'SamplingDepth_AllOK3_vs_First100.ipynb'
nb=nbf.read(path,as_version=4)
# Keep the original cross-condition section consistent with the fixed-cohort
# design: first 20 ordered seeds × five models, followed by final-QC filtering.
for cell in nb.cells:
    if cell.cell_type in {'markdown','code'}:
        cell.source=cell.source.replace('100 retained trajectories','QC survivors from nominal first 100')
if nb.cells and nb.cells[0].cell_type=='markdown':
    nb.cells[0].source="""# Sampling-depth sensitivity: complete final QC versus nominal first 100 trajectories

Here we compare complete final-QC ensembles with a fixed nominal early cohort defined before QC: the first 20 ordered seeds × five AlphaFold model trajectories = 100 intended trajectories per protocol. Only trajectories surviving the same final-QC procedure are analyzed. This is a sampling-depth sensitivity analysis, not an independent replicate or a chronological record of model generation.

CaV1.2 and NaV1.5 use their final 3 Å allOK3 distance subsets. Kv2.1 uses the stricter allOK3 structural-interface/alignment-QC subset. The NaV1.5 comparison is QQQ vanilla versus the original mask v1, matching the primary mutant analysis. Experimental markers retain the same PDB-specific colors and shapes used in the channel notebooks."""
for cell in nb.cells:
    if cell.cell_type=='markdown' and cell.source.startswith('## Complete versus reduced sampling'):
        cell.source="""## Complete versus reduced sampling

Each trajectory contributes one structure: its latest recycle surviving final QC. Each upper panel uses these representatives from the complete final-QC cohort. Each lower panel uses representatives surviving from the nominal first 100 trajectories: five model trajectories from each of the first 20 ordered seeds. This keeps the intended seed/model cohort fixed before QC rather than extending farther into the run until 100 survivors are accumulated. A difference between panels indicates sensitivity to sampling depth; it does not establish a separate conformational population."""
marker='## L403A all-distance equal'
cut=next((i for i,c in enumerate(nb.cells) if c.cell_type=='markdown' and marker in c.source),len(nb.cells))
nb.cells=nb.cells[:cut]
nb.cells.extend([
 nbf.v4.new_markdown_cell("""## Cross-project retention by recycle

These heatmaps use the same final-QC CSVs analyzed above. Each cell corresponds to one recycle and is annotated with the number of nominal seed/model trajectories represented after final QC. The upper row uses the fixed first 20 seeds (100 nominal trajectories); the lower row uses all 100 seeds (500 nominal trajectories). Missing trajectories are not replaced. Color gradients follow the project palettes: blue for CaV1.2, green for Kv2.1, and purple for NaV1.5."""),
 nbf.v4.new_code_cell("""from shared.recycle_retention_analysis import run_recycle_retention_analysis
recycle_retention_table, recycle_retention_figures = run_recycle_retention_analysis(
    DATASETS,
    repo_root/'docs'/'figures'/'sampling_depth',
    repo_root/'docs'/'tables'/'sampling_depth',
)
display(recycle_retention_table)
display(recycle_retention_figures['retained_fraction'])
display(recycle_retention_figures['excluded'])
plt.close('all')"""),
 nbf.v4.new_markdown_cell("""## L403A all-distance equal-seed-depth extension

The panels above test selected structural coordinates using the same fixed nominal early cohort: the first 20 ordered seeds × five model trajectories = 100 intended trajectories per protocol, followed by the final-QC cross-check. The raw Kv2.1 tables share 1,410 distance columns, but 864 are raw interchain A–B/A–C/etc. measurements that are not invariant to homotetramer chain relabeling. Those columns are excluded from broadening inference. This extension analyzes the 546 chain-label-safe intrachain distances (273 Cα and 273 shortest-heavy-atom distances).

At each depth, the nominal seed/model cohort is fixed before QC and missing trajectories are not replaced. One structure represents each surviving trajectory: the latest recycle that passes final QC. The nominal generated-trajectory budgets are 20, 40, 60, 80, and 100 (4, 8, 12, 16, and 20 seeds), followed by the complete 100-seed cohort. This is deterministic rather than timestamped chronological generation. Global IQR is descriptive. Primary per-distance inference uses one within-seed IQR and compares vanilla versus masked seed distributions with two-sided Mann–Whitney tests followed by Benjamini–Hochberg FDR correction within each depth. Brown–Forsythe tests provide a trajectory-level sensitivity analysis. Wasserstein separation uses the single representatives."""),
 nbf.v4.new_code_cell("""from IPython.display import Image, display
from kv21.run_l403a_all_distance_sampling_analysis import run as run_all_distance_sampling, TAB as ALL_DISTANCE_TAB, FIG as ALL_DISTANCE_FIG
all_distance_result = run_all_distance_sampling()
paper_statistics = pd.read_csv(ALL_DISTANCE_TAB/'l403a_all_distance_paper_statistics_summary.csv')
stability_statistics = pd.read_csv(ALL_DISTANCE_TAB/'l403a_all_distance_sampling_stability.csv')
trajectory_retention_summary = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_nominal_trajectory_qc_summary.csv')
trajectory_retention_audit = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_nominal_trajectory_qc_audit.csv')
retention_test = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_qc_retention_fisher_test.csv')
overall_seed_breadth_summary = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_seed_level_global_breadth_summary.csv')
equal_count_rarefaction_summary = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_equal_count_rarefaction_summary.csv')
random_seed_saturation_summary = pd.read_csv(ALL_DISTANCE_TAB/'l403a_random_seed_saturation_summary.csv')
first100_distribution_statistics_summary = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_seed_block_distribution_statistics_summary.csv')
full_distribution_statistics_summary = pd.read_csv(ALL_DISTANCE_TAB/'l403a_full_seed_block_distribution_statistics_summary.csv')
first100_vs_full_directional_statistics = pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_vs_full_directional_breadth_statistics.csv')
display(paper_statistics)
display(stability_statistics)"""),
 nbf.v4.new_markdown_cell("### Final-QC cross-check for the nominal first 100 trajectories"),
 nbf.v4.new_code_cell("""display(trajectory_retention_summary)
display(retention_test)
display(trajectory_retention_audit.loc[~trajectory_retention_audit['final_qc_retained']])
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_nominal_trajectory_qc_audit.png')))"""),
 nbf.v4.new_markdown_cell("### Overall breadth and equal-retained-count statistical checks"),
 nbf.v4.new_code_cell("""display(overall_seed_breadth_summary)
display(equal_count_rarefaction_summary)
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_bang_for_buck_breadth_statistics.png')))"""),
 nbf.v4.new_markdown_cell("### First-100 distribution statistics using the standard W1 framework"),
 nbf.v4.new_code_cell("""display(first100_distribution_statistics_summary)
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_seed_block_distribution_statistics.png')))"""),
 nbf.v4.new_markdown_cell("""### First-100 headline figures: breadth gained per fixed sampling budget

The four-panel summary separates the main claims: one breadth value per independent seed; the distribution of breadth effects across all 546 chain-label-safe intrachain distances; the direct one-sided seed-block permutation result after BH-FDR; and final-QC retention from the same 100 nominal trajectories. The compact evidence map orders all distances from narrower to broader and marks those with direct statistical support for masked broadening."""),
 nbf.v4.new_code_cell("""display(Image(filename=str(ALL_DISTANCE_FIG/'first100_masked_sampling_breadth_main_summary.png')))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_retention_vs_breadth_heatmap_scorecard.png')))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_all_distance_breadth_evidence_map.png')))"""),
 nbf.v4.new_markdown_cell("""### Direct statistical test of broader distributions: first 100 versus full QC

For each chain-label-safe intrachain distance, the directional null is tested by permuting complete seed blocks between protocols. The one-sided alternative is `IQR(masked) > IQR(vanilla)`. Benjamini–Hochberg correction is applied across all 546 distances separately within each cohort. This is the direct broadening test; W1 above tests any distributional separation and is not itself directional."""),
 nbf.v4.new_code_cell("""display(pd.concat([
    first100_distribution_statistics_summary.assign(cohort='Nominal first 100'),
    full_distribution_statistics_summary.assign(cohort='Full QC')
], ignore_index=True))
display(first100_vs_full_directional_statistics.sort_values(
    ['q_broader_masked_seed_block_BH_first100','q_broader_masked_seed_block_BH_full']
).head(30))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_vs_full_breadth_summary_heatmaps.png')))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_vs_full_directional_breadth_heatmap.png')))"""),
 nbf.v4.new_markdown_cell("""### Clear reproducibility views

The first heatmap counts distances according to whether the direct masked-broadening test is significant in the first-100 and full-QC cohorts. The scatter then shows the effect magnitude for every distance: upper-right points are broader under masking in both analyses, and points near the diagonal have similar early and full effects. The final heatmaps separate Cα and shortest-heavy distances to show whether the conclusion depends on the distance definition."""),
 nbf.v4.new_code_cell("""display(pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_vs_full_broadening_concordance.csv', index_col=0))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_vs_full_broadening_concordance_heatmap.png')))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_vs_full_breadth_quadrant_scatter.png')))
display(pd.read_csv(ALL_DISTANCE_TAB/'l403a_first100_vs_full_breadth_by_distance_type.csv'))
display(Image(filename=str(ALL_DISTANCE_FIG/'first100_vs_full_breadth_by_distance_type_heatmaps.png')))"""),
 nbf.v4.new_markdown_cell("### Random-seed sampling-efficiency and saturation curves"),
 nbf.v4.new_code_cell("""display(random_seed_saturation_summary)
display(Image(filename=str(ALL_DISTANCE_FIG/'random_seed_sampling_efficiency_saturation.png')))"""),
 nbf.v4.new_markdown_cell("### How widespread is masked broadening across sampling depth?"),
 nbf.v4.new_code_cell("display(Image(filename=str(ALL_DISTANCE_FIG/'all_distance_breadth_fraction_by_depth.png')))"),
 nbf.v4.new_markdown_cell("### QC survivors from the nominal first 100 versus full-QC breadth"),
 nbf.v4.new_code_cell("display(Image(filename=str(ALL_DISTANCE_FIG/'first100_nominal_vs_full_distance_breadth.png')))"),
 nbf.v4.new_markdown_cell("### Ensemble-wide distribution of masked/vanilla IQR ratios"),
 nbf.v4.new_code_cell("display(Image(filename=str(ALL_DISTANCE_FIG/'all_distance_iqr_ratio_distributions.png')))"),
 nbf.v4.new_markdown_cell("### Strongest reproducible broadened and narrowed contacts"),
 nbf.v4.new_code_cell("""display(pd.read_csv(ALL_DISTANCE_TAB/'top_concordant_distance_statistical_details.csv'))
display(Image(filename=str(ALL_DISTANCE_FIG/'top_all_distance_breadth_heatmap.png')))"""),
 nbf.v4.new_markdown_cell("### Representative chain-label-safe intrachain distance distributions"),
 nbf.v4.new_code_cell("display(Image(filename=str(ALL_DISTANCE_FIG/'representative_all_distance_distributions.png')))"),
 nbf.v4.new_markdown_cell("""### Statistical quantities used in this extension

- **Effect size:** masked/vanilla global IQR ratio and log2 ratio for each distance.
- **Chain safety:** raw Kv2.1 interchain columns are excluded; inference uses 546 same-chain distances that are invariant to subunit relabeling.
- **Trajectory representation:** latest final-QC recycle, one structure per surviving seed/model trajectory.
- **Cluster-aware breadth:** median within-seed IQR and masked/vanilla seed-IQR ratio.
- **Primary per-distance inference:** two-sided Mann–Whitney comparison of seed IQRs with within-depth BH-FDR (`seed_breadth_q`).
- **Overall breadth test:** one median normalized all-distance IQR per seed, compared as 20 vanilla versus 20 masked seeds.
- **Standard distribution test:** W1 and normalized W1 per distance, with 999 unpaired seed-block permutations, 500 seed-block bootstrap replicates, and BH-FDR across the 546 intrachain distances.
- **Sensitivity analyses:** trajectory-level Brown–Forsythe tests and repeated vanilla rarefaction to 85 retained trajectories.
- **Sampling saturation:** 1,000 random seed subsets at 5, 10, 20, 25, 50, 75, and 100 input seeds.
- **Distributional separation:** Wasserstein distance between one representative per retained trajectory, in Å and normalized by pooled IQR.
- **Sampling-depth reproducibility:** Spearman correlation and sign agreement between QC survivors from the nominal first-100 cohort (20 seeds × five models) and full-QC distance effects.

Very large ratios can result when the vanilla IQR is nearly zero. Individual highlighted contacts should therefore be reported with both raw IQRs and Wasserstein effect sizes, not ratios alone.""")
])
nbf.write(nb,path); print(path)
