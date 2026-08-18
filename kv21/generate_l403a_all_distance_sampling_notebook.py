"""Generate the L403A all-distance sampling notebook."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]; path=root/'kv21'/'Kv21_L403A_all_distance_sampling_analysis.ipynb'
nb=nbf.v4.new_notebook(); c=[]
c.append(nbf.v4.new_markdown_cell("""# Kv2.1 L403A: all-distance sampling breadth at equal depth

This notebook expands the S6-focused sampling analysis to every exact shared numerical distance in the established `all_ok_rmsd_3A_structural_interface_alignment_qc` vanilla and masked L403A tables: 705 Cα distances and 705 shortest-heavy-atom distances.

It asks whether masking broadens the complete precomputed distance ensemble, whether that effect is already visible in the first 1,000 retained structures, and how closely early results reproduce complete-QC results."""))
c.append(nbf.v4.new_markdown_cell("""## Statistical design

Rows are ordered deterministically by seed, model number, and recycle; this is an indexed equal-depth analysis, not timestamped wall-clock chronology. Global IQR ratios describe the complete distributions. For inference, an IQR is calculated within each seed for every distance, making seed—not recycle snapshot—the statistical unit. Vanilla and masked seed-IQR distributions are compared with two-sided Mann–Whitney tests followed by Benjamini–Hochberg FDR correction within each sampling depth. Distributional separation is additionally summarized by Wasserstein distance between trajectory medians.

Positive log2 ratios mean broader masked sampling; negative values mean broader vanilla sampling. Breadth is not equivalent to accuracy or experimental relevance."""))
c.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image
repo_root=Path.cwd()
if not (repo_root/'shared').is_dir(): repo_root=repo_root.parent
sys.path.insert(0,str(repo_root))
from kv21.run_l403a_all_distance_sampling_analysis import run, TAB, FIG
result=run(); print(json.dumps(result['audit'],indent=2))"""))
c.append(nbf.v4.new_markdown_cell("## Ensemble-wide summary across sampling depths"))
c.append(nbf.v4.new_code_cell("display(pd.read_csv(TAB/'l403a_all_distance_sampling_summary.csv')); display(Image(filename=str(FIG/'all_distance_breadth_fraction_by_depth.png')))"))
c.append(nbf.v4.new_markdown_cell("## First 1,000 versus full QC"))
c.append(nbf.v4.new_code_cell("display(pd.read_csv(TAB/'l403a_all_distance_sampling_stability.csv')); display(Image(filename=str(FIG/'first1000_vs_full_distance_breadth.png')))"))
c.append(nbf.v4.new_markdown_cell("## Distribution of breadth effects across all 1,410 distances"))
c.append(nbf.v4.new_code_cell("display(Image(filename=str(FIG/'all_distance_iqr_ratio_distributions.png')))"))
c.append(nbf.v4.new_markdown_cell("## Strongest concordant broadened and narrowed distances"))
c.append(nbf.v4.new_code_cell("display(pd.read_csv(TAB/'top_concordant_distance_breadth_changes.csv')); display(Image(filename=str(FIG/'top_all_distance_breadth_heatmap.png')))"))
c.append(nbf.v4.new_markdown_cell("## Machine-readable distance-level statistics"))
c.append(nbf.v4.new_code_cell("""stats=pd.read_csv(TAB/'l403a_all_distance_sampling_statistics.csv')
display(stats[stats.depth.astype(str).isin(['1000','Full QC'])])
print(f'{len(stats):,} distance-by-depth rows')"""))
c.append(nbf.v4.new_markdown_cell("""## Reading guide

- `global_IQR_ratio > 1` means the masked raw ensemble is broader for that distance.
- `seed_IQR_ratio > 1` means the typical within-seed masked ensemble is broader.
- `seed_breadth_q < 0.05` identifies FDR-significant protocol differences using seeds as replicates.
- `trajectory_median_W1_A` describes protocol separation after giving each seed/model trajectory one median observation.
- Agreement between first 1,000 and full QC supports sampling-efficiency claims; it does not convert deterministic seed order into chronological time.
- These all-distance results establish breadth, not whether broader states are experimentally correct."""))
nb['cells']=c; nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}; nbf.write(nb,path); print(path)
