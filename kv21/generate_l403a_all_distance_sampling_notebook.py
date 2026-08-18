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
c.append(nbf.v4.new_code_cell("display(pd.read_csv(TAB/'l403a_first100_seed_level_global_breadth_summary.csv'))"))
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
c.append(nbf.v4.new_markdown_cell("""## Figure S8 assembly: fixed-budget sampling breadth

This cell assembles the four selected, analysis-generated panels into one editable Figure S8 layout. The source panels are not screenshots from an external document: each is generated above by `run()` from the saved analysis tables. The unequal layout preserves their native aspect ratios so the 546-distance evidence stripe remains legible.

- **A:** First100 seed-level breadth, distance-wise breadth, directional test, and retention.
- **B:** Ordered effects and directional BH-significance across all 546 distances.
- **C:** Random-seed saturation from 1,000 subsets per seed budget.
- **D:** First100-versus-full directional-broadening concordance.
"""))
c.append(nbf.v4.new_code_cell("""from PIL import Image as PILImage, ImageChops
import matplotlib.pyplot as plt

panel_paths = {
    'A': FIG/'first100_masked_sampling_breadth_main_summary.png',
    'B': FIG/'first100_all_distance_breadth_evidence_map.png',
    'C': FIG/'random_seed_sampling_efficiency_saturation.png',
    'D': FIG/'first100_vs_full_broadening_concordance_heatmap.png',
}

def trim_white_margin(path, tolerance=8):
    image = PILImage.open(path).convert('RGB')
    background = PILImage.new('RGB', image.size, (255, 255, 255))
    difference = ImageChops.difference(image, background).convert('L')
    difference = difference.point(lambda value: 255 if value > tolerance else 0)
    bounds = difference.getbbox()
    return image.crop(bounds) if bounds else image

panels = {label: trim_white_margin(path) for label, path in panel_paths.items()}

fig = plt.figure(figsize=(22, 12.5), facecolor='white')
outer = fig.add_gridspec(2, 1, height_ratios=[1.80, 1.00], hspace=.025)
top = outer[0].subgridspec(1, 2, width_ratios=[1, 1], wspace=.025)
bottom = outer[1].subgridspec(1, 2, width_ratios=[3.35, 1.15], wspace=.035)
axes = {
    'A': fig.add_subplot(top[0, 0]),
    'C': fig.add_subplot(top[0, 1]),
    'B': fig.add_subplot(bottom[0, 0]),
    'D': fig.add_subplot(bottom[0, 1]),
}

for label, axis in axes.items():
    axis.imshow(panels[label], interpolation='lanczos')
    axis.set_anchor('N')
    axis.axis('off')
    axis.text(-.055, 1.005, label, transform=axis.transAxes, ha='left', va='top',
              fontsize=23, fontweight='bold', color='#111111',
              bbox={'facecolor':'white','edgecolor':'none','pad':1.5})

fig.subplots_adjust(left=.025, right=.995, bottom=.015, top=.995)
composite_png = FIG/'Figure_S8_fixed_budget_sampling_breadth.png'
composite_pdf = FIG/'Figure_S8_fixed_budget_sampling_breadth.pdf'
fig.savefig(composite_png, dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(composite_pdf, bbox_inches='tight', facecolor='white')
display(fig)
plt.close(fig)

print('Saved:', composite_png.relative_to(repo_root))
print('Saved:', composite_pdf.relative_to(repo_root))
"""))
c.append(nbf.v4.new_markdown_cell("## Manuscript-facing uncertainty audit"))
c.append(nbf.v4.new_code_cell("""from analysis.statistics_revision.scripts.run_kv21_sampling_breadth_uncertainty import run as run_uncertainty
run_uncertainty()
revision_tables=repo_root/'analysis'/'statistics_revision'/'tables'
display(pd.read_csv(TAB/'l403a_first100_seed_level_global_breadth_summary.csv'))
display(pd.read_csv(revision_tables/'kv21_s6_masked_vs_vanilla_breadth_bootstrap.csv'))
display(pd.read_csv(revision_tables/'kv21_rmsf_trajectory_block_bootstrap.csv'))
display(pd.read_csv(revision_tables/'kv21_sampling_breadth_manuscript_statistics.csv'))"""))
c.append(nbf.v4.new_markdown_cell("""## Reading guide

- `global_IQR_ratio > 1` means the masked raw ensemble is broader for that distance.
- `seed_IQR_ratio > 1` means the typical within-seed masked ensemble is broader.
- `seed_breadth_q < 0.05` identifies FDR-significant protocol differences using seeds as replicates.
- `trajectory_median_W1_A` describes protocol separation after giving each seed/model trajectory one median observation.
- Agreement between first 1,000 and full QC supports sampling-efficiency claims; it does not convert deterministic seed order into chronological time.
- These all-distance results establish breadth, not whether broader states are experimentally correct."""))
nb['cells']=c; nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}; nbf.write(nb,path); print(path)
