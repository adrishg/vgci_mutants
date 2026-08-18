"""Generate the focused L403A equal-depth sampling notebook."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
path=root/'kv21'/'Kv21_L403A_early_sampling_analysis.ipynb'
nb=nbf.v4.new_notebook(); c=[]
c.append(nbf.v4.new_markdown_cell("""# Kv2.1 L403A: equal-depth sampling and early access to shifted S6 geometry

This analysis tests two distinct hypotheses using the established `all_ok_rmsd_3A_structural_interface_alignment_qc` subset:

1. Does targeted masking produce broader L403A S6 sampling at equal sampling depth?
2. Does targeted masking reach 8SDA-like S6 geometry earlier?

The earlier `SamplingDepth_AllOK3_vs_First100.ipynb` is preserved. This notebook focuses on L403A and evaluates cumulative retained-structure depths of 250, 500, 1,000, 2,000, and complete QC."""))
c.append(nbf.v4.new_markdown_cell("""## Design and limitations

Structures are ordered deterministically by seed, model number, and recycle. No generation timestamps are available, so “early” means early in this reproducible index order, not proven wall-clock chronology. First-1,000 and full-QC sets contain correlated recycle snapshots; inference therefore resamples whole seeds and preserves every recycle/model/subunit belonging to a sampled seed.

Linear dispersion is the IQR. Angular dispersion is circular median absolute deviation. Positive log2(masked/vanilla dispersion) means broader masked sampling. Experimental relevance is analyzed separately using the lowest 5% of the pooled full-QC mean percentile error to 8SDA across kink angle, whole-S6 rotation, I401 azimuth, and I405 azimuth in remodeled subunits B/D. This threshold is descriptive and is not a new structural classifier."""))
c.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image
repo_root=Path.cwd()
if not (repo_root/'shared').is_dir(): repo_root=repo_root.parent
sys.path.insert(0,str(repo_root))
from kv21.run_l403a_early_sampling_analysis import run, TAB, FIG
result=run()
print(json.dumps(result['summary'],indent=2))"""))
c.append(nbf.v4.new_markdown_cell("## QC population and ordering audit"))
c.append(nbf.v4.new_code_cell("display(pd.read_csv(TAB/'l403a_sampling_depth_audit.csv'))"))
c.append(nbf.v4.new_markdown_cell("## Seed-clustered dispersion across cumulative sampling depths"))
c.append(nbf.v4.new_code_cell("""disp=pd.read_csv(TAB/'l403a_sampling_depth_dispersion.csv')
display(disp)
display(Image(filename=str(FIG/'l403a_sampling_depth_dispersion.png')))"""))
c.append(nbf.v4.new_markdown_cell("## First 1,000 versus complete QC"))
c.append(nbf.v4.new_code_cell("""display(disp[disp.depth.astype(str).isin(['1000','Full QC'])][['depth','canonical_subunit','metric','vanilla_dispersion','masked_dispersion','masked_vanilla_ratio','log2_ratio_ci_low','log2_ratio_ci_high','bootstrap_p_two_sided','bootstrap_q_within_depth','vanilla_seeds','masked_seeds']])"""))
c.append(nbf.v4.new_markdown_cell("## Cumulative access to 8SDA-like B/D S6 geometry"))
c.append(nbf.v4.new_code_cell("""enrich=pd.read_csv(TAB/'l403a_sampling_depth_experimental_like_enrichment.csv')
display(enrich)
display(Image(filename=str(FIG/'l403a_sampling_depth_experimental_like_enrichment.png')))"""))
c.append(nbf.v4.new_markdown_cell("""## Statistical reading guide

- A dispersion-ratio confidence interval entirely above zero supports broader masked sampling for that specific metric and subunit.
- A hit-fraction-difference interval entirely above zero supports enrichment of 8SDA-like states under masking at that depth.
- A lower best score means the protocol has reached a more extreme 8SDA-like individual structure, which is different from increasing the population frequency of such structures.
- The first-1,000 comparison is valuable only alongside the cumulative curve and full-QC result; by itself it cannot distinguish a stable sampling effect from an arbitrary ordering fluctuation."""))
nb['cells']=c
nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.write(nb,path); print(path)
