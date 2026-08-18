"""Generate the Figure 2B-style notebook restricted to nominal first 100."""

from pathlib import Path
import nbformat as nbf

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'paperFigures'/'Figure2B_First100_CrossChannel_DistanceSampling.ipynb'
nb=nbf.v4.new_notebook()
nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
nb.metadata.language_info={'name':'python','version':'3'}
nb.cells=[
 nbf.v4.new_markdown_cell("""# Figure 2B style: cross-channel sampling within the first 100 trajectories

This notebook recreates the visual language of `Figure2B_CrossChannel_DistanceSampling.ipynb` while restricting every protocol to a fixed nominal cohort: the first 20 ordered seeds × five AlphaFold model trajectories = 100 intended trajectories. Final-QC failures are not replaced. Each retained trajectory contributes exactly one structure, its latest recycle surviving final QC.

Rows follow the manuscript order Kv2.1, NaV1.5, and CaV1.2; columns show the intracellular gate, selectivity filter, and voltage sensors. Colors, split violins, experimental-reference symbols, typography, canvas, and legend construction match Figure 2B.

For Kv2.1, raw A–B/A–C chain labels are not stable under masked homotetramer rearrangement. The gate therefore uses chain-label-invariant cross-pore maxima along S6, and the filter uses the six sorted ring spans. Kv2.1 VSD distances remain intrachain. CaV1.2 and NaV1.5 coordinates are unchanged."""),
 nbf.v4.new_code_cell("""from pathlib import Path
import sys,copy,importlib
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

repo_root=next(path for path in [Path.cwd(),*Path.cwd().parents] if (path/'shared').is_dir())
sys.path.insert(0,str(repo_root))

import shared.cross_channel_distance_figure as figure2
importlib.reload(figure2)
from shared.sampling_depth_analysis import first_nominal_trajectory_cohort,latest_qc_trajectory_representatives,_kv21_ranked_ring_columns
from shared.plotting import add_s6_cross_pore_columns

configs=figure2.paper_configs(repo_root)
figures=repo_root/'docs'/'figures'/'cross_channel_distance_sampling_first100'
tables=repo_root/'docs'/'tables'/'cross_channel_distance_sampling_first100'
figures.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)"""),
 nbf.v4.new_markdown_cell("""## Fix the nominal cohort before QC

Selection is deterministic rather than timestamp-based. The selected seed IDs may differ between protocols because the original runs use different seed ranges; inference here compares equal nominal budgets rather than paired random seeds."""),
 nbf.v4.new_code_cell("""data={}; audit=[]
for channel,config in configs.items():
    data[channel]={}
    for protocol in ('vanilla','masked'):
        raw=pd.read_csv(config[protocol])
        early=first_nominal_trajectory_cohort(raw,number_seeds=20)
        representatives=latest_qc_trajectory_representatives(early).copy()
        data[channel][protocol]=representatives
        names=early.pdb_file.astype(str)
        seed=pd.to_numeric(names.str.extract(r'_seed_(\\d+)',expand=False),errors='coerce')
        model=pd.to_numeric(names.str.extract(r'_model_(\\d+)',expand=False),errors='coerce')
        audit.append({'channel':channel,'protocol':protocol,'nominal_trajectories':100,
            'retained_trajectories':len(representatives),'excluded_trajectories':100-len(representatives),
            'first_seed':int(seed.min()),'last_seed':int(seed.max()),
            'retained_seed_model_keys':pd.DataFrame({'seed':seed,'model':model}).drop_duplicates().shape[0]})
audit=pd.DataFrame(audit)
audit.to_csv(tables/'first100_final_qc_retention_audit.csv',index=False)
display(audit)"""),
 nbf.v4.new_markdown_cell("## Apply chain-label-safe Kv2.1 coordinates"),
 nbf.v4.new_code_cell("""first100_aliases=copy.deepcopy(figure2.ALIASES)
for protocol in ('vanilla','masked'):
    frame,sf_aliases=_kv21_ranked_ring_columns(data['Kv2.1'][protocol])
    frame,gate_aliases=add_s6_cross_pore_columns(frame)
    data['Kv2.1'][protocol]=frame
first100_aliases['Kv2.1']['selectivity_filter']=sf_aliases
first100_aliases['Kv2.1']['intracellular_gate']={f'{k} cross-pore':v for k,v in gate_aliases.items()}

# The Figure 2 renderer reads its alias catalog at draw time. This notebook-local
# replacement leaves the original full-QC notebook and source defaults unchanged.
figure2.ALIASES=first100_aliases
display(pd.DataFrame([
    {'region':region,'display_alias':alias,'column':column}
    for region in ('intracellular_gate','selectivity_filter','vsds')
    for alias,column in first100_aliases['Kv2.1'][region].items()
]))"""),
 nbf.v4.new_markdown_cell("""## Main Figure 2B-style panel

All violin observations are independent retained trajectories, not correlated recycle snapshots. The visual structure is otherwise the same as the full-QC Figure 2B notebook."""),
 nbf.v4.new_code_cell("""main_figure=figure2.make_grid(repo_root,configs,data,
    regions=('intracellular_gate','selectivity_filter','vsds'),
    output_path=figures/'cross_channel_first100_main_gate_filter_vsd.png')
display(main_figure); plt.close(main_figure)"""),
 nbf.v4.new_markdown_cell("## Equal-size first-100 panel exports"),
 nbf.v4.new_code_cell("""panel_paths=figure2.make_individual_panels(repo_root,configs,data,figures/'individual_panels')
print(f'Exported {len(panel_paths)} equal-size panels')
for path in panel_paths: print(path.relative_to(repo_root))"""),
]
nbf.write(nb,OUT);print(OUT)
