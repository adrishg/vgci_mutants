"""Generate the reproducible L403A conformational-validation notebook."""
from pathlib import Path
import nbformat as nbf

root=Path(__file__).resolve().parents[1]
path=root/"kv21"/"Kv21_L403A_conformational_validation.ipynb"
nb=nbf.v4.new_notebook()
cells=[]
cells.append(nbf.v4.new_markdown_cell("""# Kv2.1 L403A conformational validation against 8SD3 → 8SDA

This notebook tests the controlled question: **does the protocol-matched WT → L403A conformational shift reproduce the direction and magnitude of the experimental 8SD3 → 8SDA shift better under targeted masking than under vanilla AF2?**

Only the existing validated conformational-metric CSVs are used; no PDB-derived metric is recalculated. The primary prediction subset is the repository's established `all_ok_rmsd_3A_structural_interface_alignment_qc` selection used for distance analysis. Exact PDB basenames are joined one-to-one and condition/protocol/seed/model/recycle metadata are checked. Subunits A–D remain separate."""))
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import sys, json
import pandas as pd
from IPython.display import display, Image

repo_root = Path.cwd()
if not (repo_root / 'shared').is_dir():
    repo_root = Path.cwd().parent
sys.path.insert(0, str(repo_root))
from kv21.run_l403a_conformational_validation import run, TAB, FIG

result = run()
print(json.dumps(result['summary'], indent=2))"""))
cells.append(nbf.v4.new_markdown_cell("""## Input inventory, matching, and QC

Recycle snapshots are not independent replicates. All model/recycle observations belonging to a condition/protocol/seed/subunit are first reduced to a seed-level center; protocol-specific WT and L403A summaries are then paired by seed. Whole seeds are resampled for 2,000 bootstrap replicates. Angular centers and differences are circular and wrapped to [−180°, +180°).

The shared distance-QC subset is primary. Complete-data, final-output-only, and positive-frame-orientation analyses are saved as labeled sensitivities. The convergence-selected tables contain no records labeled `final`, so final-output sensitivity necessarily uses the complete conformational table and is not presented as a QC-filtered result. DSSP was unavailable; no DSSP-based π-helix claim is made."""))
cells.append(nbf.v4.new_code_cell("""display(pd.read_csv(TAB/'dataset_inventory.csv'))
display(pd.read_csv(TAB/'distance_qc_join_audit.csv'))
display(pd.read_csv(TAB/'qc_summary.csv'))"""))
cells.append(nbf.v4.new_markdown_cell("""## Experimental reference and protocol-matched recovery

For each metric and canonical subunit, Δexp = 8SDA − 8SD3. Model effects are paired seed-level Δvanilla and Δmasked. `masked_advantage` is the vanilla absolute experimental error minus the masked error, so positive values favor masked. It is retained in the physical unit of each metric; values should not be compared across rows with different units."""))
cells.append(nbf.v4.new_code_cell("""reference=pd.read_csv(TAB/'experimental_reference_by_subunit.csv')
effects=pd.read_csv(TAB/'protocol_effects_by_metric_subunit.csv')
display(reference.head(12))
display(effects[['canonical_subunit','metric','experimental_delta','model_delta__vanilla','model_delta__masked','ci_low__vanilla','ci_high__vanilla','ci_low__masked','ci_high__masked','direction_match__vanilla','direction_match__masked','absolute_experimental_error__vanilla','absolute_experimental_error__masked','masked_advantage','masked_advantage_ci_low','masked_advantage_ci_high']])"""))
cells.append(nbf.v4.new_markdown_cell("## A. PIP/S6 kink straightening"))
cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIG/'panel_A_pip_s6_kink.png')))"))
cells.append(nbf.v4.new_markdown_cell("## B. Whole-S6 rotation and I401/I405 reorientation"))
cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIG/'panel_B_whole_s6_rotation.png'))); display(Image(filename=str(FIG/'i401_i405_facing_changes.png')))"))
cells.append(nbf.v4.new_markdown_cell("## C. S4–S5 linker movement\n\nThe experimental ~2.4–2.5 Å observation is interpreted as the maximum local linker Cα 3D displacement in the strongly remodeled subunit, not as a centroid radial translation."))
cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIG/'panel_C_linker_radial.png')))"))
cells.append(nbf.v4.new_markdown_cell("## D. F412 and hydrophobic-nexus rearrangement\n\nF412 displacement is an unsigned magnitude; packing-distance changes provide the stronger directional mechanistic test."))
cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIG/'panel_D_f412_displacement.png'))); display(Image(filename=str(FIG/'f412_packing_rearrangement.png')))"))
cells.append(nbf.v4.new_markdown_cell("""## E. Residues 407–411 and π-like geometry

`pi_distance_preference_A = incoming_i4_O_N_distance_A − incoming_i5_O_N_distance_A`; positive values mean the candidate i+5 contact is shorter than i+4. This is a descriptive continuous variable, not a validated structural classifier. The strict binary criterion is not treated as synonymous with a π-like backbone bulge or π-helix formation."""))
cells.append(nbf.v4.new_code_cell("display(pd.read_csv(TAB/'pi_residue_summary.csv')); display(Image(filename=str(FIG/'panel_E_pi_distance_preference.png')))"))
cells.append(nbf.v4.new_markdown_cell("## Compact experimental-recovery overview"))
cells.append(nbf.v4.new_code_cell("display(Image(filename=str(FIG/'experimental_recovery_masked_advantage_heatmap.png')))"))
cells.append(nbf.v4.new_markdown_cell("## Sensitivity analyses"))
cells.append(nbf.v4.new_code_cell("sens=pd.read_csv(TAB/'final_model_sensitivity.csv'); display(sens); print(sens.analysis.value_counts())"))

nb['cells']=cells
nb['metadata']={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3'}}
nbf.write(nb,path)
print(path)
