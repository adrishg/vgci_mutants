#!/usr/bin/env python3
"""Keep the Nav1.5 main figure concise and move alternate masks to supplements."""

from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "nav15" / "Nav15_main_and_supplemental_figures.ipynb"

nb = nbformat.read(PATH, as_version=4)

# Keep this update idempotent: discard any previously inserted copies of main
# figures 2 and 3 before rebuilding the concise presentation sequence.
cleaned = []
skip_next = False
for cell in nb.cells:
    if (
        cell.cell_type == "markdown"
        and (
            cell.source.startswith("## Main Figure 2:")
            or cell.source.startswith("## Main Figure 3:")
        )
    ):
        skip_next = True
        continue
    if skip_next and cell.cell_type == "code":
        skip_next = False
        continue
    cleaned.append(cell)
nb.cells = cleaned

nb.cells[0].source = """# $\\mathrm{Na}_{\\mathrm{V}}1.5$ main and supplemental figures

The main-text analysis uses **WT vanilla**, **QQQ vanilla**, and the **original QQQ
masked protocol**. WT vanilla provides the native evolutionary baseline; QQQ
vanilla isolates the sequence substitution under the same MSA treatment; and QQQ
masked tests whether relaxing that evolutionary prior reveals additional
mutant-associated pore geometry.

Masked v2, WT masked v2, and the no-IFM control remain in the supplemental
figures. They are important mask-design controls, but including them in the main
panel obscures the primary sequence-versus-masking comparison."""

nb.cells[1].source = """from pathlib import Path
import sys
import importlib
import pandas as pd
import matplotlib.pyplot as plt

repo_root = Path.cwd() if (Path.cwd()/"shared").is_dir() else Path.cwd().parent
sys.path.insert(0,str(repo_root)) if str(repo_root) not in sys.path else None
import shared.nav15_presentation as nav15_presentation
importlib.reload(nav15_presentation)
from shared.plotting import NAV15_PALETTE, apply_kv21_style
from shared.nav15_pore_shape import experimental_gate_shape
from shared.nav15_presentation import (
    presentation_table, plot_main_mask_tradeoff, plot_selected_qqq_rmsd,
    plot_selected_qqq_reference_preference, plot_supplemental_mask_audit,
    plot_supplemental_pocket_profile,
)
apply_kv21_style()
data_dir=repo_root/'nav15'/'dataDistances'
figure_dir=repo_root.parent/'vgci_mutants_writing'/'figures'/'nav15'
figure_dir.mkdir(parents=True,exist_ok=True)"""

nb.cells[3].source = """## Main Figure 1: mutation and selected-mask distance distributions

The three columns preserve the logic of the comparison. WT vanilla is the native
reference, QQQ vanilla shows the mutation with an intact MSA, and QQQ masked shows
the additional response after relaxing the evolutionary prior. The points over
the two pore panels are experimental structures; they are not additional model
ensembles."""

nb.cells[4].source = """fig=plot_main_mask_tradeoff(summary,models,palette,experimental_shape)
fig.savefig(figure_dir/'Nav15_main_selected_mask_distances.png',dpi=600,bbox_inches='tight')
fig.savefig(figure_dir/'Nav15_main_selected_mask_distances.pdf',bbox_inches='tight')
plt.show()"""

insert_at = 5
new_cells = [
    nbformat.v4.new_markdown_cell("""## Main Figure 2: resemblance to the experimental QQQ/open pore

These panels compare QQQ vanilla directly with the selected original QQQ mask.
RMSD is calculated after alignment to the stable channel core. Lower values mean
that the modeled pore region more closely resembles the corresponding region of
7FBS. The DII-S6 panel is especially informative because it contains the
largest mask-associated improvement."""),
    nbformat.v4.new_code_cell("""rmsd_source=repo_root/'nav15'/'dataRMSD'/'Nav15_all_models_all_references_RMSD_distances_OK3.csv'
rmsd_columns=[
    'sequence_condition','protocol','reference_id','pdb_file',
    'pore_domain__ca__core_aligned_rmsd_A',
    'DII_s6__ca__core_aligned_rmsd_A',
]
rmsd=pd.read_csv(rmsd_source,usecols=rmsd_columns,low_memory=False)
fig=plot_selected_qqq_rmsd(rmsd,palette)
fig.savefig(figure_dir/'Nav15_main_QQQ_7FBS_RMSD.png',dpi=600,bbox_inches='tight')
fig.savefig(figure_dir/'Nav15_main_QQQ_7FBS_RMSD.pdf',bbox_inches='tight')
plt.show()"""),
    nbformat.v4.new_markdown_cell("""## Main Figure 3: paired experimental-reference preference

For each model, the signed score is

$$\\Delta_{7FBS-8VYJ}=RMSD_{7FBS}-RMSD_{8VYJ}.$$

Negative values indicate a pore closer to the engineered QQQ/open 7FBS
reference; positive values indicate a pore closer to the native-open 8VYJ
reference. This is a structural resemblance score, not a direct functional-state
assignment."""),
    nbformat.v4.new_code_cell("""from shared.rmsd_analysis import reference_preference
preference=reference_preference(rmsd,'pore_domain__ca__core_aligned_rmsd_A','8VYJ','7FBS')
fig=plot_selected_qqq_reference_preference(preference,palette)
fig.savefig(figure_dir/'Nav15_main_QQQ_reference_preference.png',dpi=600,bbox_inches='tight')
fig.savefig(figure_dir/'Nav15_main_QQQ_reference_preference.pdf',bbox_inches='tight')
plt.show()"""),
]
nb.cells[insert_at:insert_at] = new_cells

# The original supplemental cells move down but retain every alternate mask.
for cell in nb.cells:
    if cell.cell_type == "markdown" and cell.source.startswith("## Supplemental figure S1"):
        cell.source = """## Supplemental Figure S1: all mask designs

All WT and QQQ masking profiles are retained here to show sensitivity to mask
design. This includes masked v2 and the WT no-IFM control, which are deliberately
excluded from the main-text comparison."""

nbformat.write(nb, PATH)
print(PATH)
