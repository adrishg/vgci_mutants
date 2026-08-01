#!/usr/bin/env python3
"""Generate the three repository-relative ensemble-RMSF notebooks."""

from pathlib import Path
import nbformat as nbf

CHANNELS = {
    "kv21": {
        "label": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        "title": r"$\mathrm{K}_{\mathrm{V}}2.1$ ensemble RMSF",
        "comparisons": [("wt", "masked"), ("l403a", "masked"), ("f412l", "masked")],
        "zooms": {"l403a": (380, 430, {405: "L403A"}), "f412l": (390, 435, {414: "F412L"})},
    },
    "nav15": {
        "label": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        "title": r"$\mathrm{Na}_{\mathrm{V}}1.5$ ensemble RMSF",
        "comparisons": [
            ("wt", "masked"), ("wt", "masked_v2"), ("wt", "masked_v2_noIFM"),
            ("qqq", "masked"), ("qqq", "masked_v2"),
        ],
        "zooms": {
            "wt": (1140, 1200, {1170: "IFM motif"}),
            "qqq": (1140, 1200, {1170: "QQQ motif"}),
        },
    },
    "cav12": {
        "label": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
        "title": r"$\mathrm{Ca}_{\mathrm{V}}1.2$ ensemble RMSF",
        "comparisons": [
            ("wt", "masked"), ("g402s", "masked"), ("g406r", "masked"), ("g490r", "masked"),
        ],
        "zooms": {
            "wt": (380, 430, {}),
            "g402s": (380, 430, {402: "G402S"}),
            "g406r": (380, 430, {406: "G406R"}),
            "g490r": (465, 515, {490: "G490R"}),
        },
    },
}


def make_notebook(channel, config):
    coverage_note = ""
    if channel == "cav12":
        coverage_note = (
            "\n\nG490R is included here because paired vanilla and masked RMSF profiles are available. "
            "It is not yet present in the current distance or experimental-reference RMSD source tables, "
            "so this notebook does not imply that those additional comparisons have been completed."
        )
    cells = [
        nbf.v4.new_markdown_cell(
            f"# {config['title']}\n\n"
            "This notebook compares structural variability across independently predicted AlphaFold "
            "ensembles generated with the original MSA (vanilla) and with targeted regions of the "
            "MSA masked. The central question is whether masking broadens or narrows the conformational "
            "ensemble locally, and whether those changes extend beyond the residues whose evolutionary "
            "information was removed.\n\n"
            "RMSF is used here as a descriptive measure of variation across independently generated "
            "structures. It should not be interpreted as molecular dynamics fluctuation or as a direct "
            "measure of thermodynamic flexibility."
            + coverage_note
        ),
        nbf.v4.new_markdown_cell(
            "## Definition of ensemble RMSF\n\n"
            "For residue $i$, the ensemble RMSF is calculated from the aligned Cα coordinates of the "
            "$M$ available models:\n\n"
            "$$\n"
            "\\mathrm{RMSF}_i = \\sqrt{\\frac{1}{M}\\sum_{k=1}^{M}"
            "\\left\\lVert\\mathbf{r}_{i,k}-\\overline{\\mathbf{r}}_i\\right\\rVert^2}\n"
            "$$\n\n"
            "Here, $\\mathbf{r}_{i,k}$ is the aligned Cα coordinate of residue $i$ in model $k$, and "
            "$\\overline{\\mathbf{r}}_i$ is its mean coordinate across the ensemble. RMSF is therefore "
            "always non-negative. Larger values indicate that the predicted coordinates occupy a "
            "broader region after common structural alignment.\n\n"
            "The comparison plotted in the lower panels is\n\n"
            "$$\n"
            "\\Delta\\mathrm{RMSF}_i = \\mathrm{RMSF}_{i,\\mathrm{masked}}"
            "-\\mathrm{RMSF}_{i,\\mathrm{vanilla}}.\n"
            "$$\n\n"
            "Positive values indicate broader sampling in the masked ensemble; negative values indicate "
            "broader sampling in vanilla. A positive value is not automatically evidence of a better "
            "prediction—the sampled states must still be assessed against structural references and "
            "channel-state measurements."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nimport pandas as pd\nimport numpy as np\n"
            "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from IPython.display import Image, Markdown, display\n"
            "repo_root=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'scripts'/'ensemble_rmsf_analysis').is_dir())\n"
            "sys.path.insert(0,str(repo_root))\n"
            "from scripts.ensemble_rmsf_analysis.io import load_primary_profile,read_csv_resolving_lfs\n"
            "from scripts.ensemble_rmsf_analysis.comparisons import paired_rmsf_comparison,summarize_comparison\n"
            "from scripts.ensemble_rmsf_analysis.experimental import paired_experimental_differences\n"
            "from scripts.ensemble_rmsf_analysis.topology import TOPOLOGY\n"
            "from shared.plotting import ensemble_protocol_palette,ACCENT_PALETTE\n"
            "import importlib\n"
            "import scripts.ensemble_rmsf_analysis.plotting as rmsf_plotting\n"
            "importlib.reload(rmsf_plotting)\n"
            "plot_whole_protein=rmsf_plotting.plot_whole_protein\n"
            "plot_zoom=rmsf_plotting.plot_zoom\n"
            "plot_whole_protein_lines=rmsf_plotting.plot_whole_protein_lines\n"
            "plot_zoom_lines=rmsf_plotting.plot_zoom_lines\n"
            f"CHANNEL={channel!r}\n"
            f"CHANNEL_LABEL={config['label']!r}\n"
            "CHANNEL_TOPOLOGY=TOPOLOGY[CHANNEL]\n"
            "analysis_root=repo_root/CHANNEL/'dataRMSF'/'analysis'; figures=analysis_root/'figures'; tables=analysis_root/'tables'\n"
            "figures.mkdir(parents=True,exist_ok=True); tables.mkdir(parents=True,exist_ok=True)"
        ),
        nbf.v4.new_markdown_cell(
            "## Data and mask provenance\n\n"
            "The profiles below were calculated from the selected structural ensembles after their "
            "common alignment. Coverage fields report how consistently a residue was resolved across "
            "the contributing models.\n\n"
            "Direct-mask annotations use the supplied authoritative table in raw, 1-based "
            "AlphaFold query numbering. They are not presented as A3M-derived because the "
            "production A3M files are not present in this repository. Yellow background regions in "
            "the residue profiles mark these directly masked positions; they do not indicate statistical "
            "significance or unusually high RMSF."
        ),
        nbf.v4.new_code_cell(
            "profile,schema,profile_path=load_primary_profile(repo_root,CHANNEL)\n"
            "PROFILE_IS_ALL_OK3='all_ok_3' in profile_path.name.lower()\n"
            "display(Markdown(f'**RMSF profile in use:** `{profile_path.relative_to(repo_root)}`'))\n"
            "if not PROFILE_IS_ALL_OK3:\n"
            "    display(Markdown('> **QC warning:** this is the legacy `all_models` profile, not an All-OK-3 RMSF recomputation. Pre-convergence recycle snapshots are included, so long tails and numerical interpretations remain provisional until the cluster-generated `all_ok_3` profile is supplied.'))\n"
            "display(pd.read_csv(tables/f'{CHANNEL}_profile_schema.csv'))\n"
            "mask_path=tables/f'{CHANNEL}_a3m_mask_positions.csv'\n"
            "MASKS_AVAILABLE=mask_path.is_file()\n"
            "if MASKS_AVAILABLE:\n"
            "    mask_table=read_csv_resolving_lfs(mask_path,repo_root)\n"
            "    display(read_csv_resolving_lfs(tables/f'{CHANNEL}_a3m_mask_summary.csv',repo_root))\n"
            "else:\n"
            "    mask_table=pd.DataFrame(columns=['dataset','raw_residue_number','directly_masked'])\n"
            "    print('MASK TABLE MISSING: materialize the supplied authoritative table before running this notebook.')\n"
            "    print('Run: python -m scripts.ensemble_rmsf_analysis.materialize_authoritative_masks')"
        ),
        nbf.v4.new_markdown_cell(
            "## Sequence-mapped channel topology\n\n"
            "The compact blue block track at the bottom of each residue profile maps voltage-sensor helices, pore-forming "
            "segments and selected motifs. The plotted axes use channel sequence numbering, while "
            "the underlying analysis tables retain raw AlphaFold-model numbering. Reviewed UniProt "
            "transmembrane features were transferred by globally aligning each actual model construct "
            "to its canonical sequence: P15387 for Kv2.1, Q14524 for Nav1.5, and Q13936 for Cav1.2. "
            "This step is essential for the shortened Kv2.1 and Nav1.5 constructs.\n\n"
            "UniProt boundaries describe the membrane-spanning core rather than every structurally "
            "helical residue. The Cav1.2 DI-S6 cytosolic extension containing G402 and G406 is shown "
            "separately. Pore-helix or motif-derived annotations remain explicitly provisional in "
            "`scripts/ensemble_rmsf_analysis/topology.py`.\n\n"
            "The blue topology blocks and yellow masking bands encode different information: blue identifies "
            "the sequence-mapped structural segment; yellow identifies residues whose MSA information was directly "
            "masked."
        ),
        nbf.v4.new_code_cell(
            "topology_table=pd.DataFrame(CHANNEL_TOPOLOGY)\n"
            "topology_table.to_csv(tables/f'{CHANNEL}_sequence_mapped_topology.csv',index=False)\n"
            "display(topology_table)"
        ),
        nbf.v4.new_markdown_cell(
            "## Residue-resolved vanilla–masked comparisons\n\n"
            "Each comparison uses the vanilla and masked ensembles for the same protein sequence. "
            "The first representation uses residue bars, which makes individual positions explicit. "
            "The second uses continuous lines with a translucent fill from zero to the RMSF curve. "
            "That fill represents RMSF magnitude, not a confidence interval or standard deviation. "
            "Mutation-centered panels then enlarge the relevant local sequence window."
        ),
        nbf.v4.new_code_cell(
            f"COMPARISONS={config['comparisons']!r}\n"
            f"ZOOMS={config['zooms']!r}\n"
            "from scripts.ensemble_rmsf_analysis.topology import SEQUENCE_NUMBERING\n"
            "NUMBERING=SEQUENCE_NUMBERING[CHANNEL]\n"
            "LOCAL_NUMBER_SHIFT=NUMBERING['display_shift']\n"
            "WHOLE_DISPLAY_WINDOW=NUMBERING['core_display_window']\n"
            "LOCAL_AXIS_LABEL=NUMBERING['axis_label']\n"
            "comparison_tables=[]\n"
            "for condition,protocol in COMPARISONS:\n"
            "    condition_palette=ensemble_protocol_palette(CHANNEL,condition)\n"
            "    display(Markdown(f'### {condition.upper()}: vanilla versus {protocol.replace(\"_\", \" \")}'))\n"
            "    dataset_name=condition+'_'+protocol\n"
            "    mask=set(mask_table.loc[(mask_table.dataset==dataset_name)&mask_table.directly_masked.astype(bool),'raw_residue_number'].astype(int)) if MASKS_AVAILABLE else set()\n"
            "    comparison=paired_rmsf_comparison(profile,condition,protocol,schema['rmsf'],mask)\n"
            "    comparison.to_csv(tables/f'{CHANNEL}_{condition}_{protocol}_paired_rmsf.csv',index=False)\n"
            "    if MASKS_AVAILABLE:\n"
            "        allocation,classes=summarize_comparison(comparison)\n"
            "        allocation.assign(sequence_condition=condition,comparison_protocol=protocol).to_csv(tables/f'{CHANNEL}_{condition}_{protocol}_absolute_change_allocation.csv',index=False)\n"
            "        classes.assign(sequence_condition=condition,comparison_protocol=protocol).to_csv(tables/f'{CHANNEL}_{condition}_{protocol}_mask_classes.csv',index=False)\n"
            "    top=pd.concat([comparison.nlargest(20,'masked_minus_vanilla_rmsf_A'),comparison.nsmallest(20,'masked_minus_vanilla_rmsf_A')]).drop_duplicates()\n"
            "    top.to_csv(tables/f'{CHANNEL}_{condition}_{protocol}_top_rmsf_changes.csv',index=False)\n"
            "    paired_experimental_differences(profile,condition,protocol).to_csv(tables/f'{CHANNEL}_{condition}_{protocol}_experimental_differences.csv',index=False)\n"
            "    display(Markdown('#### Whole-protein residue profile — bar representation'))\n"
            "    whole_path=figures/f'{CHANNEL}_{condition}_{protocol}_whole_protein.png'\n"
            "    plot_whole_protein(comparison,mask,f'{CHANNEL_LABEL} | {condition.upper()} | vanilla vs {protocol}',whole_path,topology=CHANNEL_TOPOLOGY,colors=condition_palette,display_window=WHOLE_DISPLAY_WINDOW,residue_number_shift=LOCAL_NUMBER_SHIFT,residue_axis_label=LOCAL_AXIS_LABEL)\n"
            "    display(Image(filename=str(whole_path),width=1100))\n"
            "    display(Markdown('#### Whole-protein residue profile — line and magnitude-fill representation'))\n"
            "    whole_line_path=figures/f'{CHANNEL}_{condition}_{protocol}_whole_protein_line.png'\n"
            "    plot_whole_protein_lines(comparison,mask,f'{CHANNEL_LABEL} | {condition.upper()} | vanilla vs {protocol}',whole_line_path,topology=CHANNEL_TOPOLOGY,colors=condition_palette,display_window=WHOLE_DISPLAY_WINDOW,residue_number_shift=LOCAL_NUMBER_SHIFT,residue_axis_label=LOCAL_AXIS_LABEL)\n"
            "    display(Image(filename=str(whole_line_path),width=1100))\n"
            "    if condition in ZOOMS:\n"
            "        start,end,annotations=ZOOMS[condition]\n"
            "        display(Markdown(f'#### Mutation-centered window: sequence residues {start+LOCAL_NUMBER_SHIFT}–{end+LOCAL_NUMBER_SHIFT}'))\n"
            "        zoom_path=figures/f'{CHANNEL}_{condition}_{protocol}_zoom.png'\n"
            "        plot_zoom(comparison,mask,start,end,f'{CHANNEL_LABEL} | {condition.upper()} | mutation-centered ensemble RMSF',zoom_path,annotations,topology=CHANNEL_TOPOLOGY,colors=condition_palette,residue_number_shift=LOCAL_NUMBER_SHIFT,residue_axis_label=LOCAL_AXIS_LABEL)\n"
            "        display(Image(filename=str(zoom_path),width=950))\n"
            "        zoom_line_path=figures/f'{CHANNEL}_{condition}_{protocol}_zoom_line.png'\n"
            "        plot_zoom_lines(comparison,mask,start,end,f'{CHANNEL_LABEL} | {condition.upper()} | mutation-centered ensemble RMSF',zoom_line_path,annotations,topology=CHANNEL_TOPOLOGY,colors=condition_palette,residue_number_shift=LOCAL_NUMBER_SHIFT,residue_axis_label=LOCAL_AXIS_LABEL)\n"
            "        display(Image(filename=str(zoom_line_path),width=950))\n"
            "    comparison_tables.append(comparison)\n"
            "print(f'{len(comparison_tables)} paired comparisons are represented above.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Does the effect remain localized to the mask?\n\n"
            "Residues are grouped by their linear sequence distance from the nearest directly masked "
            "position: directly masked, 1–5 residues away, 6–10 residues away, or more than 10 residues "
            "away. This is sequence distance, not three-dimensional spatial distance.\n\n"
            "The first violin plot shows the distribution of $\\Delta\\mathrm{RMSF}$ in each class. "
            "Its internal lines mark the quartiles. The split violins then show the underlying vanilla "
            "and masked RMSF distributions directly: vanilla occupies the left half and masked the "
            "right half. These distributions summarize residues and do not treat neighboring residues "
            "as statistically independent replicates."
        ),
        nbf.v4.new_code_cell(
            "combined=pd.concat(comparison_tables,ignore_index=True)\n"
            "if MASKS_AVAILABLE:\n"
            "    combined['comparison_label']=combined.sequence_condition.str.upper()+' | '+combined.comparison_protocol.str.replace('_',' ',regex=False)\n"
            "    class_order=['directly_masked','adjacent_to_mask_1_to_5','adjacent_to_mask_6_to_10','unmasked']\n"
            "    class_labels={'directly_masked':'Directly masked','adjacent_to_mask_1_to_5':'1–5 residues away','adjacent_to_mask_6_to_10':'6–10 residues away','unmasked':'>10 residues away'}\n"
            "    combined['residue_class']=combined.mask_sequence_class.map(class_labels)\n"
            "    display_order=[class_labels[value] for value in class_order]\n"
            "    fig,ax=plt.subplots(figsize=(max(10,2.25*combined.comparison_label.nunique()),5.8));sns.violinplot(data=combined,x='comparison_label',y='masked_minus_vanilla_rmsf_A',hue='residue_class',hue_order=display_order,cut=0,inner='quart',density_norm='width',common_norm=False,linewidth=.75,ax=ax,palette=[ACCENT_PALETTE['ORANGE'],ACCENT_PALETTE['PEACH'],ACCENT_PALETTE['YELLOW'],ACCENT_PALETTE['CREAM']])\n"
            "    ax.axhline(0,color='.35',lw=.8);ax.set(xlabel='Sequence and masking protocol',ylabel='Masked − vanilla RMSF (Å)',title=f'{CHANNEL_LABEL} | RMSF change by distance from the supplied direct mask');ax.tick_params(axis='x',rotation=16);ax.legend(title='Distance from direct mask',frameon=False,bbox_to_anchor=(1.01,1),loc='upper left');sns.despine();fig.tight_layout();fig.savefig(figures/f'{CHANNEL}_mask_class_distributions.png',dpi=300,bbox_inches='tight');plt.show()\n"
            "    panel_labels=combined.comparison_label.drop_duplicates().tolist();ncols=2 if len(panel_labels)>1 else 1;nrows=int(np.ceil(len(panel_labels)/ncols));fig,axes=plt.subplots(nrows,ncols,figsize=(7.2*ncols,4.7*nrows),sharey=True,squeeze=False)\n"
            "    for panel_index,panel_label in enumerate(panel_labels):\n"
            "        ax=axes.flat[panel_index];part=combined.loc[combined.comparison_label.eq(panel_label)].copy();long=part.melt(id_vars=['residue_class'],value_vars=['vanilla_rmsf_A','masked_rmsf_A'],var_name='ensemble',value_name='rmsf_A');long['ensemble']=long.ensemble.map({'vanilla_rmsf_A':'Vanilla','masked_rmsf_A':'Masked'})\n"
            "        panel_condition=part.sequence_condition.iloc[0];panel_palette=ensemble_protocol_palette(CHANNEL,panel_condition)\n"
            "        sns.violinplot(data=long,x='residue_class',y='rmsf_A',hue='ensemble',order=display_order,hue_order=['Vanilla','Masked'],split=True,inner='quart',cut=0,density_norm='width',common_norm=False,linewidth=.75,palette=panel_palette,ax=ax)\n"
            "        ax.set(title=panel_label,xlabel='',ylabel='Ensemble RMSF (Å)' if panel_index%ncols==0 else '');ax.tick_params(axis='x',rotation=18);ax.legend(title=None,frameon=False,loc='upper right',ncol=2,fontsize=8.5,handlelength=1.5);sns.despine(ax=ax)\n"
            "    for extra in axes.flat[len(panel_labels):]: extra.set_visible(False)\n"
            "    fig.suptitle(f'{CHANNEL_LABEL} | vanilla versus masked RMSF by distance from the supplied direct mask',weight='bold',y=1.01);fig.tight_layout();fig.savefig(figures/f'{CHANNEL}_mask_class_split_vanilla_vs_masked.png',dpi=300,bbox_inches='tight');plt.show()\n"
            "else: print('Mask-class distribution intentionally skipped until A3M extraction succeeds.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Redistribution across proposed channel segments\n\n"
            "To examine whether RMSF changes extend beyond the targeted sequence blocks, the residue-level "
            "$\\Delta\\mathrm{RMSF}$ values are summarized within each provisional topology segment. The "
            "heatmap reports the median change for that segment. Positive cells indicate broader masked "
            "sampling; negative cells indicate broader vanilla sampling.\n\n"
            "This analysis can identify nonlocal redistribution, but it does not establish a directional "
            "upstream/downstream mechanism. A distant segment can change because of structural coupling, "
            "a different global alignment response, altered coverage, or a coherent displacement of the "
            "ensemble. Segment boundaries are provisional, and neighboring residues are not independent "
            "statistical observations."
        ),
        nbf.v4.new_code_cell(
            "segment_rows=[]\n"
            "for comparison in comparison_tables:\n"
            "    condition=str(comparison.sequence_condition.iloc[0]);protocol=str(comparison.comparison_protocol.iloc[0])\n"
            "    for segment in CHANNEL_TOPOLOGY:\n"
            "        part=comparison.loc[comparison.raw_residue_number.between(segment['start'],segment['end'])]\n"
            "        if part.empty: continue\n"
            "        delta=part.masked_minus_vanilla_rmsf_A\n"
            "        segment_rows.append({'condition':condition,'protocol':protocol,'comparison':condition.upper()+' | '+protocol.replace('_',' '),'segment':segment['label'],'domain':segment['domain'],'boundary_confidence':segment['confidence'],'raw_start':segment['start'],'raw_end':segment['end'],'number_of_residues':len(part),'number_directly_masked':int(part.directly_masked.sum()),'fraction_directly_masked':part.directly_masked.mean(),'median_delta_rmsf_A':delta.median(),'mean_delta_rmsf_A':delta.mean(),'fraction_residues_increased':delta.gt(0).mean()})\n"
            "segment_summary=pd.DataFrame(segment_rows)\n"
            "segment_summary.to_csv(tables/f'{CHANNEL}_topology_segment_rmsf_summary.csv',index=False)\n"
            "display(segment_summary.round(3))\n"
            "heat=segment_summary.pivot(index='segment',columns='comparison',values='median_delta_rmsf_A').reindex([segment['label'] for segment in CHANNEL_TOPOLOGY])\n"
            "limit=max(.1,float(np.nanquantile(np.abs(heat.to_numpy()),.95)))\n"
            "fig,ax=plt.subplots(figsize=(max(7,1.65*heat.shape[1]),max(5,.34*heat.shape[0]+1.8)))\n"
            "sns.heatmap(heat,cmap='PuOr_r',center=0,vmin=-limit,vmax=limit,annot=True,fmt='.2f',linewidths=.4,linecolor='white',cbar_kws={'label':'Median masked − vanilla RMSF (Å)'},ax=ax)\n"
            "ax.set(xlabel='Sequence and masking protocol',ylabel='Provisional topology segment',title=f'{CHANNEL_LABEL} | segment-level redistribution of ensemble RMSF');ax.tick_params(axis='x',rotation=20);ax.tick_params(axis='y',rotation=0);fig.tight_layout();fig.savefig(figures/f'{CHANNEL}_topology_segment_delta_heatmap.png',dpi=300,bbox_inches='tight');plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Descriptive summary\n\n"
            "The table reports the median masked-minus-vanilla RMSF inside and outside the direct mask, "
            "together with the fraction of residues for which masking increased RMSF. These values are "
            "compact descriptors of the profiles above; they are not hypothesis-test results.\n\n"
            "A larger inside-mask median supports a localized response to targeted masking. A substantial "
            "outside-mask median or a large fraction of increased residues suggests that the effect "
            "propagates through the predicted structure or through the alignment frame. Experimental "
            "distance and RMSD comparisons are needed to determine whether that broader sampling moves "
            "toward relevant channel conformations."
        ),
        nbf.v4.new_code_cell(
            "summary=[]\n"
            "for (condition,protocol),part in combined.groupby(['sequence_condition','comparison_protocol']):\n"
            "    inside=part.loc[part.directly_masked,'masked_minus_vanilla_rmsf_A'].median() if MASKS_AVAILABLE else np.nan;outside=part.loc[~part.directly_masked,'masked_minus_vanilla_rmsf_A'].median()\n"
            "    summary.append({'condition':condition,'protocol':protocol,'median_delta_inside_mask_A':inside,'median_delta_outside_mask_A':outside,'fraction_residues_increased':part.masked_minus_vanilla_rmsf_A.gt(0).mean()})\n"
            "summary=pd.DataFrame(summary);summary.to_csv(tables/f'{CHANNEL}_automatic_summary.csv',index=False);display(summary)\n"
            "print('Positive values indicate broader ensemble sampling, not inherently improved prediction. Experimental-distance deltas must be evaluated separately.')"
        ),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "bioadri", "language": "python", "name": "python3"}
    })


def main():
    root = Path(__file__).resolve().parents[2]
    for channel, config in CHANNELS.items():
        # RMSF notebooks are channel-level presentation entry points.
        output = root / channel
        output.mkdir(parents=True, exist_ok=True)
        path = output / f"{channel.capitalize()}_ensemble_RMSF.ipynb"
        nbf.write(make_notebook(channel, config), path)
        print(path)


if __name__ == "__main__":
    main()
