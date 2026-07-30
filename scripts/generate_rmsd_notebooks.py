#!/usr/bin/env python3
"""Generate compact, per-condition RMSD notebooks from the shared utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf
import pandas as pd


NAMES = {
    ("kv21", "wt"): "Kv21_WT_experimental_RMSD.ipynb",
    ("kv21", "l403a"): "Kv21_L403A_experimental_RMSD.ipynb",
    ("kv21", "f412l"): "Kv21_F412L_experimental_RMSD.ipynb",
    ("nav15", "wt"): "Nav15_WT_experimental_RMSD.ipynb",
    ("nav15", "qqq"): "Nav15_QQQ_experimental_RMSD.ipynb",
    ("cav12", "wt"): "Cav12_WT_experimental_RMSD.ipynb",
    ("cav12", "g402s"): "Cav12_G402S_experimental_RMSD.ipynb",
    ("cav12", "g406r"): "Cav12_G406R_experimental_RMSD.ipynb",
}


def notebook(channel: str, condition: str, source: Path) -> nbf.NotebookNode:
    channel_label = {
        "kv21": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        "nav15": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        "cav12": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
    }[channel]
    channel_markdown = {
        "kv21": r"$\mathrm{K}_{\mathrm{V}}2.1$",
        "nav15": r"$\mathrm{Na}_{\mathrm{V}}1.5$",
        "cav12": r"$\mathrm{Ca}_{\mathrm{V}}1.2$",
    }[channel]
    title = f"{channel_markdown} | {condition.upper()} | experimental RMSD"
    cells = [
        nbf.v4.new_markdown_cell(
            f"# {title}\n\n"
            "This notebook compares complete RMSD distributions for vanilla and targeted-MSA-masked "
            "ensembles after the established 3 Å convergence filter. For Kv2.1, the convergence "
            "manifest is joined explicitly by PDB basename and rows flagged by the corrected v2 "
            "chain-mapping analysis are excluded. Each experimental reference is kept separate. "
            "Recycle snapshots are correlated, so effect sizes are descriptive and nominal "
            "significance tests are intentionally not emphasized."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\nimport sys\nimport numpy as np\nimport pandas as pd\n"
            "import matplotlib.pyplot as plt\nimport seaborn as sns\n"
            "from matplotlib.lines import Line2D\n"
            "repo_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'shared').is_dir())\n"
            "sys.path.insert(0, str(repo_root))\n"
            "import importlib\nimport shared.rmsd_analysis as rmsd_utils\n"
            "importlib.reload(rmsd_utils)\n"
            "from shared.plotting import (NAV15_EXPERIMENTAL_STYLES,RMSD_REFERENCE_STYLES,"
            "ensemble_protocol_palette,NAV15_PALETTE)\n"
            "from shared.rmsd_analysis import (principal_measurements, protocol_effects, "
            "protocol_label, reference_preference, rmsd_columns, summary_statistics, "
            "summarize_reference_preference, humanize_measurement, "
            "apply_kv21_rmsd_qc)\n"
            f"CHANNEL={channel!r}; CONDITION={condition!r}\n"
            f"SOURCE=repo_root/{str(source.relative_to(source.parents[2]))!r}\n"
            f"OUT=repo_root/{channel!r}/'dataRMSD'/'analysis'/{condition!r}\n"
            "FIG=OUT/'figures'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)\n"
            "header=pd.read_csv(SOURCE,nrows=0)\n"
            "measurements=principal_measurements(header,CHANNEL,12)\n"
            "companions=[]\n"
            "for c in measurements:\n"
            "    stem=c.removesuffix('rmsd_A'); companions += [stem+'matched_atoms',stem+'atom_coverage']\n"
            "chain_measurements=[c for c in rmsd_columns(header) if 's6_bundle_working__chain_' in c and '__ca__core_aligned' in c]\n"
            "metadata=['dataset','sequence_condition','protocol','pdb_file','reference_id','analysis_status','analysis_error']\n"
            "diagnostics=['d4_mapping_rmsd_gap_A','all24_improvement_over_d4_A','selected_core_postfit_rmsd_A']\n"
            "usecols=[c for c in dict.fromkeys(metadata+measurements+companions+chain_measurements+diagnostics) if c in header]\n"
            "df=pd.read_csv(SOURCE,usecols=usecols,low_memory=False)\n"
            "df=df[df.sequence_condition.astype(str).str.lower().eq(CONDITION)].copy()\n"
            "if CHANNEL=='kv21':\n"
            "    manifest_path=repo_root/'kv21/dataRMSF/qc/kv21_all_ok3_selection_manifest.csv'\n"
            "    manifest=pd.read_csv(manifest_path,usecols=['pdb_basename','all_ok_3'])\n"
            "    selected=set(manifest.loc[manifest.all_ok_3.fillna(False),'pdb_basename'].astype(str))\n"
            "    before=len(df); df=df[df.pdb_file.astype(str).map(lambda x: Path(x).name).isin(selected)].copy()\n"
            "    after_convergence=len(df)\n"
            "    if 'analysis_status' in df: df=df[df.analysis_status.eq('ok')].copy()\n"
            "    after_mapping=len(df)\n"
            "    df=apply_kv21_rmsd_qc(df,repo_root)\n"
            "    print(f'Kv2.1 selection: {before:,} rows → {after_convergence:,} allOk3 rows → {after_mapping:,} v2 mapping-QC rows → {len(df):,} final structural-QC rows')\n"
            "if CHANNEL=='nav15': df['protocol']=df['dataset'].str.replace(CONDITION+'_','',regex=False)\n"
            "df['Protocol']=df.protocol.map(protocol_label)\n"
            "print(f'{len(df):,} rows; {df.pdb_file.nunique():,} unique models; {len(measurements)} principal measurements')\n"
            "measurements"
        ),
        nbf.v4.new_code_cell(
            "# EDIT THIS CELL to change publication-facing titles, labels, order, or colors.\n"
            "PLOT_SETTINGS = {\n"
            f"    'channel_label': {channel_label!r},\n"
            f"    'condition_label': {condition.upper()!r},\n"
            "    'reference_order': [x for x in {\n"
            "        'kv21':['8SD3 | WT','8SDA | L403A'],\n"
            "        'nav15':['8VYJ | native open I','8VYK | native open II','7DTC | intermediate-inactivated','6UZ3 | historical','7FBS | engineered QQQ open','8T6L | toxin-bound'],\n"
            "        'cav12':['8HLP','8WE6','8FD7'],\n"
            "    }[CHANNEL]],\n"
            "    'protocol_order': ['Vanilla','Masked','Masked v2','Masked v2 no IFM'],\n"
            "    'colors': ensemble_protocol_palette(CHANNEL,CONDITION),\n"
            f"    'main_title': {channel_label!r}+f\" | {{CONDITION.upper()}} | experimental-reference RMSD\",\n"
            "    'rmsd_axis': 'Cα RMSD (Å)',\n"
            "}\n"
            "if CHANNEL=='nav15':\n"
            "    PLOT_SETTINGS['colors'].update({'Masked v2':NAV15_PALETTE['WT_MASKED_V2'] if CONDITION=='wt' else NAV15_PALETTE['QQQ_MASKED_V2'],"
            "'Masked v2 no IFM':NAV15_PALETTE['WT_MASKED_V2_NOIFM']})\n"
            "REFERENCE_LABELS={\n"
            "    '8SD3':'8SD3 | WT','8SDA':'8SDA | L403A',\n"
            "    '8VYJ':'8VYJ | native open I','8VYK':'8VYK | native open II',\n"
            "    '7DTC':'7DTC | intermediate-inactivated','6UZ3':'6UZ3 | historical',\n"
            "    '7FBS':'7FBS | engineered QQQ open','8T6L':'8T6L | toxin-bound',\n"
            "    '8HLP':'8HLP','8WE6':'8WE6','8FD7':'8FD7',\n"
            "}\n"
            "df['Reference']=df.reference_id.map(REFERENCE_LABELS).fillna(df.reference_id)\n"
            "all_protocols=[x for x in PLOT_SETTINGS['protocol_order'] if x in set(df.Protocol)]\n"
            "main_protocols=([x for x in ['Vanilla','Masked'] if x in set(df.Protocol)] "
            "if CHANNEL=='nav15' else all_protocols)\n"
            "main_df=df[df.Protocol.isin(main_protocols)].copy()\n"
            "active_protocols=main_protocols\n"
            "PLOT_SETTINGS"
        ),
        nbf.v4.new_markdown_cell("## Counts and measurement completeness"),
        nbf.v4.new_code_cell(
            "counts=main_df.groupby(['Protocol','reference_id']).agg(rows=('pdb_file','size'),"
            "unique_models=('pdb_file','nunique')).reset_index()\n"
            "counts.to_csv(OUT/'model_counts.csv',index=False)\n"
            "display(counts)\n"
            "counts['Reference']=counts.reference_id.map(REFERENCE_LABELS).fillna(counts.reference_id)\n"
            "fig,ax=plt.subplots(figsize=(8.5,4.5)); sns.barplot(data=counts,x='Reference',y='unique_models',"
            "hue='Protocol',hue_order=active_protocols,palette=PLOT_SETTINGS['colors'],ax=ax); ax.set(xlabel='',"
            "ylabel='Unique models',title=f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | models retained after 3 Å convergence QC\"); ax.tick_params(axis='x',rotation=20); sns.despine();"
            "fig.tight_layout(); fig.savefig(FIG/'01_model_counts.png',dpi=300); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "missing=pd.DataFrame({'measurement':measurements,"
            "'missing_fraction':[pd.to_numeric(df[c],errors='coerce').isna().mean() for c in measurements]})\n"
            "missing['available_fraction']=1-missing['missing_fraction']\n"
            "missing['label']=missing.measurement.map(humanize_measurement)\n"
            "missing.to_csv(OUT/'coverage_missing_summary.csv',index=False)\n"
            "fig,ax=plt.subplots(figsize=(8,max(3,.32*len(missing)))); sns.barplot(data=missing,y='label',"
            "x='available_fraction',color='#6B6E9E',ax=ax); ax.set(xlim=(0,1.04),xlabel='Fraction measured',ylabel='',"
            "title='Principal-measurement completeness');\n"
            "for patch,value in zip(ax.patches,missing.available_fraction):\n"
            "    ax.text(min(value+.012,1.015),patch.get_y()+patch.get_height()/2,f'{value:.0%}',va='center',fontsize=8)\n"
            "sns.despine(); fig.tight_layout();"
            "fig.savefig(FIG/'02_coverage_missing.png',dpi=300); plt.show()"
        ),
        nbf.v4.new_markdown_cell("## Overall and regional RMSD distributions"),
        nbf.v4.new_code_cell(
            "overall=next((c for c in measurements if any(x in c.lower() for x in ['whole','best_mapping_core','stable_core'])),measurements[0])\n"
            "fig,ax=plt.subplots(figsize=(9.5,5)); sns.violinplot(data=main_df,x='Reference',y=overall,hue='Protocol',"
            "hue_order=active_protocols,palette=PLOT_SETTINGS['colors'],split=len(active_protocols)==2,cut=0,"
            "inner='quart',density_norm='width',common_norm=False,linewidth=.8,ax=ax); ax.set(xlabel='',"
            "ylabel=PLOT_SETTINGS['rmsd_axis'],title=f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | {humanize_measurement(overall)}\"); ax.tick_params(axis='x',rotation=20); sns.despine(); fig.tight_layout();"
            "fig.savefig(FIG/'03_overall_core_distribution.png',dpi=300); plt.show()"
        ),
        nbf.v4.new_code_cell(
            "regional=[c for c in measurements if c!=overall][:6]\n"
            "long=main_df.melt(id_vars=['Reference','Protocol'],value_vars=regional,var_name='measurement',value_name='RMSD')\n"
            "long['Region']=long.measurement.map(humanize_measurement)\n"
            "g=sns.catplot(data=long,x='Reference',y='RMSD',hue='Protocol',hue_order=active_protocols,col='Region',col_wrap=2,"
            "kind='violin',split=True,cut=0,inner='quart',sharey=False,palette=PLOT_SETTINGS['colors'],height=3.5,aspect=1.3)\n"
            "g.set_titles('{col_name}'); g.set_axis_labels('',PLOT_SETTINGS['rmsd_axis']);\n"
            "for ax in g.axes.flat: ax.tick_params(axis='x',rotation=22)\n"
            "g.figure.savefig(FIG/'04_regional_distributions.png',dpi=300,bbox_inches='tight'); plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Core-aligned versus locally aligned RMSD\n\n"
            "A high core-aligned value with a low locally aligned value is consistent with a mostly "
            "rigid displacement relative to the channel core. High values in both indicate additional "
            "internal deformation."
        ),
        nbf.v4.new_code_cell(
            "pairs=[]\nfor c in rmsd_columns(df):\n"
            "    if '__core_aligned_rmsd_A' in c:\n"
            "        local=c.replace('__core_aligned_rmsd_A','__local_aligned_rmsd_A')\n"
            "        if local in df: pairs.append((c,local))\n"
            "if pairs:\n"
            "    core,local=pairs[0]; sample=main_df.sample(min(len(main_df),12000),random_state=403)\n"
            "    if CHANNEL=='nav15':\n"
            "        protocols=[x for x in ['Vanilla','Masked'] if x in set(sample.Protocol)]\n"
            "        references=[x for x in ['7FBS','6UZ3','8VYJ','8VYK','7DTC','8T6L'] if x in set(sample.reference_id)]\n"
            "        sample['local_minus_core_A']=pd.to_numeric(sample[local],errors='coerce')-pd.to_numeric(sample[core],errors='coerce')\n"
            "        fig,ax=plt.subplots(figsize=(10.8,5.5))\n"
            "        sns.violinplot(data=sample,x='reference_id',y='local_minus_core_A',hue='Protocol',"
            "order=references,hue_order=protocols,palette=PLOT_SETTINGS['colors'],split=len(protocols)==2,"
            "inner='quart',cut=0,linewidth=.75,density_norm='width',ax=ax)\n"
            "        ax.axhline(0,color='#6F6875',lw=.9,ls=':')\n"
            "        ymin,ymax=ax.get_ylim(); marker_y=ymax-(ymax-ymin)*.035\n"
            "        for index,reference in enumerate(references):\n"
            "            style=NAV15_EXPERIMENTAL_STYLES[reference]\n"
            "            ax.scatter(index,marker_y,s=38,marker=style['marker'],facecolor='white',"
            "edgecolor=style['color'],linewidth=1.2,zorder=8)\n"
            "        ax.set(xlabel='Experimental RMSD reference',ylabel='Local RMSD − core-aligned RMSD (Å)')\n"
            "        ax.set_xticklabels([REFERENCE_LABELS[r] for r in references],rotation=20,ha='right')\n"
            "        handles,labels=ax.get_legend_handles_labels(); ax.get_legend().remove()\n"
            "        fig.suptitle(f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | pore-domain local relaxation\",fontweight='semibold',y=.99)\n"
            "        fig.legend(handles,labels,title='Protocol',loc='upper center',bbox_to_anchor=(.5,.94),"
            "ncol=2,frameon=False)\n"
            "        sns.despine(ax=ax); fig.tight_layout(rect=(0,0,1,.88))\n"
            "        fig.savefig(FIG/'05_core_vs_local.png',dpi=300,bbox_inches='tight')\n"
            "        # Preserve the raw two-dimensional relationship as supplemental QC.\n"
            "        fig_qc,axes=plt.subplots(1,len(protocols),figsize=(5.6*len(protocols),4.8),squeeze=False,sharex=True,sharey=True)\n"
            "        for axis,protocol in zip(axes.flat,protocols):\n"
            "            part=sample[sample.Protocol.eq(protocol)]\n"
            "            for reference,refpart in part.groupby('reference_id',sort=False):\n"
            "                style=NAV15_EXPERIMENTAL_STYLES[reference]\n"
            "                axis.scatter(refpart[core],refpart[local],s=8,alpha=.18,marker=style['marker'],color=style['color'],linewidths=0)\n"
            "            axis.set(title=protocol,xlabel='Core-aligned RMSD (Å)',ylabel='Locally aligned RMSD (Å)'); sns.despine(ax=axis)\n"
            "        fig_qc.suptitle('Supplemental QC | raw core-versus-local RMSD',fontweight='semibold')\n"
            "        fig_qc.tight_layout(); fig_qc.savefig(FIG/'S3_core_vs_local_scatter_QC.png',dpi=300,bbox_inches='tight')\n"
            "    else:\n"
            "        protocols=[x for x in ['Vanilla','Masked'] if x in set(sample.Protocol)]\n"
            "        references=[r.split(' | ')[0] for r in PLOT_SETTINGS['reference_order'] "
            "if r.split(' | ')[0] in set(sample.reference_id)]\n"
            "        fig,axes=plt.subplots(1,len(protocols),figsize=(5.7*len(protocols),5),"
            "squeeze=False,sharex=True,sharey=True)\n"
            "        values=sample[[core,local]].apply(pd.to_numeric,errors='coerce').to_numpy();"
            " lo=np.nanmin(values); hi=np.nanmax(values)\n"
            "        for ax,protocol in zip(axes.flat,protocols):\n"
            "            part=sample[sample.Protocol.eq(protocol)]\n"
            "            for reference in references:\n"
            "                refpart=part[part.reference_id.eq(reference)];"
            " style=RMSD_REFERENCE_STYLES[reference]\n"
            "                ax.scatter(refpart[core],refpart[local],s=16,alpha=.48,"
            "marker=style['marker'],facecolor=style['color'],edgecolor='white',linewidth=.22,"
            "rasterized=True)\n"
            "            ax.plot([lo,hi],[lo,hi],ls='--',lw=.85,color='.4',zorder=0)\n"
            "            ax.set(title=protocol,xlabel='Core-aligned RMSD (Å)',"
            "ylabel='Locally aligned RMSD (Å)');"
            " ax.title.set_color('#2F3136'); ax.title.set_weight('bold');"
            " sns.despine(ax=ax)\n"
            "        handles=[Line2D([0],[0],linestyle='none',marker=RMSD_REFERENCE_STYLES[r]['marker'],"
            "markerfacecolor=RMSD_REFERENCE_STYLES[r]['color'],markeredgecolor='white',markersize=7,"
            "label=REFERENCE_LABELS[r]) for r in references]\n"
            "        fig.suptitle(humanize_measurement(core),fontweight='semibold');"
            " fig.legend(handles=handles,title='Experimental RMSD reference',loc='lower center',"
            "bbox_to_anchor=(.5,-.01),ncol=len(handles),frameon=False)\n"
            "        fig.tight_layout(rect=(0,.09,1,.95));"
            " fig.savefig(FIG/'05_core_vs_local.png',dpi=300,bbox_inches='tight')\n"
            "    plt.show()\n"
            "else: print('No matched core/local RMSD pair is available.')"
        ),
        nbf.v4.new_markdown_cell(
            "## Effect of masking within each experimental reference\n\n"
            "The **protocol median difference** is\n\n"
            "$$\\Delta_{\\mathrm{protocol}}=\\mathrm{median}(RMSD_{\\mathrm{masked},R})"
            "-\\mathrm{median}(RMSD_{\\mathrm{vanilla},R}).$$\n\n"
            "It compares protocols against the same reference $R$. Negative values mean that the "
            "masked ensemble is closer to that reference; positive values mean that vanilla is "
            "closer. This quantity does **not** say whether a model is more WT-like or mutant-like."
        ),
        nbf.v4.new_code_cell(
            "stats=summary_statistics(df,measurements); effects=protocol_effects(df,measurements)\n"
            "main_effects=protocol_effects(main_df,measurements)\n"
            "stats.to_csv(OUT/'summary_statistics.csv',index=False); effects.to_csv(OUT/'effect_sizes.csv',index=False)\n"
            "display(main_effects.sort_values('masked_minus_vanilla_median',key=abs,ascending=False).head(15))"
        ),
        nbf.v4.new_code_cell(
            "plot=main_effects.dropna(subset=['masked_minus_vanilla_median']).copy();"
            "plot['label']=plot.measurement.map(humanize_measurement);"
            "plot['comparison']=plot.reference_id.astype(str)+' | '+plot.comparison_protocol.astype(str)\n"
            "if CHANNEL=='nav15':\n"
            "    focus=[\n"
            "      ('pore_domain__ca__core_aligned_rmsd_A','Pore domain'),\n"
            "      ('DII_s6__ca__core_aligned_rmsd_A','DII S6'),\n"
            "      ('ifm_motif__ca__core_aligned_rmsd_A','IFM/QQQ motif'),\n"
            "      ('ifm_receptor_pocket__ca__core_aligned_rmsd_A','IFM receptor pocket'),\n"
            "    ]\n"
            "    focus=[item for item in focus if item[0] in set(plot.measurement)]\n"
            "    references=[x for x in ['7FBS','6UZ3','8VYJ','8VYK','7DTC','8T6L'] if x in set(plot.reference_id)]\n"
            "    fig,axes=plt.subplots(2,2,figsize=(10.2,6.6)); axes=axes.flat;"
            " offsets=np.linspace(-.18,.18,len(references))\n"
            "    for ax,(measurement,region_label) in zip(axes,focus):\n"
            "        for offset,reference in zip(offsets,references):\n"
            "            row=plot[(plot.measurement==measurement)&(plot.reference_id==reference)]\n"
            "            if row.empty: continue\n"
            "            row=row.iloc[0]; value=row.masked_minus_vanilla_median\n"
            "            low=row.bootstrap_ci_low; high=row.bootstrap_ci_high\n"
            "            style=NAV15_EXPERIMENTAL_STYLES[reference]\n"
            "            ax.errorbar(value,offset,xerr=[[value-low],[high-value]],fmt=style['marker'],"
            "ms=6,mfc='white',mec=style['color'],mew=1.25,ecolor=style['color'],elinewidth=.8,capsize=2,zorder=4)\n"
            "        ax.axvline(0,color='#6F6875',lw=1,ls=':'); ax.set_yticks([])\n"
            "        ax.set(xlabel='Masked − vanilla median RMSD (Å)',ylabel='',title=region_label)\n"
            "        ax.grid(axis='x',color='#EEE9F2',lw=.45,ls='--');sns.despine(ax=ax,left=True)\n"
            "    handles=[Line2D([0],[0],linestyle='none',marker=NAV15_EXPERIMENTAL_STYLES[r]['marker'],"
            "markerfacecolor='white',markeredgecolor=NAV15_EXPERIMENTAL_STYLES[r]['color'],markersize=6,"
            "label=REFERENCE_LABELS[r]) for r in references]\n"
            "    fig.suptitle(f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | regional effect of the original mask\","
            "fontweight='semibold',y=.99)\n"
            "    fig.legend(handles=handles,title='Experimental reference',bbox_to_anchor=(.5,-.01),"
            "loc='lower center',ncol=3,frameon=False)\n"
            "    fig.text(.08,.15,'← closer to the experimental reference',ha='left',fontsize=8.5,color='#5F5666')\n"
            "    fig.text(.92,.15,'farther from the experimental reference →',ha='right',fontsize=8.5,color='#5F5666')\n"
            "    fig.tight_layout(rect=(0,.20,1,.94))\n"
            "    fig.savefig(FIG/'06_median_differences.png',dpi=300,bbox_inches='tight')\n"
            "    # Retain the complete regional effect forest as supplemental QC.\n"
            "    full_measurements=list(dict.fromkeys(plot.measurement));"
            " full_labels=[humanize_measurement(x) for x in full_measurements]\n"
            "    fig_full,ax_full=plt.subplots(figsize=(8,max(5,.58*len(full_measurements))));"
            " full_offsets=np.linspace(-.24,.24,len(references))\n"
            "    for region_index,measurement in enumerate(full_measurements):\n"
            "        for offset,reference in zip(full_offsets,references):\n"
            "            row=plot[(plot.measurement==measurement)&(plot.reference_id==reference)]\n"
            "            if row.empty: continue\n"
            "            style=NAV15_EXPERIMENTAL_STYLES[reference]\n"
            "            ax_full.scatter(row.masked_minus_vanilla_median.iloc[0],region_index+offset,s=34,"
            "marker=style['marker'],facecolor='white',edgecolor=style['color'],linewidth=1.1,zorder=4)\n"
            "    ax_full.axvline(0,color='#6F6875',lw=.9,ls=':');"
            " ax_full.set_yticks(range(len(full_measurements)),full_labels);ax_full.invert_yaxis()\n"
            "    ax_full.set(xlabel='Masked − vanilla median RMSD (Å)',ylabel='',"
            "title='Supplemental | all regional masking effects');"
            " ax_full.grid(axis='x',color='#EEE9F2',lw=.45,ls='--')\n"
            "    ax_full.legend(handles=handles,title='Experimental reference',bbox_to_anchor=(.5,-.08),"
            "loc='upper center',ncol=3,frameon=False);sns.despine(ax=ax_full);"
            " fig_full.tight_layout(rect=(0,.08,1,1));"
            " fig_full.savefig(FIG/'S4_all_regional_median_differences.png',dpi=300,bbox_inches='tight')\n"
            "else:\n"
            "    labels=list(dict.fromkeys(plot.label)); references=[r for r in PLOT_SETTINGS['reference_order'] if r.split(' | ')[0] in set(plot.reference_id)]\n"
            "    reference_ids=[r.split(' | ')[0] for r in references]; offsets=np.linspace(-.16,.16,len(reference_ids))\n"
            "    fig,ax=plt.subplots(figsize=(8,max(4,.43*len(labels))));\n"
            "    for label_index,label in enumerate(labels):\n"
            "        for offset,reference in zip(offsets,reference_ids):\n"
            "            row=plot[(plot.label==label)&(plot.reference_id==reference)]\n"
            "            if row.empty: continue\n"
            "            row=row.iloc[0]; value=row.masked_minus_vanilla_median\n"
            "            style=RMSD_REFERENCE_STYLES[reference]\n"
            "            ax.errorbar(value,label_index+offset,xerr=[[value-row.bootstrap_ci_low],[row.bootstrap_ci_high-value]],"
            "fmt=style['marker'],ms=6.2,mfc='white',mec=style['color'],mew=1.35,"
            "ecolor=style['color'],elinewidth=.8,capsize=2,zorder=4)\n"
            "    ax.axvline(0,color='#7A737D',lw=.8,ls=':');ax.set_yticks(range(len(labels)),labels);ax.invert_yaxis()\n"
            "    ax.set(xlabel='Masked − vanilla median RMSD (Å)',ylabel='',"
            "title=f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | masking effect by experimental reference\")\n"
            "    handles=[Line2D([0],[0],linestyle='none',marker=RMSD_REFERENCE_STYLES[r]['marker'],"
            "markerfacecolor='white',markeredgecolor=RMSD_REFERENCE_STYLES[r]['color'],markersize=6,"
            "label=REFERENCE_LABELS[r]) for r in reference_ids]\n"
            "    ax.legend(handles=handles,title='Experimental reference',loc='best',frameon=False);"
            "ax.grid(axis='x',color='#EEE9F2',lw=.4,ls='--');sns.despine();fig.tight_layout();"
            "    fig.savefig(FIG/'06_median_differences.png',dpi=300)\n"
            "plt.show()"
        ),
        nbf.v4.new_code_cell(
            "heat_source=main_effects.assign(column=main_effects.reference_id.map(REFERENCE_LABELS).fillna(main_effects.reference_id),Region=main_effects.measurement.map(humanize_measurement))\n"
            "heat=heat_source.pivot_table(index='Region',columns='column',"
            "values='cliffs_delta_masked_vs_vanilla')\n"
            "fig,ax=plt.subplots(figsize=(6,max(4,.32*len(heat)))); sns.heatmap(heat,cmap='vlag',center=0,"
            "vmin=-1,vmax=1,ax=ax,cbar_kws={'label':\"Cliff's delta\"}); ax.set(xlabel='Reference',ylabel='',"
            "title='Robust protocol effect sizes'); ax.tick_params(axis='x',rotation=28);"
            "plt.setp(ax.get_xticklabels(),ha='right'); fig.tight_layout();"
            "fig.savefig(FIG/'07_effect_size_heatmap.png',dpi=300); plt.show()"
        ),
        nbf.v4.new_markdown_cell(
            "## Which experimental structure does each model resemble?\n\n"
            "This is a paired, per-model comparison. For references A and B,\n\n"
            "$$\\Delta_{B-A}=RMSD_B-RMSD_A.$$\n\n"
            "A negative value means the model is closer to reference B; a positive value means it "
            "is closer to reference A. The scatterplot shows the two RMSDs directly, with the "
            "identity line marking equal resemblance. Histograms show the signed preference score, "
            "and the table reports the fraction of paired models closer to either reference. "
            "These comparisons measure structural resemblance, not functional state by themselves."
        ),
        nbf.v4.new_code_cell(
            "REFERENCE_PAIRS={\n"
            " 'kv21':[('8SD3','8SDA')],\n"
            " 'nav15':[('8VYJ','7FBS'),('8VYK','7FBS'),('8VYJ','7DTC')],\n"
            " 'cav12':[('8HLP','8WE6'),('8HLP','8FD7')],\n"
            "}[CHANNEL]\n"
            "preference_tables=[]; preference_summaries=[]\n"
            "for ref_a,ref_b in REFERENCE_PAIRS:\n"
            "    paired=reference_preference(main_df,overall,ref_a,ref_b)\n"
            "    if paired.empty:\n"
            "        print(f'No complete paired observations for {ref_a} versus {ref_b}.'); continue\n"
            "    summary=summarize_reference_preference(paired)\n"
            "    preference_tables.append(paired); preference_summaries.append(summary)\n"
            "    safe=f'{ref_a}_vs_{ref_b}'\n"
            "    paired.to_csv(OUT/f'reference_preference_{safe}.csv',index=False)\n"
            "    summary.to_csv(OUT/f'reference_preference_summary_{safe}.csv',index=False)\n"
            "    fig,(ax_scatter,ax_hist)=plt.subplots(1,2,figsize=(11.5,4.6),gridspec_kw={'width_ratios':[1,1.15]})\n"
            "    sample=paired.sample(min(len(paired),12000),random_state=412)\n"
            "    sns.scatterplot(data=sample,x='rmsd_reference_a_A',y='rmsd_reference_b_A',hue='Protocol',"
            "hue_order=active_protocols,palette=PLOT_SETTINGS['colors'],s=16,alpha=.32,linewidth=0,ax=ax_scatter)\n"
            "    lo=np.nanmin(sample[['rmsd_reference_a_A','rmsd_reference_b_A']].to_numpy());"
            " hi=np.nanmax(sample[['rmsd_reference_a_A','rmsd_reference_b_A']].to_numpy())\n"
            "    ax_scatter.plot([lo,hi],[lo,hi],ls='--',lw=1,color='.35');"
            " ax_scatter.set(xlabel=f'RMSD to {REFERENCE_LABELS.get(ref_a,ref_a)} (Å)',"
            "ylabel=f'RMSD to {REFERENCE_LABELS.get(ref_b,ref_b)} (Å)')\n"
            "    sns.histplot(data=paired,x='delta_b_minus_a_A',hue='Protocol',hue_order=active_protocols,"
            "palette=PLOT_SETTINGS['colors'],element='step',fill=False,stat='density',common_norm=False,"
            "linewidth=1.8,ax=ax_hist)\n"
            "    ax_hist.axvline(0,ls='--',lw=1,color='.35');"
            " ax_hist.set(xlabel=f\"RMSD({REFERENCE_LABELS.get(ref_b,ref_b)}) − RMSD({REFERENCE_LABELS.get(ref_a,ref_a)}) (Å)\",ylabel='Density')\n"
            "    sns.move_legend(ax_hist,'lower right',title='Protocol',frameon=True)\n"
            "    ax_hist.text(.02,.98,f\"← closer to {REFERENCE_LABELS.get(ref_b,ref_b)}\",transform=ax_hist.transAxes,ha='left',va='top',fontsize=9)\n"
            "    ax_hist.text(.98,.98,f\"closer to {REFERENCE_LABELS.get(ref_a,ref_a)} →\",transform=ax_hist.transAxes,ha='right',va='top',fontsize=9)\n"
            "    resemblance_label='WT (8SD3) versus L403A mutant (8SDA)' if CHANNEL=='kv21' else 'experimental-reference'\n"
            "    fig.suptitle(f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | {resemblance_label} resemblance\",fontweight='bold')\n"
            "    sns.despine(); fig.tight_layout(); fig.savefig(FIG/f'reference_preference_{safe}.png',dpi=300,bbox_inches='tight'); plt.show()\n"
            "if preference_summaries:\n"
            "    preference_summary=pd.concat(preference_summaries,ignore_index=True)\n"
            "    display(preference_summary)\n"
            "else: preference_summary=pd.DataFrame()"
        ),
        nbf.v4.new_markdown_cell("## Experimental-baseline entry\n\nNo separate reference-to-reference baseline is assumed. If a compatible baseline column is present in the table, the next cell reports entry fractions; otherwise it records that this analysis is unavailable."),
        nbf.v4.new_code_cell(
            "baseline_cols=[c for c in df if 'baseline' in c.lower() and ('rmsd' in c.lower() or 'range' in c.lower())]\n"
            "if baseline_cols:\n"
            "    baseline_note=pd.DataFrame({'available_column':baseline_cols})\n"
            "else:\n"
            "    baseline_note=pd.DataFrame({'status':['Separate experimental reference-to-reference baseline unavailable.']})\n"
            "baseline_note.to_csv(OUT/'experimental_baseline_status.csv',index=False); display(baseline_note)\n"
            "fig,ax=plt.subplots(figsize=(7,1.7)); ax.axis('off'); ax.text(.5,.5,baseline_note.iloc[0,0],"
            "ha='center',va='center',wrap=True); fig.tight_layout();"
            "fig.savefig(FIG/'08_experimental_baseline_status.png',dpi=300); plt.show()"
        ),
    ]
    if channel == "kv21":
        cells.extend([
            nbf.v4.new_markdown_cell("## Kv2.1 chain asymmetry and mapping QC"),
            nbf.v4.new_code_cell(
                "chain_cols=[c for c in rmsd_columns(df) if 's6_bundle_working__chain_' in c and '__ca__core_aligned' in c]\n"
                "if chain_cols:\n"
                "    chain_long=df.melt(id_vars=['Protocol','reference_id'],value_vars=chain_cols,var_name='chain',value_name='RMSD')\n"
                "    chain_long['chain']=chain_long.chain.str.extract(r'chain_([A-D])')\n"
                "    fig,ax=plt.subplots(figsize=(8,5)); sns.violinplot(data=chain_long,x='chain',y='RMSD',hue='Protocol',"
                "hue_order=['Vanilla','Masked'],palette=PLOT_SETTINGS['colors'],split=True,cut=0,inner='quart',"
                "density_norm='width',linewidth=.75,ax=ax); ax.set(title='Chain-resolved S6 displacement',"
                "xlabel='Chain',ylabel='Core-aligned RMSD (Å)'); sns.despine(); fig.tight_layout();"
                "fig.savefig(FIG/'09_chain_resolved_s6.png',dpi=300); plt.show()\n"
                "std=df[chain_cols].std(axis=1); asym=pd.DataFrame({'asymmetry_std_A':std,'Protocol':df.Protocol,"
                "'reference_id':df.reference_id}); fig,ax=plt.subplots(figsize=(8,4.5));"
                "sns.violinplot(data=asym,x='reference_id',y='asymmetry_std_A',hue='Protocol',"
                "hue_order=active_protocols,cut=0,"
                "inner='quart',palette=PLOT_SETTINGS['colors'],ax=ax); ax.set(title='S6 chain asymmetry',"
                "xlabel='Reference',ylabel='Across-chain RMSD SD (Å)'); sns.despine();fig.tight_layout();"
                "fig.savefig(FIG/'10_chain_asymmetry.png',dpi=300);plt.show()"
            ),
            nbf.v4.new_code_cell(
                "if 'd4_mapping_rmsd_gap_A' in df:\n"
                "    fig,ax=plt.subplots(figsize=(8,4.5)); sns.histplot(data=df,x='d4_mapping_rmsd_gap_A',hue='Protocol',"
                "hue_order=active_protocols,"
                "element='step',stat='density',common_norm=False,palette=PLOT_SETTINGS['colors'],ax=ax);"
                "    ax.set(title='Cyclic chain-mapping separation',xlabel='Best-to-second mapping RMSD gap (Å)');"
                "    sns.despine();fig.tight_layout();fig.savefig(FIG/'11_mapping_gap_qc.png',dpi=300);plt.show()"
            ),
        ])
    if channel == "nav15":
        if condition == "qqq":
            cells.extend([
                nbf.v4.new_markdown_cell(
                    "## WT versus QQQ experimental-reference resemblance\n\n"
                    "This comparison separates sequence and protocol effects. Each column compares "
                    "WT with QQQ under the same MSA treatment. The upper panels show the paired RMSD "
                    "to 8VYJ and 7FBS; the lower panels show "
                    "$\\Delta_{7FBS-8VYJ}=RMSD_{7FBS}-RMSD_{8VYJ}$. Negative values indicate greater "
                    "resemblance to the 7FBS QQQ/open pore."
                ),
                nbf.v4.new_code_cell(
                    "cross_cols=['dataset','sequence_condition','protocol','pdb_file','reference_id',overall]\n"
                    "cross=pd.read_csv(SOURCE,usecols=cross_cols,low_memory=False)\n"
                    "cross['Protocol']=cross.protocol.map(protocol_label)\n"
                    "cross['Sequence']=cross.sequence_condition.astype(str).str.lower().map({'wt':'WT','qqq':'QQQ'})\n"
                    "sequence_colors={\n"
                    " ('Vanilla','WT'):'#D9D7F2',('Vanilla','QQQ'):'#F2CDE8',\n"
                    " ('Masked','WT'):'#6764B8',('Masked','QQQ'):'#C24F94',\n"
                    "}\n"
                    "fig,axes=plt.subplots(2,2,figsize=(11.8,8.2),sharex='row')\n"
                    "comparison_tables=[]\n"
                    "for column,protocol in enumerate(['Vanilla','Masked']):\n"
                    "    protocol_data=cross[cross.Protocol.eq(protocol)].copy()\n"
                    "    paired=reference_preference(protocol_data,overall,'8VYJ','7FBS')\n"
                    "    paired['Sequence']=paired.sequence_condition.astype(str).str.lower().map({'wt':'WT','qqq':'QQQ'})\n"
                    "    comparison_tables.append(paired)\n"
                    "    colors={sequence:sequence_colors[(protocol,sequence)] for sequence in ['WT','QQQ']}\n"
                    "    sample=paired.sample(min(len(paired),10000),random_state=1170+column)\n"
                    "    sns.scatterplot(data=sample,x='rmsd_reference_a_A',y='rmsd_reference_b_A',hue='Sequence',"
                    "hue_order=['WT','QQQ'],palette=colors,s=13,alpha=.28,linewidth=0,ax=axes[0,column])\n"
                    "    lo=np.nanmin(sample[['rmsd_reference_a_A','rmsd_reference_b_A']].to_numpy());"
                    " hi=np.nanmax(sample[['rmsd_reference_a_A','rmsd_reference_b_A']].to_numpy())\n"
                    "    axes[0,column].plot([lo,hi],[lo,hi],ls='--',lw=.9,color='.4')\n"
                    "    axes[0,column].set(title=protocol,xlabel='RMSD to 8VYJ native open (Å)',"
                    "ylabel='RMSD to 7FBS QQQ/open pore (Å)')\n"
                    "    sns.histplot(data=paired,x='delta_b_minus_a_A',hue='Sequence',hue_order=['WT','QQQ'],"
                    "palette=colors,element='step',fill=False,stat='density',common_norm=False,linewidth=1.8,"
                    "ax=axes[1,column])\n"
                    "    axes[1,column].axvline(0,ls='--',lw=.9,color='.4')\n"
                    "    axes[1,column].set(xlabel='RMSD(7FBS) − RMSD(8VYJ) (Å)',ylabel='Density')\n"
                    "    for axis in axes[:,column]: sns.despine(ax=axis)\n"
                    "fig.suptitle(f\"{PLOT_SETTINGS['channel_label']} | WT versus IFM→QQQ reference resemblance\","
                    "fontweight='semibold',y=.99)\n"
                    "fig.text(.5,.01,'Negative preference values indicate a pore closer to 7FBS; positive values indicate a pore closer to 8VYJ.',"
                    "ha='center',fontsize=8.5,color='#5F5666')\n"
                    "fig.tight_layout(rect=(0,.04,1,.96));"
                    "fig.savefig(FIG/'reference_preference_WT_vs_QQQ.png',dpi=300,bbox_inches='tight');plt.show()\n"
                    "pd.concat(comparison_tables,ignore_index=True).to_csv(OUT/'reference_preference_WT_vs_QQQ.csv',index=False)"
                ),
            ])
        cells.extend([
            nbf.v4.new_markdown_cell(
                "## Supplemental: alternative masking profiles\n\n"
                "The main analysis above is restricted to vanilla and the original mask. "
                "Masked v2 and masked v2 no-IFM are retained here as mask-design controls. "
                "They should not be mixed into the primary protocol comparison."
            ),
            nbf.v4.new_code_cell(
                "fig,ax=plt.subplots(figsize=(10,5)); sns.violinplot(data=df,x='Reference',y=overall,"
                "hue='Protocol',hue_order=all_protocols,palette=PLOT_SETTINGS['colors'],cut=0,"
                "inner='quart',linewidth=.75,ax=ax); ax.set(xlabel='',ylabel=PLOT_SETTINGS['rmsd_axis'],"
                "title=f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | supplemental mask-design comparison\");"
                "ax.tick_params(axis='x',rotation=20); sns.despine(); fig.tight_layout();"
                "fig.savefig(FIG/'S1_all_mask_profiles_overall_RMSD.png',dpi=300,bbox_inches='tight'); plt.show()"
            ),
            nbf.v4.new_code_cell(
                "supp=effects[effects.comparison_protocol.isin(['Masked v2','Masked v2 no IFM'])].copy()\n"
                "if not supp.empty:\n"
                "    supp['column']=supp.reference_id.astype(str)+' | '+supp.comparison_protocol.astype(str)\n"
                "    supp['Region']=supp.measurement.map(humanize_measurement)\n"
                "    matrix=supp.pivot_table(index='Region',columns='column',values='masked_minus_vanilla_median')\n"
                "    fig,ax=plt.subplots(figsize=(7,max(4,.32*len(matrix)))); sns.heatmap(matrix,cmap='vlag',"
                "center=0,ax=ax,cbar_kws={'label':'Alternative mask − vanilla median RMSD (Å)'});"
                "    ax.set(xlabel='',ylabel='',title='Supplemental alternative-mask effects');"
                "    fig.tight_layout(); fig.savefig(FIG/'S2_alternative_mask_effects.png',dpi=300,bbox_inches='tight'); plt.show()\n"
                "else: print('No alternative Nav1.5 mask profiles are present for this condition.')"
            ),
        ])
    cells.append(nbf.v4.new_markdown_cell(
        "## Interpretation and limitations\n\n"
        "The effect-size table should be read together with measurement coverage and chain-mapping "
        "quality control. A lower masked median means that the masked ensemble is closer to the "
        "specified experimental geometry for that coordinate; it does not by itself establish a "
        "functional state. Agreement across complementary regions is stronger evidence than an "
        "isolated RMSD shift, particularly when core-aligned and locally aligned measurements differ."
    ))
    return nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "bioadri", "language": "python", "name": "python3"}
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--channel", required=True, choices=("kv21", "nav15", "cav12"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    head = pd.read_csv(args.source, usecols=["sequence_condition"])
    conditions = sorted(head.sequence_condition.astype(str).str.lower().unique())
    # Analysis notebooks are the primary entry points for each channel, so keep
    # them at the channel root beside the existing distance notebooks.
    out = args.repo_root / args.channel
    out.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        name = NAMES.get((args.channel, condition), f"{args.channel}_{condition}_experimental_RMSD.ipynb")
        nbf.write(notebook(args.channel, condition, args.source.resolve()), out / name)
        print(out / name)


if __name__ == "__main__":
    main()
