"""Configure plot notebooks for consistent 300-dpi writing-folder exports."""

from __future__ import annotations

import json
from pathlib import Path


TARGETS = {
    "kv21": (
        "Kv21_F412L_mutationSite_analysis.ipynb",
        "Kv21_L403A_mutationSite_analysis.ipynb",
        "Kv21_distanceDistribution_vsExperimental.ipynb",
    ),
    "nav15": (
        "Nav15_IFM_latching_analysis.ipynb",
        "Nav15_QQQ_mutationSite_analysis.ipynb",
        "Nav15_distanceDistribution_vsExperimental.ipynb",
    ),
    "cav12": (
        "Cav12_G402S_mutationSite_analysis.ipynb",
        "Cav12_G406R_mutationSite_analysis.ipynb",
        "Cav12_distanceDistribution_vsExperimental.ipynb",
    ),
}


def export_cell(channel: str, notebook_name: str) -> dict:
    label = Path(notebook_name).stem
    source = (
        "from shared.presentation_export import install_notebook_figure_export\n"
        "WRITING_FIGURE_DIR = repo_root.parent / 'vgci_mutants_writing' / "
        f"'figures' / '{channel}'\n"
        f"install_notebook_figure_export(WRITING_FIGURE_DIR, '{label}')\n"
    )
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": ["presentation-export"]},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def configure(path: Path, channel: str) -> bool:
    notebook = json.loads(path.read_text())
    cells = notebook["cells"]
    if any("presentation-export" in cell.get("metadata", {}).get("tags", []) for cell in cells):
        return False
    setup_index = next(
        index for index, cell in enumerate(cells)
        if cell.get("cell_type") == "code"
        and "repo_root" in "".join(cell.get("source", []))
    )
    cells.insert(setup_index + 1, export_cell(channel, path.name))
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return True


def improve_rmsf_panel_layout(path: Path) -> bool:
    notebook = json.loads(path.read_text())
    changed = False
    old = (
        "ncols=2 if len(panel_labels)>1 else 1;"
        "nrows=int(np.ceil(len(panel_labels)/ncols));"
        "fig,axes=plt.subplots(nrows,ncols,figsize=(7.2*ncols,4.7*nrows)"
    )
    new = (
        "ncols=min(3,len(panel_labels));"
        "nrows=int(np.ceil(len(panel_labels)/ncols));"
        "fig,axes=plt.subplots(nrows,ncols,figsize=(6.0*ncols,4.7*nrows)"
    )
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if old in source:
            source = source.replace(old, new)
            cell["source"] = source.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def standardize_nav15_distance_palette(path: Path) -> bool:
    """Use the canonical purple ensembles and PDB encodings in legacy overviews."""
    notebook = json.loads(path.read_text())
    changed = False
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        revised = source
        revised = revised.replace(
            "from shared.plotting import NAV15_PALETTE",
            "from shared.plotting import NAV15_PALETTE, NAV15_EXPERIMENTAL_STYLES",
        )
        revised = revised.replace(
            'NAV15_EXPERIMENTAL_COLORS = ["#E57373", "#F48FB1", "#F06292", "#FF8A65"]',
            "NAV15_EXPERIMENTAL_COLORS = "
            "[NAV15_EXPERIMENTAL_STYLES[pdb]['color'] "
            "for pdb in ('7FBS','6UZ3','8VYJ','8VYK')]",
        )
        revised = revised.replace(
            "pdb_colors = ['#AC5336', '#D77614','#F99E09','#AC5336','#AB524D','#6BD0A3']",
            "pdb_colors = NAV15_EXPERIMENTAL_COLORS[:2]",
        )
        revised = revised.replace(
            "pdb_colors = ['#AC5336', '#D77614','#F99E09','#AC5336','#AB524D','#6BD0A3','#50C878','#88D499']",
            "pdb_colors = NAV15_EXPERIMENTAL_COLORS[:2]",
        )
        revised = revised.replace(
            "pdb_colors = ['#ff5733', '#33ff57', '#3375ff', '#ff33c4', '#ffd633', '#33fff5']",
            "pdb_colors = NAV15_EXPERIMENTAL_COLORS[:2]",
        )
        if "plot_distances_by_alias_violin" in revised and "colors=custom_colors" in revised:
            palette_key = "WT_VAN" if "df_vanilla" in revised and "df_mask" not in revised else "WT_MASKED_V2"
            revised = revised.replace(
                "colors=custom_colors",
                f"colors=[NAV15_PALETTE['{palette_key}']] * len(alias_dict)",
            )
        if revised != source:
            cell["source"] = revised.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def clarify_kv21_reference_figure(path: Path) -> bool:
    notebook = json.loads(path.read_text())
    changed = False
    replacements = {
        "ax_hist.set(xlabel=f'RMSD({ref_b}) − RMSD({ref_a}) (Å)',ylabel='Density')":
            "ax_hist.set(xlabel=f\"RMSD({REFERENCE_LABELS.get(ref_b,ref_b)}) − "
            "RMSD({REFERENCE_LABELS.get(ref_a,ref_a)}) (Å)\",ylabel='Density')",
        "f'← closer to {ref_b}'": "f\"← closer to {REFERENCE_LABELS.get(ref_b,ref_b)}\"",
        "f'closer to {ref_a} →'": "f\"closer to {REFERENCE_LABELS.get(ref_a,ref_a)} →\"",
        "fig.suptitle(f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | reference resemblance\",fontweight='bold')":
            "fig.suptitle(f\"{PLOT_SETTINGS['channel_label']} {PLOT_SETTINGS['condition_label']} | "
            "whole matched tetramer (Cα, core-aligned) | experimental-reference resemblance\","
            "fontweight='bold')",
    }
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        revised = source
        for old, new in replacements.items():
            revised = revised.replace(old, new)
        if revised != source:
            cell["source"] = revised.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def use_split_regional_rmsd_violins(path: Path) -> bool:
    notebook = json.loads(path.read_text())
    changed = False
    old = "kind='violin',cut=0,inner='quart',sharey=False"
    new = "kind='violin',split=True,cut=0,inner='quart',sharey=False"
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if old in source:
            cell["source"] = source.replace(old, new).splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def restyle_core_local_reference_scatter(path: Path) -> bool:
    notebook = json.loads(path.read_text())
    changed = False
    old = (
        "    else:\n"
        "        fig,ax=plt.subplots(figsize=(6.5,5.5)); "
        "sns.scatterplot(data=sample,x=core,y=local,hue='Protocol',"
        "style='reference_id',palette=PLOT_SETTINGS['colors'],s=13,alpha=.32,ax=ax);"
        "        ax.set(title=core.split('__')[0].replace('_',' ').title(),"
        "xlabel='Core-aligned RMSD (Å)',ylabel='Locally aligned RMSD (Å)'); sns.despine()\n"
        "        fig.tight_layout()\n"
        "        fig.savefig(FIG/'05_core_vs_local.png',dpi=300,bbox_inches='tight')\n"
    )
    new = (
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
        " ax.title.set_color('#2F3136');"
        " ax.title.set_weight('bold'); sns.despine(ax=ax)\n"
        "        handles=[Line2D([0],[0],linestyle='none',"
        "marker=RMSD_REFERENCE_STYLES[r]['marker'],"
        "markerfacecolor=RMSD_REFERENCE_STYLES[r]['color'],markeredgecolor='white',"
        "markersize=7,label=REFERENCE_LABELS[r]) for r in references]\n"
        "        fig.suptitle(humanize_measurement(core),fontweight='semibold');"
        " fig.legend(handles=handles,title='Experimental RMSD reference',loc='lower center',"
        "bbox_to_anchor=(.5,-.01),ncol=len(handles),frameon=False)\n"
        "        fig.tight_layout(rect=(0,.09,1,.95));"
        " fig.savefig(FIG/'05_core_vs_local.png',dpi=300,bbox_inches='tight')\n"
    )
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        revised = source.replace(old, new)
        revised = revised.replace(
            "ax.title.set_color(PLOT_SETTINGS['colors'][protocol])",
            "ax.title.set_color('#2F3136')",
        )
        if revised != source:
            cell["source"] = revised.splitlines(keepends=True)
            changed = True
    if changed:
        path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
    return changed


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    configured = 0
    for channel, names in TARGETS.items():
        for name in names:
            configured += configure(root / channel / name, channel)
    layouts = sum(
        improve_rmsf_panel_layout(root / channel / f"{prefix}_ensemble_RMSF.ipynb")
        for channel, prefix in (("kv21", "Kv21"), ("nav15", "Nav15"), ("cav12", "Cav12"))
    )
    nav_palette = standardize_nav15_distance_palette(
        root / "nav15" / "Nav15_distanceDistribution_vsExperimental.ipynb"
    )
    kv_reference = clarify_kv21_reference_figure(
        root / "kv21" / "Kv21_WT_experimental_RMSD.ipynb"
    )
    split_regional = sum(
        use_split_regional_rmsd_violins(path)
        for channel in ("kv21", "nav15", "cav12")
        for path in (root / channel).glob("*_experimental_RMSD.ipynb")
    )
    core_local = sum(
        restyle_core_local_reference_scatter(path)
        for channel in ("kv21", "cav12")
        for path in (root / channel).glob("*_experimental_RMSD.ipynb")
    )
    print(
        f"Configured exports in {configured} notebooks; improved {layouts} RMSF layouts; "
        f"Nav1.5 palette updated={nav_palette}; Kv2.1 reference labels updated={kv_reference}; "
        f"split regional RMSD notebooks={split_regional}; core/local scatters={core_local}."
    )


if __name__ == "__main__":
    main()
