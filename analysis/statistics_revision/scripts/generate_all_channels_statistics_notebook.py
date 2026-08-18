"""Build one executed statistics notebook covering CaV1.2, Kv2.1, and Nav1.5."""

from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "notebooks/statistics/01_distance_distribution_statistics.ipynb"
KV_NOTEBOOK = ROOT / "analysis/statistics_revision/Kv21_sampling_breadth_statistics.ipynb"
OUTPUT = ROOT / "analysis/statistics_revision/All_channels_reproducible_statistics.ipynb"


source = nbf.read(SOURCE, as_version=4)
kv = nbf.read(KV_NOTEBOOK, as_version=4)

# Preserve the complete all-channel executable analysis, but give the combined
# artifact an unambiguous scope and remove stale execution state.
cells = [nbf.v4.new_markdown_cell("""
# Reproducible distribution and breadth statistics for all channels

This master notebook covers **CaV1.2, Kv2.1, and Nav1.5**. The first part recalculates the trajectory-aware distance-distribution comparisons for every registered WT, mutant, vanilla, and masking-protocol combination. The second part adds the First100, revised S6, and RMSF uncertainty calculations that are specifically supported for Kv2.1.

Statistical meanings are kept separate throughout:

- Wasserstein distance measures **distributional separation**, not breadth.
- IQR or SD ratios measure **relative breadth**; values above 1 mean the numerator ensemble is broader.
- Recycle snapshots are never described as independent biological replicates.
- Channel-specific coordinates are analyzed only where their source definition and QC selection exist.
""".strip())]

# Skip only the source title cell; retain its methods, registry, calculations,
# tables, figures, and validation cells verbatim.
cells.extend(source.cells[1:])
# Make repository discovery independent of the combined notebook's location.
for cell in cells:
    if cell.cell_type == "code" and "repo_root = Path.cwd() if" in cell.source:
        cell.source = cell.source.replace(
            "repo_root = Path.cwd() if (Path.cwd() / 'shared').is_dir() else Path.cwd().parents[1]",
            "repo_root = Path.cwd()\nwhile not (repo_root / 'shared').is_dir():\n    if repo_root.parent == repo_root:\n        raise FileNotFoundError('Could not locate vgci_mutants repository root')\n    repo_root = repo_root.parent",
        )
        break
for cell in cells:
    if cell.cell_type == "code" and "def load_registry_row(row):" in cell.source:
        cell.source = cell.source.replace(
            "distance_csv_options(repo_root, row.original_path, row.channel, row.condition, row.protocol)",
            "distance_csv_options(repo_root, repo_root / row.original_path, row.channel, row.condition, row.protocol)",
        )
        break
cells.extend([
    nbf.v4.new_markdown_cell("""
# Cross-channel RMSF heatmaps

These heatmaps summarize the existing final-QC RMSF analyses for all three channels. Values are **masked minus vanilla RMSF in Å**. Positive values mean greater positional dispersion under masking; negative values mean greater dispersion under vanilla. They are RMSF effects—not Wasserstein distances and not IQR ratios.

The compact heatmap separates directly masked residues from the rest of the protein. The topology heatmaps show the median residue-level RMSF change within each annotated segment. Each channel uses its project palette and its own symmetric color limit, so compare printed numbers—not color intensity—between panels. Blank cells mean that a segment/comparison combination is unavailable.
""".strip()),
    nbf.v4.new_code_cell("""
from matplotlib.colors import LinearSegmentedColormap

rmsf_tables = {}
rmsf_summary_parts = []
for channel in ['cav12', 'kv21', 'nav15']:
    table = pd.read_csv(repo_root / channel / 'dataRMSF/analysis/tables' / f'{channel}_topology_segment_rmsf_summary.csv')
    table.insert(0, 'channel', channel)
    rmsf_tables[channel] = table
    compact = pd.read_csv(repo_root / channel / 'dataRMSF/analysis/tables' / f'{channel}_automatic_summary.csv')
    compact.insert(0, 'channel', channel)
    compact['ensemble'] = channel + ' | ' + compact.condition.str.upper() + ' | ' + compact.protocol.str.replace('_', ' ')
    rmsf_summary_parts.append(compact)

rmsf_compact = pd.concat(rmsf_summary_parts, ignore_index=True)
rmsf_topology = pd.concat(rmsf_tables.values(), ignore_index=True)
rmsf_compact.to_csv(repo_root / 'analysis/statistics_revision/tables/all_channels_rmsf_region_summary.csv', index=False)
rmsf_topology.to_csv(repo_root / 'analysis/statistics_revision/tables/all_channels_rmsf_topology_summary.csv', index=False)

palette_pairs = {
    'cav12': (CAV12_PALETTE['WT_VAN'], CAV12_PALETTE['WT_HM']),
    'kv21': (KV21_PALETTE['WT_VAN'], KV21_PALETTE['WT_HM']),
    'nav15': (NAV15_PALETTE['WT_VAN'], NAV15_PALETTE['WT_HM']),
}

fig, axes = plt.subplots(1, 3, figsize=(15, 5.4), gridspec_kw={'width_ratios':[1, 1, 1.3]})
for ax, channel in zip(axes, ['cav12', 'kv21', 'nav15']):
    part = rmsf_compact[rmsf_compact.channel.eq(channel)].set_index('ensemble')
    matrix = part[['median_delta_inside_mask_A', 'median_delta_outside_mask_A']]
    matrix.columns = ['Directly masked', 'Outside mask']
    limit = max(.05, np.nanmax(np.abs(matrix.to_numpy())))
    light, dark = palette_pairs[channel]
    cmap = LinearSegmentedColormap.from_list(f'{channel}_rmsf_delta', [light, '#FFFFFF', dark])
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, center=0, vmin=-limit, vmax=limit,
                linewidths=.7, linecolor='white', cbar_kws={'label':'Median ΔRMSF (Å)'}, ax=ax)
    ax.set_title(channel + ' | RMSF by mask region', fontsize=13, fontweight='bold')
    ax.set_xlabel('Residue region'); ax.set_ylabel(''); ax.tick_params(axis='x', rotation=15)
fig.tight_layout()
rmsf_figure_dir = repo_root / 'analysis/statistics_revision/figures'
rmsf_figure_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(rmsf_figure_dir / 'all_channels_rmsf_mask_region_heatmaps.png', dpi=300, bbox_inches='tight')
fig.savefig(rmsf_figure_dir / 'all_channels_rmsf_mask_region_heatmaps.pdf', bbox_inches='tight')
display(fig); plt.close(fig)
""".strip()),
    nbf.v4.new_code_cell("""
fig, axes = plt.subplots(1, 3, figsize=(20, 10), constrained_layout=True)
for ax, channel in zip(axes, ['cav12', 'kv21', 'nav15']):
    table = rmsf_tables[channel]
    matrix = table.pivot(index='segment', columns='comparison', values='median_delta_rmsf_A')
    limit = max(.05, np.nanmax(np.abs(matrix.to_numpy())))
    light, dark = palette_pairs[channel]
    cmap = LinearSegmentedColormap.from_list(f'{channel}_topology_delta', [light, '#FFFFFF', dark])
    sns.heatmap(matrix, annot=True, fmt='.2f', cmap=cmap, center=0, vmin=-limit, vmax=limit,
                linewidths=.35, linecolor='white', cbar_kws={'label':'Median ΔRMSF (Å)'}, ax=ax)
    ax.set_title(channel + ' | topology redistribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Sequence / masking protocol'); ax.set_ylabel('Topology segment')
    ax.tick_params(axis='x', rotation=35, labelsize=9); ax.tick_params(axis='y', labelsize=9)
fig.savefig(rmsf_figure_dir / 'all_channels_rmsf_topology_heatmaps.png', dpi=300, bbox_inches='tight')
fig.savefig(rmsf_figure_dir / 'all_channels_rmsf_topology_heatmaps.pdf', bbox_inches='tight')
display(fig); plt.close(fig)

assert set(rmsf_compact.channel) == {'cav12', 'kv21', 'nav15'}
assert set(rmsf_topology.channel) == {'cav12', 'kv21', 'nav15'}
print(f'RMSF region rows: {len(rmsf_compact)}; topology rows: {len(rmsf_topology)}')
""".strip()),
])
cells.append(nbf.v4.new_markdown_cell("""
# Additional reproducible uncertainty analyses supported for Kv2.1

The following cells are copied from the dedicated Kv2.1 uncertainty notebook. They do not imply that an S6 or First100 estimator was available for CaV1.2 or Nav1.5. The current executable S6 estimator also does not reproduce the historical 3.50/2.08/1.92 values, so its intervals are labeled revised estimates.
""".strip()))

# The dedicated notebook's cells establish their own repository root and imports.
cells.extend(kv.cells[1:])
for cell in cells:
    if cell.cell_type == "code":
        cell.outputs = []
        cell.execution_count = None

combined = nbf.v4.new_notebook(cells=cells, metadata=source.metadata)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(combined, OUTPUT)
print(OUTPUT)
