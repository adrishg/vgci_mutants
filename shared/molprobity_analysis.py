"""Audited local MolProbity analysis; no coordinate generation or data upload.

Snapshot rows are observations, not independent replicates. Inference averages
recycles within AF model, then AF models within seed. Existing QC selectors,
palettes, metadata parser, and paired bootstrap are reused.
"""
from __future__ import annotations

import hashlib
import itertools
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.formula.api as smf

from shared.data_access import resolve
from shared.dataset_selection import select_manifest_rows
from shared.distribution_statistics import parse_trajectory_metadata
from shared.paired_seed_statistics import paired_common_seed_bootstrap
from shared.plotting import (CAV12_PALETTE, KV21_PALETTE, NAV15_PALETTE,
                             apply_kv21_style, format_channel_title)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ['clashscore', 'n_bad_clashes', 'max_overlap']
LABELS = dict(clashscore='Clashscore', n_bad_clashes='Number of bad clashes',
              max_overlap='Maximum overlap (Å)')
PREFIX = dict(cav12='Cav12', kv21='Kv21', nav15='Nav15')
TITLES = dict(cav12='Cav1.2', kv21='Kv2.1', nav15='Nav1.5')
GENOTYPES = dict(cav12=['WT', 'G402S', 'G406R'], kv21=['WT', 'L403A', 'F412L'],
                 nav15=['WT', 'QQQ'])
PROTOCOLS = ['vanilla', 'masked', 'masked v2', 'masked v2 no-IFM']
SNAPSHOTS = [f'r{i}' for i in range(11)] + ['final']
BOOTSTRAPS = 3000
RANDOM_SEED = 20260930


def output_dir(channel):
    path = ROOT / channel / 'dataExtra' / 'molprobity'
    for folder in ['tables', 'figures']:
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def load_molprobity(channel, data_root=None):
    """Only data-access boundary for new tables; local by design.

    VGIC_MOLPROBITY_ROOT points to the downloaded directory, not the repo.
    Future migration: replace discovery/read_csv here with an explicit manifest
    and shared.data_access.resolve paths. No downstream functions need change.
    """
    root = Path(data_root or os.environ.get('VGIC_MOLPROBITY_ROOT', ROOT / 'molprobity_download'))
    files = sorted(root.rglob('*.csv'))
    selected = [p for p in files if p.name.startswith(PREFIX[channel] + '_')]
    if not selected:
        raise FileNotFoundError(f'No {PREFIX[channel]} CSVs in {root}; set VGIC_MOLPROBITY_ROOT')
    frames, inventory = [], []
    required = {'run_name', 'input_folder', 'model_name', 'rank', 'af_model', 'seed',
                'recycle', 'snapshot', *METRICS, 'status', 'phenix_exit_code',
                'elapsed_seconds', 'source_path'}
    for path in selected:
        d = pd.read_csv(path)  # bad CSV lines raise; never skipped
        missing = required - set(d)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        if not d.run_name.eq(path.stem).all():
            raise ValueError(f'{path}: run_name does not match the CSV identity')
        run = path.stem.removesuffix('_clashscores')
        _, genotype, protocol = run.split('_', 2)
        protocol = {'maskedv2': 'masked v2', 'maskedv2_noIFM': 'masked v2 no-IFM'}.get(protocol, protocol)
        if genotype.upper() not in GENOTYPES[channel] or protocol not in PROTOCOLS:
            raise ValueError(f'Unrecognized ensemble: {run}')
        d['genotype'], d['protocol'] = genotype.upper(), protocol
        d['ensemble'] = genotype.upper() + ' | ' + protocol
        d['csv_file'] = str(path.relative_to(root))
        frames.append(d)
        inventory.append(dict(csv_file=str(path.relative_to(root)), rows=len(d),
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return pd.concat(frames, ignore_index=True), pd.DataFrame(inventory), len(files)


def describe(frame, cohort):
    rows = []
    for ensemble, d in frame.groupby('ensemble', sort=False):
        for metric in METRICS:
            x = d[metric].replace([np.inf, -np.inf], np.nan).dropna()
            q = x.quantile([.25, .5, .75])
            rows.append(dict(cohort=cohort, ensemble=ensemble, metric=metric, N=len(x),
                             n_seeds=d.seed.nunique(), median=q[.5], Q1=q[.25], Q3=q[.75],
                             IQR=q[.75]-q[.25], mean=x.mean(), SD=x.std(), minimum=x.min(), maximum=x.max()))
    return pd.DataFrame(rows)


def audit_data(raw, channel, out):
    """Write complete anomalies and expected-grid omissions before allowing plots."""
    d = raw.copy()
    numeric = ['rank', 'af_model', 'seed', 'recycle', *METRICS, 'phenix_exit_code', 'elapsed_seconds']
    malformed = pd.Series(False, index=d.index)
    for col in numeric:
        converted = pd.to_numeric(d[col], errors='coerce')
        malformed |= d[col].notna() & converted.isna()
        d[col] = converted
    identity = d.model_name.astype(str).str.extract(r'_model_(?P<model>\d+)_seed_(?P<seed>\d+)(?:\.r(?P<recycle>\d+))?\.pdb$')
    bad_id = (pd.to_numeric(identity.model, errors='coerce').ne(d.af_model)
              | pd.to_numeric(identity.seed, errors='coerce').ne(d.seed)
              | identity.recycle.fillna('final').map(lambda x: 'r'+x if x != 'final' else x).ne(d.snapshot)
              | pd.to_numeric(identity.recycle, errors='coerce').fillna(-1).ne(d.recycle.fillna(-1))
              | d.source_path.astype(str).map(lambda x: Path(x).name).ne(d.model_name)
              | ~d.snapshot.isin(SNAPSHOTS))
    filename_rank = pd.to_numeric(d.model_name.str.extract(r'_rank_(\d+)_', expand=False), errors='coerce')
    bad_id |= filename_rank.fillna(-1).ne(d['rank'].fillna(-1))
    duplicate = d.duplicated(['ensemble', 'model_name'], keep=False)
    duplicate_key = d.duplicated(['ensemble', 'seed', 'af_model', 'snapshot'], keep=False)
    bad_values = ~np.isfinite(d[METRICS]).all(axis=1) | d[METRICS].lt(0).any(axis=1)
    failed = d.status.ne('ok') | d.phenix_exit_code.ne(0)
    d['analysis_eligible'] = ~(malformed | bad_id | duplicate | duplicate_key | bad_values | failed)
    anomalies = d.assign(
        malformed_numeric=malformed, invalid_identity=bad_id, duplicate_name=duplicate,
        duplicate_key=duplicate_key, invalid_metric=bad_values, failed_status_or_exit=failed).loc[~d.analysis_eligible]
    anomalies.to_csv(out/'tables/row_anomalies.csv', index=False)
    missing_rows, audits, manifest_diffs = [], [], []
    for ensemble, part in d.groupby('ensemble', sort=False):
        genotype, protocol = ensemble.split(' | ')
        token = {'masked v2':'masked_v2', 'masked v2 no-IFM':'masked_v2_noIFM'}.get(protocol, protocol)
        manifest = ROOT/channel/'rmsd_convergence_filtering'/f'{PREFIX[channel]}_rmsd_convergence_{genotype.lower()}_{token}'/'all_models_manifest.csv'
        m = pd.read_csv(manifest)
        names = set(part.model_name)
        for name in set(m.pdb_basename) - names:
            manifest_diffs.append(dict(ensemble=ensemble, model_name=name, issue='manifest model absent from MolProbity'))
        for name in names - set(m.pdb_basename):
            manifest_diffs.append(dict(ensemble=ensemble, model_name=name, issue='MolProbity model absent from manifest'))
        grid = pd.MultiIndex.from_product([sorted(part.seed.dropna().unique()), range(1,6), SNAPSHOTS], names=['seed','af_model','snapshot'])
        actual = pd.MultiIndex.from_frame(part[['seed','af_model','snapshot']])
        absent = grid.difference(actual)
        for seed, model, snap in absent:
            missing_rows.append(dict(ensemble=ensemble, seed=seed, af_model=model, snapshot=snap))
        both = part[part.snapshot.eq('final')].merge(part[part.snapshot.eq('r10')], on=['seed','af_model'], suffixes=('_final','_r10'), validate='one_to_one')
        equal = np.logical_and.reduce([both[x+'_final'].eq(both[x+'_r10']) for x in METRICS])
        audits.append(dict(ensemble=ensemble, rows=len(part), ok=int(part.status.eq('ok').sum()),
                           ok_fraction=part.status.eq('ok').mean(), eligible=int(part.analysis_eligible.sum()),
                           failed=int(failed.loc[part.index].sum()), missing_clashscore=int(part.clashscore.isna().sum()),
                           nonfinite_clashscore=int((~np.isfinite(part.clashscore)).sum()),
                           duplicate_names=int(duplicate.loc[part.index].sum()), duplicate_keys=int(duplicate_key.loc[part.index].sum()),
                           missing_grid_snapshots=len(absent), seeds=part.seed.nunique(),
                           seed_min=part.seed.min(), seed_max=part.seed.max(), missing_rank=int(part['rank'].isna().sum()),
                           final_r10_pairs=len(both), final_r10_identical_metrics=int(equal.sum()),
                           manifest_rows=len(m), manifest_missing=len(set(m.pdb_basename)-names),
                           manifest_extra=len(names-set(m.pdb_basename))))
    audit = pd.DataFrame(audits)
    audit.to_csv(out/'tables/data_audit.csv', index=False)
    pd.DataFrame(missing_rows, columns=['ensemble','seed','af_model','snapshot']).to_csv(out/'tables/missing_snapshots.csv', index=False)
    pd.DataFrame(manifest_diffs, columns=['ensemble','model_name','issue']).to_csv(out/'tables/manifest_discrepancies.csv', index=False)
    if (malformed | bad_id | duplicate | duplicate_key).any():
        raise ValueError('Malformed or ambiguous rows: inspect tables/row_anomalies.csv before proceeding')
    if manifest_diffs:
        raise ValueError('Manifest identity mismatch; see manifest_discrepancies.csv')
    print(f'Audit: {len(d):,} rows, {failed.sum()} failed, {bad_values.sum()} invalid metric rows; '
          f'{len(missing_rows)} absent expected snapshots; {len(anomalies)} explicitly excluded rows.')
    # Reuse the project parser, preserving final/base status rather than inventing a recycle.
    d = parse_trajectory_metadata(d.assign(pdb_file=d.model_name))
    return d, audit


def select_cohorts(data, channel, out):
    good = data[data.analysis_eligible].copy()
    selected, audits = [], []
    for ensemble, part in good.groupby('ensemble', sort=False):
        genotype, protocol = ensemble.split(' | ')
        if channel == 'kv21':
            rel = f'kv21/dataDistances/26-02-11_Kv2.1_{genotype.lower()}_{protocol}AF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv'
            q = pd.read_csv(resolve(rel), usecols=['pdb_file'])
            names = q.pdb_file.map(lambda x: Path(x).name)
            selection = '3 Å + structural/interface/alignment QC'
        else:
            token = {'masked v2':'masked_v2','masked v2 no-IFM':'masked_v2_noIFM'}.get(protocol,protocol)
            rel = f'{channel}/rmsd_convergence_filtering/{PREFIX[channel]}_rmsd_convergence_{genotype.lower()}_{token}/all_models_manifest.csv'
            q = select_manifest_rows(ROOT/rel, 'all_ok_3')
            names = q.pdb_basename
            selection = '3 Å convergence QC'
        if names.duplicated().any():
            raise ValueError(f'Duplicate selected identities in {rel}')
        absent = set(names)-set(part.model_name)
        # Failures remain explicit in the audit; successful missing models are forbidden.
        unexpected = absent-set(data.loc[~data.analysis_eligible,'model_name'])
        if unexpected:
            raise ValueError(f'{ensemble}: {len(unexpected)} QC names cannot join')
        keep = part[part.model_name.isin(names)].copy()
        selected.append(keep)
        audits.append(dict(ensemble=ensemble, selection=selection, source=rel, requested=len(names),
                           matched=len(keep), excluded_failed=len(absent), excluded_by_structural_qc=len(part)-len(keep)))
    pd.DataFrame(audits).to_csv(out/'tables/selection_audit.csv', index=False)
    qc = pd.concat(selected, ignore_index=True)
    return {'all_recorded': good, 'all_recycles': good[good.snapshot.ne('final')],
            'final_only': good[good.snapshot.eq('final')], 'repository_QC': qc,
            'r10_repository_QC': qc[qc.snapshot.eq('r10')]}


def comparisons(channel, ensembles):
    pairs = []
    available = set(ensembles)
    for genotype in GENOTYPES[channel]:
        for protocol in PROTOCOLS[1:]:
            a,b = f'{genotype} | vanilla',f'{genotype} | {protocol}'
            if {a,b} <= available: pairs.append((a,b))
    for protocol in PROTOCOLS:
        for a,b in itertools.combinations(GENOTYPES[channel],2):
            a,b = f'{a} | {protocol}',f'{b} | {protocol}'
            if {a,b} <= available: pairs.append((a,b))
    if channel == 'nav15':
        pairs += [('QQQ | masked','QQQ | masked v2'), ('WT | masked','WT | masked v2'),
                  ('WT | masked v2','WT | masked v2 no-IFM')]
    return pairs


def contrast_statistics(cohorts, channel, out):
    rows = []
    for cohort in ['all_recycles', 'repository_QC', 'final_only', 'r10_repository_QC']:
        d = cohorts[cohort]
        for pair_index,(a,b) in enumerate(comparisons(channel,d.ensemble.unique())):
            da, db = d[d.ensemble.eq(a)], d[d.ensemble.eq(b)]
            ka,kb = set(zip(da.seed,da.af_model,da.snapshot)),set(zip(db.seed,db.af_model,db.snapshot))
            for metric in METRICS:
                ta=da.groupby(['seed','model_number'],as_index=False)[metric].mean()
                tb=db.groupby(['seed','model_number'],as_index=False)[metric].mean()
                sa,sb=ta.groupby('seed')[metric].mean(),tb.groupby('seed')[metric].mean()
                common=sa.index.intersection(sb.index)
                full=set(sa.index)==set(sb.index)
                rng=np.random.default_rng(RANDOM_SEED+pair_index)
                unpaired=(sb.to_numpy()[rng.integers(0,len(sb),(BOOTSTRAPS,len(sb)))].mean(1)
                          -sa.to_numpy()[rng.integers(0,len(sa),(BOOTSTRAPS,len(sa)))].mean(1))
                ci=np.quantile(unpaired,[.025,.975])
                independent_p=stats.ttest_ind(sb,sa,equal_var=False).pvalue
                paired=paired_common_seed_bootstrap(ta,tb,metric,n_bootstrap=BOOTSTRAPS,random_seed=RANDOM_SEED+pair_index)
                paired_p=stats.ttest_rel(sb.loc[common],sa.loc[common]).pvalue
                u=stats.mannwhitneyu(sb,sa,alternative='two-sided').statistic
                rows.append(dict(cohort=cohort,A=a,B=b,metric=metric,N_A=len(da),N_B=len(db),
                    median_A=da[metric].median(),median_B=db[metric].median(),
                    median_difference=db[metric].median()-da[metric].median(),
                    seed_mean_A=sa.mean(),seed_mean_B=sb.mean(),seed_mean_difference=sb.mean()-sa.mean(),
                    CI_low=paired['CI_low_B_minus_A'] if full else ci[0],
                    CI_high=paired['CI_high_B_minus_A'] if full else ci[1],
                    primary_test='paired nominal seed t' if full else 'Welch seed t',
                    p=paired_p if full else independent_p, independent_p=independent_p,
                    independent_CI_low=ci[0],independent_CI_high=ci[1],
                    common_seed_difference=paired['estimate_B_minus_A'],common_seed_p=paired_p,
                    common_seed_CI_low=paired['CI_low_B_minus_A'],common_seed_CI_high=paired['CI_high_B_minus_A'],
                    n_seeds_A=len(sa),n_seeds_B=len(sb),common_seeds=len(common),
                    common_model_snapshots=len(ka&kb),only_A_snapshots=len(ka-kb),only_B_snapshots=len(kb-ka),
                    seed_rank_biserial=2*u/(len(sa)*len(sb))-1,
                    paired_seed_correlation=sa.loc[common].corr(sb.loc[common])))
    result=pd.DataFrame(rows)
    for _, indices in result.groupby('cohort').groups.items():
        for pcol,qcol in [('p','q_BH'),('independent_p','independent_q_BH'),('common_seed_p','common_seed_q_BH')]:
            result.loc[indices,qcol]=multipletests(result.loc[indices,pcol],method='fdr_bh')[1]
    result.to_csv(out/'tables/contrasts.csv',index=False)
    return result


def colors(channel):
    palette={'cav12':CAV12_PALETTE,'kv21':KV21_PALETTE,'nav15':NAV15_PALETTE}[channel]
    result={f'{g} | {p}':palette[f'{g}_{suffix}'] for g in GENOTYPES[channel]
            for p,suffix in [('vanilla','VAN'),('masked','HM')]}
    if channel=='nav15':
        result.update({'WT | masked v2':palette['WT_MASKED_V2'],
                       'WT | masked v2 no-IFM':palette['WT_MASKED_V2_NOIFM'],
                       'QQQ | masked v2':palette['QQQ_MASKED_V2']})
    return result


def order(channel, frame, main=False):
    if main and channel=='nav15': return ['WT | vanilla','QQQ | vanilla','QQQ | masked']
    return [f'{g} | {p}' for p in PROTOCOLS for g in GENOTYPES[channel] if f'{g} | {p}' in set(frame.ensemble)]


def finish(fig, out, name):
    fig.tight_layout()
    for ext in ['png','pdf']:
        fig.savefig(out/'figures'/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    return fig


def violin(ax,d,metric,labels,palette):
    sns.violinplot(data=d,x='ensemble',y=metric,order=labels,hue='ensemble',
                   palette=palette,legend=False,inner='quartile',cut=0,linewidth=.65,
                   density_norm='width',ax=ax)
    tick_labels = [x.replace(' | ','\n').replace('masked v2 no-IFM','masked v2\nno-IFM') for x in labels]
    ax.set_xticks(range(len(labels)),tick_labels,rotation=0,fontsize=9 if len(labels)>6 else 10)
    ax.set_xlabel(''); ax.set_ylabel(LABELS[metric])
    ax.grid(axis='x',visible=False); ax.grid(axis='y',color='#E8EEF5',linewidth=.45,linestyle='--')
    sns.despine(ax=ax)


def plot_distributions(cohorts,channel,out):
    apply_kv21_style(); palette=colors(channel)
    fig,axs=plt.subplots(1,2,figsize=(12.4,5.3))
    for ax,cohort,title in zip(axs,['all_recycles','repository_QC'],['A. All recycle snapshots','B. Repository QC subset']):
        d=cohorts[cohort];labels=order(channel,d,main=True)
        violin(ax,d,'clashscore',labels,palette);ax.set_title(title,fontweight='bold')
    fig.suptitle(format_channel_title(TITLES[channel]+' | global steric clashes'),y=1.03)
    return finish(fig,out,'clashscore_distributions')


def plot_effects(contrasts,channel,out):
    d=contrasts[(contrasts.cohort=='repository_QC')&contrasts.metric.eq('clashscore')]
    if channel=='nav15':
        d=d[((d.A=='WT | vanilla')&(d.B=='QQQ | vanilla'))|((d.A=='QQQ | vanilla')&(d.B=='QQQ | masked'))]
    fig,ax=plt.subplots(figsize=(9.8,max(3.2,.43*len(d)+1.6)))
    for i,(_,r) in enumerate(d.iterrows()):
        ax.plot([r.CI_low,r.CI_high],[i,i],color=colors(channel)[r.B],lw=1.5)
        ax.scatter(r.seed_mean_difference,i,s=35,color=colors(channel)[r.B],edgecolor='#333333',lw=.6)
        ax.text(1.01,i,f'q={r.q_BH:.2g}',transform=ax.get_yaxis_transform(),va='center',fontsize=9)
    ax.set_yticks(range(len(d)),[f'{r.B} − {r.A}' for r in d.itertuples()]);ax.invert_yaxis()
    ax.axvline(0,color='#555555',lw=.8,ls='--');ax.set_xlabel('Seed-balanced mean clashscore difference (95% bootstrap CI)')
    ax.set_title('Repository QC | condition contrasts',fontweight='bold');sns.despine(ax=ax)
    return finish(fig,out,'clashscore_effects')


def plot_diagnostics(cohorts,channel,out):
    d=cohorts['repository_QC'];labels=order(channel,d);palette=colors(channel)
    fig,axs=plt.subplots(1,2,figsize=(13.2,5.2))
    for ax,metric,title in zip(axs,['n_bad_clashes','max_overlap'],['A. Bad-clash count','B. Maximum overlap']):
        violin(ax,d,metric,labels,palette);ax.set_title(title,fontweight='bold')
    return finish(fig,out,'complementary_diagnostics')


def plot_nav_controls(cohorts,out):
    d=cohorts['repository_QC'];labels=order('nav15',d)
    fig,ax=plt.subplots(figsize=(11.5,5.2))
    violin(ax,d,'clashscore',labels,colors('nav15'))
    ax.set_title(format_channel_title('Nav1.5 | all mask designs'),fontweight='bold')
    return finish(fig,out,'supplemental_mask_designs')


def stratification(cohorts,channel,out):
    d=cohorts['all_recorded'];rows=[]
    for factor in ['snapshot','af_model','seed','rank']:
        for (ensemble,level),part in d.groupby(['ensemble',factor]):
            rows.append(dict(factor=factor,ensemble=ensemble,level=level,N=len(part),median=part.clashscore.median(),Q1=part.clashscore.quantile(.25),Q3=part.clashscore.quantile(.75)))
    table=pd.DataFrame(rows);table.to_csv(out/'tables/stratified_clashscore.csv',index=False)
    fig,axs=plt.subplots(2,2,figsize=(12.4,8.4));palette=colors(channel)
    for ax,factor in zip(axs.flat,['snapshot','af_model','seed','rank']):
        for ensemble in order(channel,d):
            t=table[(table.factor==factor)&table.ensemble.eq(ensemble)].copy()
            if t.empty:continue
            if factor=='snapshot': t['x']=t.level.map(dict(zip(SNAPSHOTS,range(12))))
            else:t['x']=pd.to_numeric(t.level)
            t=t.sort_values('x')
            if factor in ['rank','seed']:
                ax.scatter(t.x,t['median'],s=6,alpha=.5,color=palette[ensemble],rasterized=True)
            else:ax.plot(t.x,t['median'],marker='o',ms=3,lw=.9,color=palette[ensemble],label=ensemble)
        ax.set_xlabel({'af_model':'AF model','rank':'Rank (unavailable for L403A vanilla)' if channel=='kv21' else 'Rank','seed':'Recorded seed','snapshot':'Snapshot'}[factor]);ax.set_ylabel('Median clashscore')
        if factor=='snapshot':ax.set_xticks(range(12),[str(i) for i in range(11)]+['final'],rotation=45)
        if factor=='af_model':ax.set_xticks(range(1,6))
        sns.despine(ax=ax)
    axs[0,1].legend(fontsize=8,loc='best');fig.suptitle('Sampling diagnostics | all recorded models',y=1.01)
    return finish(fig,out,'sampling_diagnostics')


def adjusted_contrasts(cohorts,channel,out):
    """Snapshot/AF-stratum adjustment with seed-clustered uncertainty."""
    rows=[];d=cohorts['repository_QC']
    for a,b in comparisons(channel,d.ensemble.unique()):
        w=d[d.ensemble.isin([a,b])].copy();w['condition_B']=w.ensemble.eq(b).astype(int)
        # Shared numeric seeds are treated as clusters, retaining nominal pairing.
        fit=smf.ols('clashscore ~ condition_B + C(af_model) + C(snapshot)',w).fit(cov_type='cluster',cov_kwds={'groups':w.seed})
        ci=fit.conf_int().loc['condition_B']
        rows.append(dict(A=a,B=b,adjusted_difference=fit.params['condition_B'],CI_low=ci.iloc[0],CI_high=ci.iloc[1],p=fit.pvalues['condition_B']))
    result=pd.DataFrame(rows);result['q_BH']=multipletests(result.p,method='fdr_bh')[1]
    result.to_csv(out/'tables/snapshot_AF_adjusted_contrasts.csv',index=False)
    return result


def af_model_sensitivity(cohorts, channel, out):
    rows=[]; d=cohorts['repository_QC']
    for a,b in comparisons(channel,d.ensemble.unique()):
        for model in range(1,6):
            for mode in ['only_AF_model','leave_AF_model_out']:
                mask=d.af_model.eq(model) if mode=='only_AF_model' else d.af_model.ne(model)
                w=d[mask & d.ensemble.isin([a,b])]
                values=w.groupby(['ensemble','seed','af_model']).clashscore.mean().groupby(['ensemble','seed']).mean()
                rows.append(dict(A=a,B=b,mode=mode,AF_model=model,
                                 seed_mean_difference=values.loc[b].mean()-values.loc[a].mean()))
    result=pd.DataFrame(rows);result.to_csv(out/'tables/AF_model_sensitivity.csv',index=False)
    return result


def kv21_conformation(cohorts,out):
    """Exact ensemble+basename join, checked against seed/model/snapshot metadata."""
    path='kv21/dataExtra/kv21_l403a_conformational_metrics_chain_resolved.csv'
    cols=['source_type','condition','protocol','source_path','structure_id','seed','model_number','recycle_label','canonical_subunit','whole_s6_rotation_vs_8SD3_deg','F412_ca_displacement_vs_8SD3_A']
    chain=pd.read_csv(resolve(path),usecols=cols)
    exp=chain[chain.source_type.eq('experimental')].copy()
    refs=exp.pivot(index='canonical_subunit',columns='structure_id',values='whole_s6_rotation_vs_8SD3_deg')
    ref_delta=(refs['8SDA']-refs['8SD3']+180)%360-180
    pred=chain[chain.source_type.eq('prediction')].copy()
    pred['ensemble']=pred.condition.str.upper()+' | '+pred.protocol
    pred['model_name']=pred.source_path.map(lambda p:Path(p).name)
    keys=['ensemble','model_name']
    if pred.duplicated(keys+['canonical_subunit']).any():raise ValueError('Duplicate conformational chain identities')
    rotations=pred.pivot(index=keys,columns='canonical_subunit',values='whole_s6_rotation_vs_8SD3_deg').reindex(columns=refs.index)
    rotation_delta=(rotations-refs['8SD3']+180)%360-180
    projection=(rotation_delta*ref_delta).sum(axis=1,min_count=4)/(ref_delta**2).sum()
    metrics=pred.groupby(keys).agg(F412_displacement_A=('F412_ca_displacement_vs_8SD3_A','median'),chains=('canonical_subunit','nunique'),seed_conf=('seed','first'),af_model_conf=('model_number','first'),snapshot_conf=('recycle_label','first')).reset_index()
    metrics=metrics.merge(projection.rename('S6_shift_projection').reset_index(),on=keys,validate='one_to_one')
    if metrics.chains.ne(4).any():raise ValueError('Incomplete tetramers in conformation table')
    all_models=cohorts['all_recorded'].merge(metrics,on=keys,how='outer',validate='one_to_one',indicator=True)
    join_audit=all_models.groupby(['ensemble','_merge'],observed=True).size().rename('N').reset_index()
    join_audit.to_csv(out/'tables/conformation_join_audit.csv',index=False)
    unmatched=all_models[all_models._merge.ne('both')]
    unmatched.to_csv(out/'tables/conformation_unmatched.csv',index=False)
    if len(unmatched):raise ValueError('Incomplete conformation join; inspect conformation_unmatched.csv')
    for left,right in [('seed','seed_conf'),('af_model','af_model_conf'),('snapshot','snapshot_conf')]:
        if all_models[left].ne(all_models[right]).any():raise ValueError(f'Conformation metadata mismatch: {left}')
    joined=cohorts['repository_QC'].merge(metrics,on=keys,validate='one_to_one')
    refs.assign(experimental_delta=ref_delta).to_csv(out/'tables/S6_rotation_reference.csv')
    joined[keys+['seed','af_model','snapshot','clashscore','F412_displacement_A','S6_shift_projection']].to_csv(out/'tables/conformation_joined_models.csv',index=False)
    results=[]
    for cohort,frame in [('repository_QC',joined),('r10_repository_QC',joined[joined.snapshot.eq('r10')]),('all_recycles',all_models[all_models.snapshot.ne('final')]),('final_only',all_models[all_models.snapshot.eq('final')])]:
        for ensemble,part in frame.groupby('ensemble'):
            for metric in ['S6_shift_projection','F412_displacement_A']:
                w=part.dropna(subset=[metric,'clashscore']).copy()
                if len(w)!=len(part):print(f'{ensemble}: {len(part)-len(w)} nonfinite conformational rows explicitly omitted')
                # Repeat observations contribute to the fit, but uncertainty clusters by seed.
                fit=smf.ols(f'clashscore ~ {metric} + C(af_model) + C(snapshot)',w).fit(cov_type='cluster',cov_kwds={'groups':w.seed})
                ci=fit.conf_int().loc[metric]
                # Independent seed summaries provide a robust complementary association.
                seed=w.groupby(['seed','af_model'])[[metric,'clashscore']].median().groupby('seed').mean()
                rho,p=stats.spearmanr(seed[metric],seed.clashscore)
                rng=np.random.default_rng(RANDOM_SEED);boot=[]
                for _ in range(1000):
                    v=seed.iloc[rng.integers(0,len(seed),len(seed))]
                    boot.append(stats.spearmanr(v[metric],v.clashscore).statistic)
                low,high=np.quantile(boot,[.025,.975])
                results.append(dict(cohort=cohort,ensemble=ensemble,metric=metric,N=len(w),missing_metric=len(part)-len(w),seeds=len(seed),
                    model_spearman_rho=stats.spearmanr(w[metric],w.clashscore).statistic,
                    seed_spearman_rho=rho,rho_CI_low=low,rho_CI_high=high,spearman_p=p,
                    adjusted_slope=fit.params[metric],slope_CI_low=ci.iloc[0],slope_CI_high=ci.iloc[1],slope_p=fit.pvalues[metric],
                    metric_Q1=w[metric].quantile(.25),metric_Q3=w[metric].quantile(.75),
                    fitted_clashscore_change_per_IQR=fit.params[metric]*(w[metric].quantile(.75)-w[metric].quantile(.25))))
    result=pd.DataFrame(results)
    for _,idx in result.groupby('cohort').groups.items():
        for p,q in [('slope_p','slope_q_BH'),('spearman_p','spearman_q_BH')]:result.loc[idx,q]=multipletests(result.loc[idx,p],method='fdr_bh')[1]
    result.to_csv(out/'tables/conformation_associations.csv',index=False)
    # Direct slope-difference tests; comparing two separate p-values is insufficient.
    interactions=[]
    for metric in ['S6_shift_projection','F412_displacement_A']:
        for a,b in comparisons('kv21',joined.ensemble.unique()):
            w=joined[joined.ensemble.isin([a,b])].copy();w['condition_B']=w.ensemble.eq(b).astype(int)
            fit=smf.ols(f'clashscore ~ {metric} * condition_B + condition_B * (C(af_model) + C(snapshot))',w).fit(cov_type='cluster',cov_kwds={'groups':w.seed})
            term=f'{metric}:condition_B';ci=fit.conf_int().loc[term]
            interactions.append(dict(metric=metric,A=a,B=b,slope_difference=fit.params[term],CI_low=ci.iloc[0],CI_high=ci.iloc[1],p=fit.pvalues[term]))
    interactions=pd.DataFrame(interactions);interactions['q_BH']=multipletests(interactions.p,method='fdr_bh')[1]
    interactions.to_csv(out/'tables/conformation_slope_interactions.csv',index=False)
    return joined,result,interactions


def plot_conformation(joined,results,out):
    fig,axs=plt.subplots(2,3,figsize=(13.2,8.2),sharex=True,sharey=True)
    palette=colors('kv21')
    for ax,ensemble in zip(axs.flat,order('kv21',joined)):
        d=joined[joined.ensemble.eq(ensemble)]
        ax.scatter(d.S6_shift_projection,d.clashscore,s=7,alpha=.2,color=palette[ensemble],edgecolors='none',rasterized=True)
        # Binned medians are descriptive; formal adjusted slopes are in the table.
        bins=pd.qcut(d.S6_shift_projection,8,duplicates='drop')
        centers=d.groupby(bins,observed=True)[['S6_shift_projection','clashscore']].median()
        ax.plot(centers.S6_shift_projection,centers.clashscore,color='#333333',lw=1.0,marker='o',ms=3)
        r=results[(results.cohort=='repository_QC')&results.ensemble.eq(ensemble)&results.metric.eq('S6_shift_projection')].iloc[0]
        ax.set_title(ensemble,fontweight='bold');ax.text(.03,.97,f'Seed ρ = {r.seed_spearman_rho:.2f}\n95% CI [{r.rho_CI_low:.2f}, {r.rho_CI_high:.2f}]',transform=ax.transAxes,va='top',fontsize=9)
        ax.axvline(0,color='#888888',lw=.6,ls='--');ax.axvline(1,color='#888888',lw=.6,ls=':');sns.despine(ax=ax)
    for ax in axs[-1]:ax.set_xlabel('S6 rotation projection\n0 = 8SD3; 1 = 8SDA')
    for ax in axs[:,0]:ax.set_ylabel('Clashscore')
    return finish(fig,out,'S6_shift_clashscore')


def plot_conformation_adjusted(joined,results,out):
    fig,axs=plt.subplots(2,3,figsize=(13.2,8.2),sharex=True,sharey=True)
    for ax,ensemble in zip(axs.flat,order('kv21',joined)):
        d=joined[joined.ensemble.eq(ensemble)].copy()
        x=smf.ols('S6_shift_projection ~ C(af_model) + C(snapshot)',d).fit().resid
        y=smf.ols('clashscore ~ C(af_model) + C(snapshot)',d).fit().resid
        r=results[results.cohort.eq('repository_QC')&results.ensemble.eq(ensemble)&results.metric.eq('S6_shift_projection')].iloc[0]
        ax.scatter(x,y,s=7,alpha=.25,color=colors('kv21')[ensemble],edgecolors='none',rasterized=True)
        xx=np.array([x.min(),x.max()]);ax.plot(xx,r.adjusted_slope*xx,color='#333333',lw=1)
        ax.axhline(0,color='#888888',lw=.5,ls='--');ax.axvline(0,color='#888888',lw=.5,ls='--')
        ax.set_title(ensemble,fontweight='bold')
        ax.text(.03,.97,f'Slope = {r.adjusted_slope:.1f}\n95% CI [{r.slope_CI_low:.1f}, {r.slope_CI_high:.1f}]',transform=ax.transAxes,va='top',fontsize=9)
        sns.despine(ax=ax)
    for ax in axs[-1]:ax.set_xlabel('Residual S6 rotation projection')
    for ax in axs[:,0]:ax.set_ylabel('Residual clashscore')
    fig.suptitle('Within AF-model / recycle strata | partial regression',y=1.01)
    return finish(fig,out,'S6_shift_adjusted_clashscore')


def interpretation(summary,contrasts,channel):
    """Executed notebook text stays tied to recomputed results."""
    lines=['### Results on the repository QC subset', '',
           'Model medians describe retained snapshots. Differences and confidence intervals below use equal AF-model weights within seed and seed-level inference.']
    d=contrasts[contrasts.cohort.eq('repository_QC')&contrasts.metric.eq('clashscore')]
    for r in d.itertuples():
        lines.append(f'- **{r.B} versus {r.A}:** model medians {r.median_B:.2f} versus {r.median_A:.2f} (Δ {r.median_difference:+.2f}); seed-balanced mean Δ {r.seed_mean_difference:+.2f}, 95% CI [{r.CI_low:.2f}, {r.CI_high:.2f}]; seed rank-biserial effect {r.seed_rank_biserial:+.2f}; BH q={r.q_BH:.3g}.')
    return '\n'.join(lines)


def focused_interpretation(summary,contrasts,channel,out):
    s=summary[summary.cohort.eq('repository_QC')].set_index(['ensemble','metric'])
    c=contrasts[contrasts.cohort.eq('repository_QC')]
    def median(ensemble,metric='clashscore'):return s.loc[(ensemble,metric),'median']
    def difference(a,b,metric='clashscore'):
        r=c[c.A.eq(a)&c.B.eq(b)&c.metric.eq(metric)].iloc[0]
        return f'{r.seed_mean_difference:+.3f} (95% CI {r.CI_low:+.3f} to {r.CI_high:+.3f})'
    if channel=='cav12':
        lines=['**G406R focus.** The global signal is small relative to within-ensemble dispersion.']
        for protocol in ['vanilla','masked']:
            wt,g402,g406=[f'{g} | {protocol}' for g in ['WT','G402S','G406R']]
            lines.append(f'- {protocol.capitalize()}: WT/G402S/G406R clashscore medians are {median(wt):.3f}/{median(g402):.3f}/{median(g406):.3f}; G406R−WT seed-balanced difference is {difference(wt,g406)}. G406R−G402S is {difference(g402,g406)}. Bad-clash medians are {median(wt,"n_bad_clashes"):.0f}/{median(g402,"n_bad_clashes"):.0f}/{median(g406,"n_bad_clashes"):.0f}; maximum-overlap medians are {median(wt,"max_overlap"):.3f}/{median(g402,"max_overlap"):.3f}/{median(g406,"max_overlap"):.3f} Å.')
        lines.append('The current data support modest protocol-associated global shifts, not a distinctive large G406R global clash phenotype. They cannot locate a clash at R406 or exclude a severe local contact diluted in a global score. The existing short local distances still justify parsing atom-level cluster logs: identify R406-involving atom pairs, contact frequencies and overlap magnitudes, with WT/G402S controls. Keep this global comparison as QC; local attribution remains untested.')
    elif channel=='nav15':
        lines=[f'**Primary sequence versus mask comparison.** WT vanilla and QQQ vanilla medians are {median("WT | vanilla"):.2f} and {median("QQQ | vanilla"):.2f}; seed-balanced QQQ−WT is {difference("WT | vanilla","QQQ | vanilla")}. QQQ masking raises the median to {median("QQQ | masked"):.2f}, with a seed-balanced change of {difference("QQQ | vanilla","QQQ | masked")}.',
               f'QQQ masked v2 versus original masked changes the mean by {difference("QQQ | masked","QQQ | masked v2")}. The WT no-IFM control versus WT masked v2 changes it by {difference("WT | masked v2","WT | masked v2 no-IFM")}.',
               'The principal global difference follows mask design, not the QQQ substitution under vanilla. Maximum-overlap medians remain close across the ensembles; there is no unique catastrophic global-quality outlier among the retained conditions. This is a relative QC statement, not certification of atomistic quality in these unrelaxed models. The mask-related clashscore cost is useful alongside the pore-sampling figures; most alternative-mask results belong in supplementary QC.']
    else:
        a=pd.read_csv(out/'tables/conformation_associations.csv')
        lines=['**Global quality and S6.** Masking changes the seed-balanced mean clashscore by '+ '; '.join(f'{g}: {difference(g+" | vanilla",g+" | masked")}' for g in GENOTYPES[channel])+'.',
               f'The original conformational table joins {pd.read_csv(out/"tables/conformation_join_audit.csv").N.sum():,} available models exactly, including seed/AF-model/snapshot checks; {int(s.xs("clashscore",level="metric").N.sum()):,} enter established structural QC. These are successful identity matches, not proof of coordinate checksums across cluster runs.',
               'Directional S6 projection associations depend on aggregation and selection. For each ensemble below, compare the seed-level rank association with the AF-model/recycle-adjusted model-level slope; they estimate different relationships:']
        for r in a[a.cohort.eq('repository_QC')&a.metric.eq('S6_shift_projection')].itertuples():
            lines.append(f'- {r.ensemble}: seed ρ={r.seed_spearman_rho:+.2f} (95% CI {r.rho_CI_low:+.2f} to {r.rho_CI_high:+.2f}); adjusted slope {r.adjusted_slope:+.2f} clashscore per projection unit, CI [{r.slope_CI_low:.2f}, {r.slope_CI_high:.2f}], equivalent to {r.fitted_clashscore_change_per_IQR:+.2f} over that ensemble’s coordinate IQR.')
        lines.append('**r10 within the same QC population:**')
        for r in a[a.cohort.eq('r10_repository_QC')&a.metric.eq('S6_shift_projection')].itertuples():
            lines.append(f'- {r.ensemble}: seed ρ={r.seed_spearman_rho:+.2f} (95% CI {r.rho_CI_low:+.2f} to {r.rho_CI_high:+.2f}); adjusted slope {r.adjusted_slope:+.2f}, CI [{r.slope_CI_low:.2f}, {r.slope_CI_high:.2f}].')
        lines.append('Unsigned F412-position displacement has positive adjusted slopes in the pooled QC analysis, but this is not universal across the r10 sensitivities and does not establish motion toward 8SDA. A universal steric ceiling on the experimentally directed S6 shift is not established: marginal and conditional associations differ, early-recycle behavior matters, and the analysis observes sampled survivors rather than an energetic barrier. The S6 association is exploratory and potentially figure-worthy only with the selection/adjustment sensitivity shown. Atom-level S6/interface clash localization followed by controlled relaxation of matched models would be more informative about a steric mechanism; correlation alone is not causation.')
    af=pd.read_csv(out/'tables/AF_model_sensitivity.csv')
    lines.append('**AF-model dependence.** The following ranges are effect magnitudes, not new hypothesis tests:')
    mixed_signs=False
    for genotype in GENOTYPES[channel]:
        subset=af[af.A.eq(genotype+' | vanilla')&af.B.eq(genotype+' | masked')]
        only=subset[subset['mode'].eq('only_AF_model')].seed_mean_difference
        leave=subset[subset['mode'].eq('leave_AF_model_out')].seed_mean_difference
        mixed_signs |= only.min()<0<only.max()
        lines.append(f'- {genotype} masked−vanilla: individual AF-model mean differences range from {only.min():+.2f} to {only.max():+.2f}; leaving out one AF model at a time gives {leave.min():+.2f} to {leave.max():+.2f}.')
    lines.append(('Mixed signs across AF models preclude a uniform per-model masking effect. ' if mixed_signs else 'Original masking raises the mean within every AF-model stratum. ')
                 +'The stratified tables and final-r10 sensitivities should accompany claims based on pooled means.')
    return '\n\n'.join(lines)
