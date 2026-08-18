"""All-distance equal-seed-depth analysis for QC-passing Kv2.1 L403A models."""
from __future__ import annotations

from pathlib import Path
import json
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, levene, mannwhitneyu, spearmanr, wasserstein_distance
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from shared.distribution_statistics import parse_trajectory_metadata, exact_common_distance_columns, stable_seed
from shared.plotting import apply_kv21_style, KV21_PALETTE

DATA=ROOT/'kv21'/'dataDistances'
OUT=ROOT/'kv21'/'dataExtra'/'conformation_analysis'/'all_distance_sampling'
FIG=OUT/'figures'; TAB=OUT/'tables'; FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
PATHS={
 'vanilla':DATA/'26-02-11_Kv2.1_l403a_vanillaAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv',
 'masked':DATA/'26-02-11_Kv2.1_l403a_maskedAF2_distances_all_ok_rmsd_3A_structural_interface_alignment_qc.csv'}
DEPTHS=[4,8,12,16,20,'Full QC']; FOCAL_DEPTH=20; EPS=1e-9

def distance_chain_pair(column):
    """Return chain labels encoded by a Kv2.1 distance-table column."""
    if column.startswith('CA_'):
        match=re.match(r'^CA_CA_([A-Za-z0-9])_.*?-([A-Za-z0-9])_',column)
        return match.groups() if match else None
    match=re.match(r'^shortest_[A-Z]{3}\d+([A-Za-z0-9])-[A-Z]{3}\d+([A-Za-z0-9])$',column)
    return match.groups() if match else None

def is_intrachain_distance(column):
    pair=distance_chain_pair(column)
    return pair is not None and pair[0]==pair[1]

def load():
    frames={p:parse_trajectory_metadata(pd.read_csv(path,low_memory=False)) for p,path in PATHS.items()}
    shared=exact_common_distance_columns(frames['vanilla'],frames['masked'])
    cols=[column for column in shared if is_intrachain_distance(column)]
    if len(cols)!=546:
        raise ValueError(f'Expected 546 chain-label-safe intrachain distances; found {len(cols)}')
    for p,d in frames.items():
        d['_recycle_order']=pd.to_numeric(d.recycle_number,errors='coerce').fillna(999)
        d.sort_values(['seed','model_number','_recycle_order','pdb_file'],inplace=True)
        d['sampling_index']=np.arange(1,len(d)+1)
    return frames,cols

def subset(frame,depth):
    """Retain every QC-passing row belonging to the first N ordered seeds."""
    if depth=='Full QC':
        return frame.copy()
    retained_seeds=frame['seed'].drop_duplicates().iloc[:int(depth)]
    return frame[frame['seed'].isin(retained_seeds)].copy()

def trajectory_representatives(frame):
    """Use one structure per retained trajectory: latest final-QC recycle."""
    return (frame.sort_values(['seed','model_number','_recycle_order','pdb_file'])
            .groupby(['seed','model_number'],sort=False,as_index=False).tail(1).copy())

def seed_iqr(frame,cols):
    g=frame.groupby('seed')[cols]
    return g.quantile(.75)-g.quantile(.25)

def nominal_trajectory_retention_audit(frames,n_seeds=20):
    """Audit final-QC survival in the nominal first 20 seeds × 5 models."""
    rows=[]
    for protocol,frame in frames.items():
        models=sorted(pd.to_numeric(frame['model_number'],errors='raise').astype(int).unique())
        if models != [1,2,3,4,5]:
            raise ValueError(f'Expected model numbers 1–5 for {protocol}; observed {models}')
        seeds=frame['seed'].drop_duplicates().iloc[:n_seeds].tolist()
        for seed_rank,seed in enumerate(seeds,start=1):
            for model_rank,model in enumerate(models,start=1):
                kept=frame[(frame['seed']==seed)&(frame['model_number']==model)]
                recycles=sorted(pd.to_numeric(kept['recycle_number'],errors='coerce').dropna().astype(int).unique())
                rows.append({'protocol':protocol,'nominal_trajectory_order':(seed_rank-1)*len(models)+model_rank,
                    'seed_rank':seed_rank,'seed':int(seed),'model_number':int(model),
                    'final_qc_retained':not kept.empty,'final_qc_rows':len(kept),
                    'retained_recycles':','.join(map(str,recycles))})
    audit=pd.DataFrame(rows)
    summary=(audit.groupby('protocol',sort=False).agg(
        nominal_trajectories=('final_qc_retained','size'),
        retained_trajectories=('final_qc_retained','sum'),
        first_seed=('seed','min'),last_seed=('seed','max'),
        retained_model_recycle_rows=('final_qc_rows','sum')).reset_index())
    summary['excluded_trajectories']=summary.nominal_trajectories-summary.retained_trajectories
    summary['trajectory_retention_fraction']=summary.retained_trajectories/summary.nominal_trajectories
    return audit,summary

def analyze_depth(frames,cols,depth):
    a=trajectory_representatives(subset(frames['vanilla'],depth)); b=trajectory_representatives(subset(frames['masked'],depth))
    av=a[cols].apply(pd.to_numeric,errors='coerce'); bv=b[cols].apply(pd.to_numeric,errors='coerce')
    aiqr=av.quantile(.75)-av.quantile(.25); biqr=bv.quantile(.75)-bv.quantile(.25)
    amed=av.median(); bmed=bv.median(); pooled=pd.concat([av,bv],ignore_index=True); piqr=pooled.quantile(.75)-pooled.quantile(.25)
    asi=seed_iqr(a,cols); bsi=seed_iqr(b,cols)
    # Each seed contributes one within-seed breadth estimate per distance.
    stat,p=mannwhitneyu(asi.to_numpy(float),bsi.to_numpy(float),axis=0,alternative='two-sided',nan_policy='omit')
    q=np.full(len(p),np.nan); finite=np.isfinite(p)
    q[finite]=multipletests(np.asarray(p)[finite],method='fdr_bh')[1]
    # Brown-Forsythe is a trajectory-level sensitivity test for dispersion;
    # seed-level inference above remains primary because trajectories cluster.
    bf_stat=[]; bf_p=[]
    for c in cols:
        x=av[c].dropna().to_numpy(); y=bv[c].dropna().to_numpy()
        s,pv=levene(x,y,center='median')
        bf_stat.append(s); bf_p.append(pv)
    bf_p=np.asarray(bf_p); bf_q=np.full(len(bf_p),np.nan); finite_bf=np.isfinite(bf_p)
    bf_q[finite_bf]=multipletests(bf_p[finite_bf],method='fdr_bh')[1]
    w1=np.array([wasserstein_distance(av[c].dropna(),bv[c].dropna()) for c in cols])
    out=pd.DataFrame({'depth':depth,'distance':cols,'distance_type':['CA' if c.startswith('CA_') else 'shortest_heavy' for c in cols],
      'vanilla_median_A':amed.values,'masked_median_A':bmed.values,'delta_median_A':(bmed-amed).values,
      'vanilla_global_IQR_A':aiqr.values,'masked_global_IQR_A':biqr.values,
      'global_IQR_ratio':((biqr+EPS)/(aiqr+EPS)).values,'global_log2_IQR_ratio':np.log2((biqr+EPS)/(aiqr+EPS)).values,
      'vanilla_median_seed_IQR_A':asi.median().values,'masked_median_seed_IQR_A':bsi.median().values,
      'seed_IQR_ratio':((bsi.median()+EPS)/(asi.median()+EPS)).values,
      'seed_log2_IQR_ratio':np.log2((bsi.median()+EPS)/(asi.median()+EPS)).values,
      'seed_breadth_U':stat,'seed_breadth_p':p,'seed_breadth_q':q,
      'brown_forsythe_statistic':bf_stat,'brown_forsythe_p':bf_p,'brown_forsythe_q':bf_q,
      'trajectory_median_W1_A':w1,'W1_normalized_pooled_IQR':w1/(piqr.to_numpy(float)+EPS),
      'vanilla_rows':len(a),'masked_rows':len(b),'vanilla_seeds':a.seed.nunique(),'masked_seeds':b.seed.nunique(),
      'vanilla_trajectories':len(a),'masked_trajectories':len(b)})
    return out

def summarize(results):
    rows=[]
    for depth,g in results.groupby('depth',sort=False):
        rows.append({'depth':depth,'distances':len(g),'median_global_IQR_ratio':g.global_IQR_ratio.median(),
          'fraction_global_broader_masked':(g.global_log2_IQR_ratio>0).mean(),
          'median_seed_IQR_ratio':g.seed_IQR_ratio.median(),'fraction_seed_broader_masked':(g.seed_log2_IQR_ratio>0).mean(),
          'fraction_seed_breadth_q_lt_0.05':(g.seed_breadth_q<.05).mean(),
          'fraction_significant_broader_masked':((g.seed_breadth_q<.05)&(g.seed_log2_IQR_ratio>0)).mean(),
          'fraction_significant_narrower_masked':((g.seed_breadth_q<.05)&(g.seed_log2_IQR_ratio<0)).mean(),
          'median_trajectory_W1_A':g.trajectory_median_W1_A.median(),'median_normalized_W1':g.W1_normalized_pooled_IQR.median(),
          'vanilla_rows':g.vanilla_rows.iloc[0],'masked_rows':g.masked_rows.iloc[0],
          'vanilla_seeds':g.vanilla_seeds.iloc[0],'masked_seeds':g.masked_seeds.iloc[0]})
    return pd.DataFrame(rows)

def stability(results):
    full=results[results.depth.astype(str)=='Full QC'].set_index('distance')
    rows=[]
    for depth in [4,8,12,16,20]:
        e=results[results.depth.astype(str)==str(depth)].set_index('distance').loc[full.index]
        for metric in ['global_log2_IQR_ratio','seed_log2_IQR_ratio','trajectory_median_W1_A']:
            rho,p=spearmanr(e[metric],full[metric],nan_policy='omit')
            rows.append({'depth':depth,'metric':metric,'spearman_vs_full':rho,'p':p,
              'same_direction_fraction':(np.sign(e[metric])==np.sign(full[metric])).mean() if 'ratio' in metric else np.nan})
    return pd.DataFrame(rows)

def overall_seed_breadth(frames,cols):
    """One normalized all-distance breadth value per seed in the focal cohort."""
    full_reps={p:trajectory_representatives(f) for p,f in frames.items()}
    pooled=pd.concat([full_reps['vanilla'][cols],full_reps['masked'][cols]],ignore_index=True)
    scale=(pooled.quantile(.75)-pooled.quantile(.25)).replace(0,np.nan)
    records=[]
    for protocol in ['vanilla','masked']:
        reps=trajectory_representatives(subset(frames[protocol],FOCAL_DEPTH))
        spread=seed_iqr(reps,cols).div(scale,axis=1).median(axis=1,skipna=True)
        records.extend({'protocol':protocol,'seed':int(seed),'median_normalized_seed_IQR':value}
                       for seed,value in spread.items())
    values=pd.DataFrame(records)
    vanilla=values.loc[values.protocol=='vanilla','median_normalized_seed_IQR'].dropna()
    masked=values.loc[values.protocol=='masked','median_normalized_seed_IQR'].dropna()
    u,p=mannwhitneyu(masked,vanilla,alternative='two-sided')
    rng=np.random.default_rng(403)
    boot=np.array([np.median(rng.choice(masked,len(masked),replace=True))-
                   np.median(rng.choice(vanilla,len(vanilla),replace=True)) for _ in range(10000)])
    summary=pd.DataFrame([{'vanilla_seeds':len(vanilla),'masked_seeds':len(masked),
        'vanilla_median_normalized_seed_IQR':vanilla.median(),
        'masked_median_normalized_seed_IQR':masked.median(),
        'masked_over_vanilla_median_ratio':(masked.median()+EPS)/(vanilla.median()+EPS),
        'median_difference_masked_minus_vanilla':masked.median()-vanilla.median(),
        'bootstrap_95CI_difference_low':np.quantile(boot,.025),
        'bootstrap_95CI_difference_high':np.quantile(boot,.975),
        'mannwhitney_U_masked_vs_vanilla':u,'mannwhitney_p_two_sided':p,
        'rank_biserial_masked_greater':2*u/(len(masked)*len(vanilla))-1}])
    return values,summary

def equal_retained_count_rarefaction(frames,cols,iterations=2000):
    """Compare all masked survivors with repeated equal-sized vanilla subsets."""
    reps={p:trajectory_representatives(subset(f,FOCAL_DEPTH)) for p,f in frames.items()}
    vanilla=reps['vanilla'][cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    masked=reps['masked'][cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not (np.isfinite(vanilla).all() and np.isfinite(masked).all()):
        raise ValueError('Rarefaction requires complete finite intrachain distances')
    target=min(len(vanilla),len(masked)); rng=np.random.default_rng(100403)
    masked_q=np.quantile(masked,[.25,.75],axis=0); masked_iqr=masked_q[1]-masked_q[0]
    rows=[]
    for iteration in range(iterations):
        sample=vanilla[rng.choice(len(vanilla),target,replace=False)]
        vanilla_q=np.quantile(sample,[.25,.75],axis=0); vanilla_iqr=vanilla_q[1]-vanilla_q[0]
        ratios=(masked_iqr+EPS)/(vanilla_iqr+EPS)
        rows.append({'iteration':iteration+1,'retained_trajectories_per_protocol':target,
            'median_IQR_ratio_masked_over_vanilla':np.nanmedian(ratios),
            'fraction_distances_broader_masked':np.nanmean(ratios>1)})
    draws=pd.DataFrame(rows)
    summary=pd.DataFrame([{'iterations':iterations,'retained_trajectories_per_protocol':target,
        'median_IQR_ratio':draws.median_IQR_ratio_masked_over_vanilla.median(),
        'IQR_ratio_95CI_low':draws.median_IQR_ratio_masked_over_vanilla.quantile(.025),
        'IQR_ratio_95CI_high':draws.median_IQR_ratio_masked_over_vanilla.quantile(.975),
        'median_fraction_distances_broader_masked':draws.fraction_distances_broader_masked.median(),
        'fraction_draws_median_IQR_ratio_gt_1':(draws.median_IQR_ratio_masked_over_vanilla>1).mean()}])
    return draws,summary

def random_seed_saturation(frames,cols,budgets=(5,10,20,25,50,75,100),iterations=1000):
    """Random-seed saturation curves using one final-QC representative/trajectory."""
    reps={p:trajectory_representatives(f).reset_index(drop=True) for p,f in frames.items()}
    pooled=pd.concat([reps['vanilla'][cols],reps['masked'][cols]],ignore_index=True)
    scale=(pooled.quantile(.75)-pooled.quantile(.25)).to_numpy(float).copy()
    scale[scale==0]=np.nan
    full_breadth={}
    seed_rows={}
    rep_values={}
    for protocol,frame in reps.items():
        values=frame[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f'Saturation requires complete finite intrachain distances: {protocol}')
        rep_values[protocol]=values
        q=np.quantile(values,[.25,.75],axis=0); iq=q[1]-q[0]
        full_breadth[protocol]=np.nanmedian(iq/scale)
        seed_rows[protocol]={int(seed):group.index.to_numpy() for seed,group in frame.groupby('seed')}
    rng=np.random.default_rng(210403); rows=[]
    for n_seeds in budgets:
        n_iter=1 if n_seeds==100 else iterations
        for iteration in range(n_iter):
            row={'seeds_sampled':n_seeds,'iteration':iteration+1}
            for protocol,frame in reps.items():
                seeds=np.array(sorted(seed_rows[protocol]))
                chosen=seeds if n_seeds==len(seeds) else rng.choice(seeds,n_seeds,replace=False)
                indices=np.concatenate([seed_rows[protocol][int(seed)] for seed in chosen])
                values=rep_values[protocol][indices]
                q=np.quantile(values,[.25,.75],axis=0); iq=q[1]-q[0]
                breadth=np.nanmedian(iq/scale)
                row[f'{protocol}_retained_trajectories']=len(values)
                row[f'{protocol}_retention_fraction']=len(values)/(n_seeds*5)
                row[f'{protocol}_normalized_breadth']=breadth
                row[f'{protocol}_fraction_full_breadth']=breadth/(full_breadth[protocol]+EPS)
            row['breadth_gain_masked_over_vanilla']=(row['masked_normalized_breadth']+EPS)/(row['vanilla_normalized_breadth']+EPS)
            rows.append(row)
    draws=pd.DataFrame(rows)
    metrics=['vanilla_retention_fraction','masked_retention_fraction','vanilla_normalized_breadth',
             'masked_normalized_breadth','vanilla_fraction_full_breadth','masked_fraction_full_breadth',
             'breadth_gain_masked_over_vanilla']
    summary_rows=[]
    for n_seeds,g in draws.groupby('seeds_sampled',sort=True):
        row={'seeds_sampled':n_seeds,'iterations':len(g)}
        for metric in metrics:
            row[f'{metric}_median']=g[metric].median()
            row[f'{metric}_CI_low']=g[metric].quantile(.025)
            row[f'{metric}_CI_high']=g[metric].quantile(.975)
        row['fraction_draws_gain_gt_1']=(g.breadth_gain_masked_over_vanilla>1).mean()
        summary_rows.append(row)
    return draws,pd.DataFrame(summary_rows)

def seed_block_distribution_statistics(
    frames,cols,depth,n_permutations=999,n_bootstrap=500,base_seed=20260818
):
    """W1 and directional breadth statistics for a fixed seed cohort.

    Whole seeds, rather than individual trajectories, are exchanged or
    resampled so the five AlphaFold model trajectories remain clustered.
    """
    reps={p:trajectory_representatives(subset(f,depth)).reset_index(drop=True)
          for p,f in frames.items()}
    seed_indices={p:[g.index.to_numpy() for _,g in frame.groupby('seed',sort=True)]
                  for p,frame in reps.items()}
    n_seeds={p:len(seed_indices[p]) for p in ['vanilla','masked']}
    if n_seeds['vanilla'] != n_seeds['masked']:
        raise ValueError('Seed-block statistics require equal seed counts per protocol')
    n=n_seeds['vanilla']
    matrices={p:frame[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
              for p,frame in reps.items()}
    rows=[]
    for distance_index,distance in enumerate(cols):
        blocks={p:[matrices[p][idx,distance_index] for idx in seed_indices[p]]
                for p in ['vanilla','masked']}
        vanilla=np.concatenate(blocks['vanilla']); masked=np.concatenate(blocks['masked'])
        observed=wasserstein_distance(vanilla,masked)
        pooled=np.concatenate([vanilla,masked]); pooled_iqr=np.quantile(pooled,.75)-np.quantile(pooled,.25)
        viqr=np.quantile(vanilla,.75)-np.quantile(vanilla,.25)
        miqr=np.quantile(masked,.75)-np.quantile(masked,.25)
        ratio=(miqr+EPS)/(viqr+EPS); delta=np.median(masked)-np.median(vanilla)
        rng=np.random.default_rng(stable_seed(base_seed,'seed_blocks',str(depth),distance))
        combined=blocks['vanilla']+blocks['masked']; exceed=0; breadth_exceed=0
        observed_log_iqr=np.log2(ratio)
        for _ in range(n_permutations):
            order=rng.permutation(2*n)
            a=np.concatenate([combined[i] for i in order[:n]])
            b=np.concatenate([combined[i] for i in order[n:]])
            exceed += wasserstein_distance(a,b) >= observed-1e-15
            aiqr=np.quantile(a,.75)-np.quantile(a,.25); biqr=np.quantile(b,.75)-np.quantile(b,.25)
            breadth_exceed += np.log2((biqr+EPS)/(aiqr+EPS)) >= observed_log_iqr-1e-15
        p=(1+exceed)/(1+n_permutations)
        p_broader=(1+breadth_exceed)/(1+n_permutations)
        boot=np.empty((n_bootstrap,3),float)
        for iteration in range(n_bootstrap):
            a=np.concatenate([blocks['vanilla'][i] for i in rng.integers(0,n,n)])
            b=np.concatenate([blocks['masked'][i] for i in rng.integers(0,n,n)])
            aiqr=np.quantile(a,.75)-np.quantile(a,.25); biqr=np.quantile(b,.75)-np.quantile(b,.25)
            boot[iteration]=[wasserstein_distance(a,b),np.median(b)-np.median(a),
                             np.log2((biqr+EPS)/(aiqr+EPS))]
        ci=np.quantile(boot,[.025,.975],axis=0)
        rows.append({'distance':distance,'distance_type':'CA' if distance.startswith('CA_') else 'shortest_heavy',
            'vanilla_trajectories':len(vanilla),'masked_trajectories':len(masked),
            'cohort':'first100' if depth==FOCAL_DEPTH else 'full_QC',
            'vanilla_seeds':n,'masked_seeds':n,'vanilla_median_A':np.median(vanilla),
            'masked_median_A':np.median(masked),'delta_median_A':delta,
            'vanilla_IQR_A':viqr,'masked_IQR_A':miqr,'IQR_ratio_masked_over_vanilla':ratio,
            'log2_IQR_ratio':np.log2(ratio),'W1_A':observed,'pooled_IQR_A':pooled_iqr,
            'W1_normalized_by_pooled_IQR':observed/(pooled_iqr+EPS),
            'p_W1_seed_block_permutation':p,'p_broader_masked_seed_block_permutation':p_broader,
            'W1_CI_low_A':ci[0,0],'W1_CI_high_A':ci[1,0],
            'delta_median_CI_low_A':ci[0,1],'delta_median_CI_high_A':ci[1,1],
            'log2_IQR_ratio_CI_low':ci[0,2],'log2_IQR_ratio_CI_high':ci[1,2],
            'n_permutations':n_permutations,'n_bootstrap':n_bootstrap})
    result=pd.DataFrame(rows)
    result['q_W1_seed_block_BH']=multipletests(result.p_W1_seed_block_permutation,method='fdr_bh')[1]
    result['q_broader_masked_seed_block_BH']=multipletests(
        result.p_broader_masked_seed_block_permutation,method='fdr_bh')[1]
    sig=result.q_W1_seed_block_BH<.05
    broader=(result.q_broader_masked_seed_block_BH<.05)&(result.log2_IQR_ratio>0)
    summary=pd.DataFrame([{'distances':len(result),'CA_distances':(result.distance_type=='CA').sum(),
        'shortest_heavy_distances':(result.distance_type=='shortest_heavy').sum(),
        'median_W1_A':result.W1_A.median(),'median_normalized_W1':result.W1_normalized_by_pooled_IQR.median(),
        'median_IQR_ratio_masked_over_vanilla':result.IQR_ratio_masked_over_vanilla.median(),
        'n_W1_q_lt_0.05':int(sig.sum()),'fraction_W1_q_lt_0.05':sig.mean(),
        'n_significant_broader_masked':int((sig&(result.log2_IQR_ratio>0)).sum()),
        'n_significant_narrower_masked':int((sig&(result.log2_IQR_ratio<0)).sum()),
        'n_directionally_broader_masked_q_lt_0.05':int(broader.sum()),
        'fraction_directionally_broader_masked_q_lt_0.05':broader.mean(),
        'n_permutations_per_distance':n_permutations,'n_bootstrap_per_distance':n_bootstrap,
        'permutation_unit':'seed block','bootstrap_unit':'seed block'}])
    return result,summary

def first100_seed_block_distribution_statistics(frames,cols,**kwargs):
    return seed_block_distribution_statistics(frames,cols,FOCAL_DEPTH,**kwargs)

def savefig(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white'); fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)

def figures(results,summary,stable,frames,trajectory_audit,seed_breadth,rarefaction,saturation,distribution_stats,full_distribution_stats):
    apply_kv21_style(); early=results[results.depth.astype(str)==str(FOCAL_DEPTH)]; full=results[results.depth.astype(str)=='Full QC']
    fig,axs=plt.subplots(1,2,figsize=(12,4.8))
    for ax,g,title in [(axs[0],early,'Nominal first 100 (20 seeds × 5 models)'),(axs[1],full,'Full QC')]:
        sns.histplot(g.global_log2_IQR_ratio,bins=55,color=KV21_PALETTE['L403A_HM'],stat='density',alpha=.65,ax=ax)
        ax.axvline(0,color='.25',lw=1); ax.axvline(g.global_log2_IQR_ratio.median(),color=KV21_PALETTE['L403A_HM'],ls='--'); ax.set_title(title); ax.set_xlabel('log2 masked/vanilla global IQR')
    axs[0].set_ylabel(f'Density across {len(early):,} intrachain distances'); fig.suptitle('Chain-label-safe intrachain L403A ensemble breadth'); fig.tight_layout(); savefig(fig,'all_distance_iqr_ratio_distributions')
    z=early[['distance','global_log2_IQR_ratio']].merge(full[['distance','global_log2_IQR_ratio']],on='distance',suffixes=('_first20seeds','_full'))
    fig,ax=plt.subplots(figsize=(6.5,6)); ax.scatter(z.global_log2_IQR_ratio_first20seeds,z.global_log2_IQR_ratio_full,s=8,alpha=.3,color=KV21_PALETTE['L403A_HM']); lim=np.nanmax(np.abs(z.filter(like='ratio').to_numpy())); ax.plot([-lim,lim],[-lim,lim],color='.3',lw=.8); ax.axhline(0,color='.7',lw=.6); ax.axvline(0,color='.7',lw=.6); ax.set(xlabel='Nominal first 100 (20 seeds) log2 IQR ratio',ylabel='Full-QC log2 IQR ratio',title='Does the fixed early cohort reproduce full QC?'); fig.tight_layout(); savefig(fig,'first100_nominal_vs_full_distance_breadth')
    # Top reproducible broadened/narrowed distances ranked by the weaker absolute effect.
    z['concordant']=np.sign(z.global_log2_IQR_ratio_first20seeds)==np.sign(z.global_log2_IQR_ratio_full)
    z['min_abs']=np.minimum(abs(z.global_log2_IQR_ratio_first20seeds),abs(z.global_log2_IQR_ratio_full))
    broad=z[z.concordant&(z.global_log2_IQR_ratio_full>0)].nlargest(15,'min_abs'); narrow=z[z.concordant&(z.global_log2_IQR_ratio_full<0)].nlargest(15,'min_abs'); top=pd.concat([broad,narrow]).set_index('distance')
    heat=top[['global_log2_IQR_ratio_first20seeds','global_log2_IQR_ratio_full']]; lim=np.nanmax(abs(heat.to_numpy())); fig,ax=plt.subplots(figsize=(8,12)); sns.heatmap(heat,cmap=sns.diverging_palette(18,135,as_cmap=True),center=0,vmin=-lim,vmax=lim,annot=True,fmt='.2f',ax=ax,cbar_kws={'label':'log2 masked/vanilla IQR'}); ax.set_xticklabels(['Nominal first 100','Full QC'],rotation=0); ax.set_ylabel(''); ax.set_title('Largest concordant all-distance breadth changes'); fig.tight_layout(); savefig(fig,'top_all_distance_breadth_heatmap')
    top.reset_index().to_csv(TAB/'top_concordant_distance_breadth_changes.csv',index=False)
    # Representative raw distributions: two strongest broadened contacts and
    # one strongest narrowed contact, using one median per trajectory.
    representative=pd.concat([broad.head(2),narrow.head(1)]).distance.tolist()
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for row,depth in enumerate([FOCAL_DEPTH,'Full QC']):
        records=[]
        for protocol in ['vanilla','masked']:
            x=subset(frames[protocol],depth)
            tm=trajectory_representatives(x)[['seed','model_number',*representative]]
            records.append(tm.melt(id_vars=['seed','model_number'],value_vars=representative,var_name='distance',value_name='Distance_A').assign(protocol=protocol))
        plot=pd.concat(records,ignore_index=True)
        for col,distance in enumerate(representative):
            ax=axs[row,col]; z=plot[plot.distance==distance]
            sns.violinplot(data=z,x='protocol',y='Distance_A',order=['vanilla','masked'],
                palette=[KV21_PALETTE['L403A_VAN'],KV21_PALETTE['L403A_HM']],inner='quart',cut=0,linewidth=.7,ax=ax)
            ax.set_title(distance.replace('CA_CA_','').replace('_CA','') if row==0 else '')
            ax.set_xlabel('' if row==0 else str(depth)); ax.set_ylabel('Distance (Å)' if col==0 else '')
    fig.suptitle('Chain-label-safe intrachain distance examples\none final-QC representative per trajectory; upper: nominal first 100, lower: full QC',y=1.02)
    fig.tight_layout(); savefig(fig,'representative_all_distance_distributions')
    order=[4,8,12,16,20,'Full QC']; labels=['20','40','60','80','100','Full QC\n(500 nominal)']; s=summary.set_index('depth').reindex(order); fig,ax=plt.subplots(figsize=(8.5,4.8)); ax.plot(range(6),s.fraction_global_broader_masked,marker='o',color=KV21_PALETTE['L403A_HM'],label='Global IQR broader'); ax.plot(range(6),s.fraction_significant_broader_masked,marker='o',color=KV21_PALETTE['L403A_VAN'],label='Seed breadth broader, q<0.05'); ax.plot(range(6),s.fraction_significant_narrower_masked,marker='o',color=KV21_PALETTE['WT_HM'],label='Seed breadth narrower, q<0.05'); ax.set_xticks(range(6),labels); ax.set_ylabel(f'Fraction of {len(results.distance.unique()):,} intrachain distances'); ax.set_xlabel('Nominal generated-trajectory budget'); ax.set_title('How widespread is chain-label-safe masked broadening?'); ax.legend(); fig.tight_layout(); savefig(fig,'all_distance_breadth_fraction_by_depth')
    # QC survival within the fixed nominal first-100 cohort. Cell annotations
    # are the number of retained model/recycle rows; zero denotes exclusion.
    fig,axs=plt.subplots(1,2,figsize=(9,10),sharex=True)
    for ax,protocol,color in zip(axs,['vanilla','masked'],[KV21_PALETTE['L403A_VAN'],KV21_PALETTE['L403A_HM']]):
        z=trajectory_audit[trajectory_audit.protocol==protocol]
        matrix=z.pivot(index='seed',columns='model_number',values='final_qc_rows')
        sns.heatmap(matrix,cmap=sns.light_palette(color,as_cmap=True),annot=True,fmt='g',cbar=False,
            linewidths=.5,linecolor='white',vmin=0,vmax=matrix.to_numpy().max(),ax=ax)
        retained=int(z.final_qc_retained.sum())
        ax.set_title(f'{protocol.capitalize()}: {retained}/100 retained')
        ax.set_xlabel('Model trajectory'); ax.set_ylabel('Seed' if protocol=='vanilla' else '')
    fig.suptitle('Final-QC survival within nominal first 100 trajectories\n20 ordered seeds × 5 models; annotations = retained recycle rows',y=.995)
    fig.tight_layout(); savefig(fig,'first100_nominal_trajectory_qc_audit')
    # Overall cluster-aware breadth and retained-count-matched sensitivity.
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    sns.violinplot(data=seed_breadth,x='protocol',y='median_normalized_seed_IQR',order=['vanilla','masked'],
        hue='protocol',palette={'vanilla':KV21_PALETTE['L403A_VAN'],'masked':KV21_PALETTE['L403A_HM']},
        legend=False,inner=None,cut=0,ax=axs[0])
    sns.stripplot(data=seed_breadth,x='protocol',y='median_normalized_seed_IQR',order=['vanilla','masked'],
        color='.2',size=4,alpha=.7,ax=axs[0])
    axs[0].set(xlabel='',ylabel='Median normalized within-seed IQR',title='One breadth value per seed (20 vs 20)')
    sns.histplot(rarefaction.median_IQR_ratio_masked_over_vanilla,bins=35,
        color=KV21_PALETTE['L403A_HM'],ax=axs[1])
    axs[1].axvline(1,color='.25',ls='--',lw=1); axs[1].set(
        xlabel='Median masked/vanilla IQR ratio',ylabel='Rarefaction draws',
        title='Equal retained count: 85 vs 85 trajectories')
    fig.suptitle('Does masking yield more distance breadth within the same nominal first-100 budget?')
    fig.tight_layout(); savefig(fig,'first100_bang_for_buck_breadth_statistics')
    # Random-seed saturation: typical performance at a fixed input-seed budget.
    fig,axs=plt.subplots(2,2,figsize=(12,9),sharex=True)
    x=saturation.seeds_sampled.to_numpy(float)
    for protocol,color,label in [('vanilla',KV21_PALETTE['L403A_VAN'],'Vanilla'),('masked',KV21_PALETTE['L403A_HM'],'Masked')]:
        for ax,metric,ylabel in [(axs[0,0],'retention_fraction','Final-QC retention fraction'),
                                 (axs[0,1],'normalized_breadth','Median normalized intrachain IQR'),
                                 (axs[1,0],'fraction_full_breadth','Fraction of protocol full-QC breadth')]:
            y=saturation[f'{protocol}_{metric}_median'].to_numpy(float)
            lo=saturation[f'{protocol}_{metric}_CI_low'].to_numpy(float); hi=saturation[f'{protocol}_{metric}_CI_high'].to_numpy(float)
            ax.plot(x,y,marker='o',color=color,label=label); ax.fill_between(x,lo,hi,color=color,alpha=.18)
            ax.set_ylabel(ylabel)
    gain=saturation.breadth_gain_masked_over_vanilla_median.to_numpy(float)
    lo=saturation.breadth_gain_masked_over_vanilla_CI_low.to_numpy(float); hi=saturation.breadth_gain_masked_over_vanilla_CI_high.to_numpy(float)
    axs[1,1].plot(x,gain,marker='o',color=KV21_PALETTE['L403A_HM']); axs[1,1].fill_between(x,lo,hi,color=KV21_PALETTE['L403A_HM'],alpha=.18)
    axs[1,1].axhline(1,color='.3',ls='--',lw=1); axs[1,1].set_ylabel('Masked/vanilla breadth gain')
    for ax in axs[1]: ax.set_xlabel('Randomly sampled input seeds')
    for ax in axs.flat: ax.set_xticks([5,10,20,25,50,75,100]); sns.despine(ax=ax)
    axs[0,0].legend(); fig.suptitle('L403A sampling efficiency and saturation\n1,000 random seed subsets per budget; one final-QC representative per retained trajectory')
    fig.tight_layout(); savefig(fig,'random_seed_sampling_efficiency_saturation')
    plot=distribution_stats.copy(); sig=plot.q_W1_seed_block_BH<.05
    plot['result']='Not significant'
    plot.loc[sig&(plot.log2_IQR_ratio>0),'result']='Broader masked'
    plot.loc[sig&(plot.log2_IQR_ratio<0),'result']='Narrower masked'
    colors={'Not significant':'#B8B8B8','Broader masked':KV21_PALETTE['L403A_HM'],'Narrower masked':KV21_PALETTE['WT_HM']}
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    for category in ['Not significant','Narrower masked','Broader masked']:
        z=plot[plot.result==category]
        axs[0].scatter(z.log2_IQR_ratio,-np.log10(z.q_W1_seed_block_BH.clip(lower=1e-300)),
                       s=18,alpha=.65,color=colors[category],label=f'{category} (n={len(z)})')
    axs[0].axvline(0,color='.35',lw=.8); axs[0].axhline(-np.log10(.05),color='.35',ls='--',lw=.8)
    axs[0].set(xlabel='log₂ masked/vanilla IQR',ylabel='−log₁₀ BH q',title='Seed-block W1 permutation tests')
    axs[0].legend(fontsize=8)
    sns.histplot(plot.W1_normalized_by_pooled_IQR,bins=40,color=KV21_PALETTE['L403A_HM'],ax=axs[1])
    axs[1].axvline(plot.W1_normalized_by_pooled_IQR.median(),color='.25',ls='--',lw=1)
    axs[1].set(xlabel='W1 / pooled IQR',ylabel='Intrachain distances',title='Normalized distributional separation')
    fig.suptitle('Nominal first 100: chain-label-safe distribution statistics')
    fig.tight_layout(); savefig(fig,'first100_seed_block_distribution_statistics')

    # Manuscript-style first-100 summary: one question per panel and no raw
    # chain-label-sensitive Kv2.1 coordinates.
    breadth_summary=pd.read_csv(TAB/'l403a_first100_seed_level_global_breadth_summary.csv').iloc[0]
    retention_summary=(trajectory_audit.groupby('protocol',as_index=False)
        .agg(retained=('final_qc_retained','sum'),nominal=('final_qc_retained','size')))
    fig,axs=plt.subplots(2,2,figsize=(12.5,9))
    sns.violinplot(data=seed_breadth,x='protocol',y='median_normalized_seed_IQR',order=['vanilla','masked'],
        hue='protocol',palette={'vanilla':KV21_PALETTE['L403A_VAN'],'masked':KV21_PALETTE['L403A_HM']},
        legend=False,inner=None,cut=0,linewidth=.8,ax=axs[0,0])
    sns.stripplot(data=seed_breadth,x='protocol',y='median_normalized_seed_IQR',order=['vanilla','masked'],
        color='.18',size=5,jitter=.13,alpha=.75,ax=axs[0,0])
    med=seed_breadth.groupby('protocol').median_normalized_seed_IQR.median()
    axs[0,0].set(xlabel='',ylabel='Median normalized within-seed IQR',title='A  Breadth per independent seed')
    axs[0,0].text(.03,.97,f"Median: {med['vanilla']:.2f} → {med['masked']:.2f} ({med['masked']/med['vanilla']:.2f}×)\nMann–Whitney p = {breadth_summary.mannwhitney_p_two_sided:.1e}",
        transform=axs[0,0].transAxes,va='top',fontsize=10)

    effects=distribution_stats.log2_IQR_ratio.replace([np.inf,-np.inf],np.nan).dropna()
    display_effects=effects.clip(-3,3)
    axs[0,1].hist(display_effects[effects<=0],bins=np.linspace(-3,0,25),color=KV21_PALETTE['WT_HM'],alpha=.8,label='Narrower masked')
    axs[0,1].hist(display_effects[effects>0],bins=np.linspace(0,3,25),color=KV21_PALETTE['L403A_HM'],alpha=.8,label='Broader masked')
    axs[0,1].axvline(0,color='.25',lw=1); axs[0,1].axvline(effects.median(),color='.2',ls='--',lw=1.2)
    axs[0,1].set(xlim=(-3,3),xlabel='log₂ masked/vanilla IQR (display clipped at ±3)',ylabel='Intrachain distances',title='B  Breadth effect across 546 distances')
    axs[0,1].text(.03,.97,f"Median IQR ratio = {2**effects.median():.2f}×\n{(effects>0).mean():.1%} broader descriptively",
        transform=axs[0,1].transAxes,va='top',fontsize=10); axs[0,1].legend(fontsize=8)

    direct=((distribution_stats.q_broader_masked_seed_block_BH<.05)&(distribution_stats.log2_IQR_ratio>0))
    category=pd.DataFrame({'result':['Broader\nBH q < 0.05','Not supported'],'count':[int(direct.sum()),int((~direct).sum())]})
    sns.barplot(data=category,x='result',y='count',hue='result',legend=False,
        palette=[KV21_PALETTE['L403A_HM'],'#D8DDD9'],ax=axs[1,0])
    for patch,count in zip(axs[1,0].patches,category['count']):
        axs[1,0].text(patch.get_x()+patch.get_width()/2,patch.get_height()+8,f'{count}/546\n({count/546:.1%})',ha='center',fontweight='bold')
    axs[1,0].set(xlabel='',ylabel='Intrachain distances',ylim=(0,590),title='C  Direct directional broadening test')
    axs[1,0].text(.03,.97,'One-sided seed-block permutation\nBH-FDR across 546 distances',transform=axs[1,0].transAxes,va='top',fontsize=10)

    retention_summary['label']=retention_summary.protocol.str.capitalize()
    colors=[KV21_PALETTE['L403A_VAN'] if p=='vanilla' else KV21_PALETTE['L403A_HM'] for p in retention_summary.protocol]
    bars=axs[1,1].bar(retention_summary.label,retention_summary.retained,color=colors,width=.62)
    for bar,(_,row) in zip(bars,retention_summary.iterrows()):
        axs[1,1].text(bar.get_x()+bar.get_width()/2,bar.get_height()+2,f"{int(row.retained)}/{int(row.nominal)}",ha='center',fontweight='bold',fontsize=12)
    axs[1,1].axhline(100,color='.35',ls='--',lw=.8); axs[1,1].set(ylim=(0,112),ylabel='Trajectories retained after final QC',title='D  Same nominal budget, different retention')
    axs[1,1].text(.03,.88,'Fixed first 20 seeds × 5 models\nFailed trajectories are not replaced',transform=axs[1,1].transAxes,va='top',fontsize=10)
    for ax in axs.flat: sns.despine(ax=ax)
    fig.suptitle('Kv2.1 L403A: masked sampling yields more breadth within the first 100 trajectories',fontsize=16,fontweight='semibold')
    fig.tight_layout(); savefig(fig,'first100_masked_sampling_breadth_main_summary')

    # Heatmap scorecard designed to make the retention-versus-breadth tradeoff
    # readable without combining unlike quantities on one color scale.
    protocol_score=pd.DataFrame({
        'Vanilla':[float(retention_summary.loc[retention_summary.protocol=='vanilla','retained'].iloc[0])/100,med['vanilla']],
        'Masked':[float(retention_summary.loc[retention_summary.protocol=='masked','retained'].iloc[0])/100,med['masked']]},
        index=['Final-QC retention','Median breadth per seed'])
    protocol_ann=pd.DataFrame({
        'Vanilla':['99/100\n(99%)',f"{med['vanilla']:.2f}"],
        'Masked':['85/100\n(85%)',f"{med['masked']:.2f}" ]},index=protocol_score.index)
    magnitude=pd.DataFrame({'First 100':[2**effects.median()]},index=['Median masked/vanilla IQR'])
    magnitude_ann=pd.DataFrame({'First 100':[f'{2**effects.median():.2f}×']},index=magnitude.index)
    prevalence=pd.DataFrame({'First 100':[(effects>0).mean(),direct.mean()]},
        index=['Broader descriptively','Directionally broader, BH q<0.05'])
    prevalence_ann=prevalence.map(lambda x:f'{x:.1%}')
    fig,axs=plt.subplots(1,3,figsize=(14.5,4.8),gridspec_kw={'width_ratios':[1.35,.8,1.15]})
    sns.heatmap(protocol_score,annot=protocol_ann,fmt='',vmin=0,vmax=1,
        cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),linewidths=1,linecolor='white',
        cbar_kws={'label':'Fraction or normalized breadth'},annot_kws={'fontsize':13,'fontweight':'bold'},ax=axs[0])
    sns.heatmap(magnitude,annot=magnitude_ann,fmt='',vmin=1,vmax=2,
        cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),linewidths=1,linecolor='white',
        cbar_kws={'label':'IQR ratio'},annot_kws={'fontsize':16,'fontweight':'bold'},ax=axs[1])
    sns.heatmap(prevalence,annot=prevalence_ann,fmt='',vmin=0,vmax=1,
        cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),linewidths=1,linecolor='white',
        cbar_kws={'label':'Fraction of 546 distances'},annot_kws={'fontsize':14,'fontweight':'bold'},ax=axs[2])
    for ax,title in zip(axs,['Protocol trade-off','Typical distance effect','How widespread?']):
        ax.set_title(title,fontweight='bold',fontsize=13); ax.set_xlabel(''); ax.set_ylabel('')
        ax.tick_params(axis='x',rotation=0,labelsize=10); ax.tick_params(axis='y',rotation=0,labelsize=10)
    axs[0].text(.5,-.27,'Masked: −14 percentage points retained, +0.27 normalized breadth',
        transform=axs[0].transAxes,ha='center',fontsize=10,fontweight='bold',color=KV21_PALETTE['L403A_HM'])
    fig.suptitle('First 100 nominal trajectories: masking trades retention for broader structural sampling',
        fontsize=16,fontweight='semibold')
    fig.tight_layout(); savefig(fig,'first100_retention_vs_breadth_heatmap_scorecard')

    # A compact map of every analyzed distance. Columns are sorted by effect;
    # the lower stripe marks which effects pass the direct broadening test.
    ordered=distribution_stats.sort_values('log2_IQR_ratio').reset_index(drop=True)
    effect_row=ordered.log2_IQR_ratio.clip(-3,3).to_numpy()[None,:]
    sig_row=((ordered.q_broader_masked_seed_block_BH<.05)&(ordered.log2_IQR_ratio>0)).astype(int).to_numpy()[None,:]
    fig,axs=plt.subplots(2,1,figsize=(13,2.8),gridspec_kw={'height_ratios':[2,1]},sharex=True)
    sns.heatmap(effect_row,cmap=sns.diverging_palette(18,135,as_cmap=True),center=0,vmin=-3,vmax=3,
        xticklabels=False,yticklabels=['IQR effect'],cbar_kws={'label':'log₂ masked/vanilla IQR'},ax=axs[0])
    sns.heatmap(sig_row,cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),vmin=0,vmax=1,
        xticklabels=False,yticklabels=['Broader q<0.05'],cbar=False,ax=axs[1])
    axs[1].set_xlabel('All 546 intrachain distances, ordered from narrower to broader under masking')
    fig.suptitle('First 100: effect size and direct evidence of masked broadening',fontweight='semibold')
    fig.tight_layout(); savefig(fig,'first100_all_distance_breadth_evidence_map')
    # Direct first-100 versus full-QC comparison. Asterisks mark a one-sided
    # seed-block permutation test of masked IQR > vanilla IQR after BH-FDR.
    paired=distribution_stats.merge(full_distribution_stats,on='distance',suffixes=('_first100','_full'))
    paired['min_broadening']=paired[['log2_IQR_ratio_first100','log2_IQR_ratio_full']].min(axis=1)
    chosen=paired.nlargest(30,'min_broadening').copy().set_index('distance')
    heat=chosen[['log2_IQR_ratio_first100','log2_IQR_ratio_full']]
    annotations=heat.map(lambda x:f'{x:.2f}')
    for suffix,column in [('first100','log2_IQR_ratio_first100'),('full','log2_IQR_ratio_full')]:
        sig=chosen[f'q_broader_masked_seed_block_BH_{suffix}']<.05
        annotations.loc[sig,column]=annotations.loc[sig,column]+'*'
    lim=max(1,np.nanmax(abs(heat.to_numpy())))
    fig,ax=plt.subplots(figsize=(8.5,12))
    sns.heatmap(heat,cmap=sns.diverging_palette(18,135,as_cmap=True),center=0,vmin=-lim,vmax=lim,
                annot=annotations,fmt='',linewidths=.35,linecolor='white',ax=ax,
                cbar_kws={'label':'log₂ masked/vanilla IQR'})
    ax.set_xticklabels(['Nominal first 100','Full QC'],rotation=0); ax.set_ylabel('')
    ax.set_title('Strongest reproducible masked broadening\n* one-sided seed-block permutation, BH q < 0.05')
    fig.tight_layout(); savefig(fig,'first100_vs_full_directional_breadth_heatmap')
    paired.to_csv(TAB/'l403a_first100_vs_full_directional_breadth_statistics.csv',index=False)

    # Compact summary heatmaps keep counts/fractions separate from continuous
    # effect sizes so unlike quantities are not encoded on one color scale.
    cohorts={'Nominal first 100':distribution_stats,'Full QC':full_distribution_stats}
    effect=pd.DataFrame({name:{'Median masked/vanilla IQR':g.IQR_ratio_masked_over_vanilla.median(),
        'Median W1 / pooled IQR':g.W1_normalized_by_pooled_IQR.median()} for name,g in cohorts.items()})
    fractions=pd.DataFrame({name:{'Masked IQR > vanilla IQR':(g.log2_IQR_ratio>0).mean(),
        'Directional BH q < 0.05':((g.q_broader_masked_seed_block_BH<.05)&(g.log2_IQR_ratio>0)).mean()}
        for name,g in cohorts.items()})
    fig,axs=plt.subplots(1,2,figsize=(13.5,5.4))
    sns.heatmap(effect,annot=True,fmt='.2f',cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),
                vmin=0,cbar_kws={'label':'Effect size'},ax=axs[0])
    sns.heatmap(fractions,annot=True,fmt='.1%',cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),
                vmin=0,vmax=1,cbar_kws={'label':'Fraction of 546 distances'},ax=axs[1])
    for ax in axs: ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(axis='x',rotation=0,labelsize=9); ax.tick_params(axis='y',labelsize=9)
    axs[0].set_title('Typical effect'); axs[1].set_title('Prevalence of broadening')
    fig.suptitle('Masked versus vanilla: nominal first 100 and full QC')
    fig.tight_layout(); savefig(fig,'first100_vs_full_breadth_summary_heatmaps')

    # 2 × 2 reproducibility table: an immediately readable count of which
    # distances pass the direct directional broadening test in each cohort.
    sig100=(paired.q_broader_masked_seed_block_BH_first100<.05)&(paired.log2_IQR_ratio_first100>0)
    sigfull=(paired.q_broader_masked_seed_block_BH_full<.05)&(paired.log2_IQR_ratio_full>0)
    concordance=pd.DataFrame([[int((~sig100&~sigfull).sum()),int((~sig100&sigfull).sum())],
                              [int((sig100&~sigfull).sum()),int((sig100&sigfull).sum())]],
        index=['Not broader','Broader, q < 0.05'],columns=['Not broader','Broader, q < 0.05'])
    fig,ax=plt.subplots(figsize=(7.5,6))
    sns.heatmap(concordance,annot=True,fmt='d',cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),
                cbar_kws={'label':'Number of intrachain distances'},linewidths=1,linecolor='white',ax=ax,
                annot_kws={'fontsize':16})
    ax.set_xlabel('Full QC'); ax.set_ylabel('Nominal first 100')
    ax.set_title('Does significant masked broadening reproduce?\nOne-sided seed-block permutation; BH q < 0.05')
    fig.tight_layout(); savefig(fig,'first100_vs_full_broadening_concordance_heatmap')
    concordance.to_csv(TAB/'l403a_first100_vs_full_broadening_concordance.csv')

    # Every distance in one quadrant plot; colors encode reproducibility of the
    # direct test, while the axes encode effect magnitude rather than p-values.
    paired['significance']='Neither cohort'
    paired.loc[sig100&~sigfull,'significance']='First 100 only'
    paired.loc[~sig100&sigfull,'significance']='Full QC only'
    paired.loc[sig100&sigfull,'significance']='Both cohorts'
    point_colors={'Neither cohort':'#B8B8B8','First 100 only':KV21_PALETTE['L403A_VAN'],
                  'Full QC only':KV21_PALETTE['WT_HM'],'Both cohorts':KV21_PALETTE['L403A_HM']}
    fig,ax=plt.subplots(figsize=(7.5,6.5))
    for category in ['Neither cohort','First 100 only','Full QC only','Both cohorts']:
        g=paired[paired.significance==category]
        ax.scatter(g.log2_IQR_ratio_first100,g.log2_IQR_ratio_full,s=22,alpha=.65,
                   color=point_colors[category],label=f'{category} (n={len(g)})')
    lim=np.nanquantile(np.abs(paired[['log2_IQR_ratio_first100','log2_IQR_ratio_full']]),.99)
    ax.plot([-lim,lim],[-lim,lim],color='.3',lw=1); ax.axhline(0,color='.65',lw=.8); ax.axvline(0,color='.65',lw=.8)
    ax.set(xlim=(-lim,lim),ylim=(-lim,lim),xlabel='First 100: log₂ masked/vanilla IQR',
           ylabel='Full QC: log₂ masked/vanilla IQR',title='Breadth effects agree from early to full sampling')
    ax.legend(fontsize=8,loc='lower right'); sns.despine(ax=ax); fig.tight_layout()
    savefig(fig,'first100_vs_full_breadth_quadrant_scatter')

    # Stratify the headline result by distance definition. This checks that the
    # conclusion is not carried solely by one of the two measurement families.
    type_rows=[]
    for cohort,g in [('Nominal first 100',distribution_stats),('Full QC',full_distribution_stats)]:
        for distance_type,label in [('CA','Cα'),('shortest_heavy','Shortest-heavy')]:
            x=g[g.distance_type==distance_type]
            type_rows.append({'cohort':cohort,'distance_type':label,
                'median_IQR_ratio':x.IQR_ratio_masked_over_vanilla.median(),
                'fraction_broader':(x.log2_IQR_ratio>0).mean(),
                'fraction_significantly_broader':((x.q_broader_masked_seed_block_BH<.05)&(x.log2_IQR_ratio>0)).mean()})
    type_summary=pd.DataFrame(type_rows)
    ratio=type_summary.pivot(index='distance_type',columns='cohort',values='median_IQR_ratio').reindex(['Cα','Shortest-heavy'])
    sigfrac=type_summary.pivot(index='distance_type',columns='cohort',values='fraction_significantly_broader').reindex(['Cα','Shortest-heavy'])
    ratio=ratio[['Nominal first 100','Full QC']]; sigfrac=sigfrac[['Nominal first 100','Full QC']]
    fig,axs=plt.subplots(1,2,figsize=(10.5,4.5))
    sns.heatmap(ratio,annot=True,fmt='.2f',vmin=1,cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),
                cbar_kws={'label':'Median masked/vanilla IQR'},ax=axs[0])
    sns.heatmap(sigfrac,annot=True,fmt='.1%',vmin=0,vmax=1,cmap=sns.light_palette(KV21_PALETTE['L403A_HM'],as_cmap=True),
                cbar_kws={'label':'Fraction of distances'},ax=axs[1])
    axs[0].set_title('Broadening magnitude'); axs[1].set_title('Significantly broader')
    for ax in axs: ax.set_xlabel(''); ax.set_ylabel(''); ax.tick_params(axis='x',rotation=0)
    fig.suptitle('Conclusion is consistent across distance definitions')
    fig.tight_layout(); savefig(fig,'first100_vs_full_breadth_by_distance_type_heatmaps')
    type_summary.to_csv(TAB/'l403a_first100_vs_full_breadth_by_distance_type.csv',index=False)

def run():
    frames,cols=load(); trajectory_audit,retention_summary=nominal_trajectory_retention_audit(frames,FOCAL_DEPTH)
    results=pd.concat([analyze_depth(frames,cols,d) for d in DEPTHS],ignore_index=True); summary=summarize(results); stable=stability(results)
    seed_breadth,seed_breadth_summary=overall_seed_breadth(frames,cols)
    rarefaction,rarefaction_summary=equal_retained_count_rarefaction(frames,cols)
    saturation_draws,saturation_summary=random_seed_saturation(frames,cols)
    distribution_stats,distribution_stats_summary=first100_seed_block_distribution_statistics(frames,cols)
    full_distribution_stats,full_distribution_stats_summary=seed_block_distribution_statistics(frames,cols,'Full QC')
    retention_counts=retention_summary.set_index('protocol')
    retention_odds,retention_p=fisher_exact([
        [retention_counts.loc['masked','retained_trajectories'],retention_counts.loc['masked','excluded_trajectories']],
        [retention_counts.loc['vanilla','retained_trajectories'],retention_counts.loc['vanilla','excluded_trajectories']]],alternative='two-sided')
    retention_test=pd.DataFrame([{'test':'two-sided Fisher exact','masked_retained':int(retention_counts.loc['masked','retained_trajectories']),
        'masked_excluded':int(retention_counts.loc['masked','excluded_trajectories']),
        'vanilla_retained':int(retention_counts.loc['vanilla','retained_trajectories']),
        'vanilla_excluded':int(retention_counts.loc['vanilla','excluded_trajectories']),
        'odds_ratio_masked_vs_vanilla':retention_odds,'p':retention_p}])
    results.to_csv(TAB/'l403a_all_distance_sampling_statistics.csv',index=False); summary.to_csv(TAB/'l403a_all_distance_sampling_summary.csv',index=False); stable.to_csv(TAB/'l403a_all_distance_sampling_stability.csv',index=False)
    trajectory_audit.to_csv(TAB/'l403a_first100_nominal_trajectory_qc_audit.csv',index=False)
    retention_summary.to_csv(TAB/'l403a_first100_nominal_trajectory_qc_summary.csv',index=False)
    retention_test.to_csv(TAB/'l403a_first100_qc_retention_fisher_test.csv',index=False)
    seed_breadth.to_csv(TAB/'l403a_first100_seed_level_global_breadth.csv',index=False)
    seed_breadth_summary.to_csv(TAB/'l403a_first100_seed_level_global_breadth_summary.csv',index=False)
    rarefaction.to_csv(TAB/'l403a_first100_equal_count_rarefaction.csv',index=False)
    rarefaction_summary.to_csv(TAB/'l403a_first100_equal_count_rarefaction_summary.csv',index=False)
    saturation_draws.to_csv(TAB/'l403a_random_seed_saturation_draws.csv',index=False)
    saturation_summary.to_csv(TAB/'l403a_random_seed_saturation_summary.csv',index=False)
    distribution_stats.to_csv(TAB/'l403a_first100_seed_block_distribution_statistics.csv',index=False)
    distribution_stats_summary.to_csv(TAB/'l403a_first100_seed_block_distribution_statistics_summary.csv',index=False)
    full_distribution_stats.to_csv(TAB/'l403a_full_seed_block_distribution_statistics.csv',index=False)
    full_distribution_stats_summary.to_csv(TAB/'l403a_full_seed_block_distribution_statistics_summary.csv',index=False)
    figures(results,summary,stable,frames,trajectory_audit,seed_breadth,rarefaction,saturation_summary,distribution_stats,full_distribution_stats)
    paper=summary.copy()
    for fraction,count in [('fraction_global_broader_masked','n_global_broader_masked'),
                           ('fraction_significant_broader_masked','n_significant_broader_masked'),
                           ('fraction_significant_narrower_masked','n_significant_narrower_masked')]:
        paper[count]=(paper[fraction]*paper.distances).round().astype(int)
    paper.to_csv(TAB/'l403a_all_distance_paper_statistics_summary.csv',index=False)
    top=pd.read_csv(TAB/'top_concordant_distance_breadth_changes.csv')[['distance']]
    details=[]
    for depth in [FOCAL_DEPTH,'Full QC']:
        z=results[results.depth.astype(str)==str(depth)].set_index('distance').loc[top.distance].reset_index()
        z=z[['distance','depth','vanilla_global_IQR_A','masked_global_IQR_A','global_IQR_ratio',
             'vanilla_median_seed_IQR_A','masked_median_seed_IQR_A','seed_IQR_ratio','seed_breadth_q',
             'trajectory_median_W1_A','W1_normalized_pooled_IQR']]
        details.append(z)
    pd.concat(details,ignore_index=True).to_csv(TAB/'top_concordant_distance_statistical_details.csv',index=False)
    audit={'shared_distance_columns_before_chain_safety_filter':1410,'analyzed_intrachain_distance_columns':len(cols),'excluded_raw_interchain_columns':1410-len(cols),'distance_types':pd.Series(['CA' if c.startswith('CA_') else 'shortest_heavy' for c in cols]).value_counts().to_dict(),'chain_safety_rule':'Kv2.1 raw interchain columns excluded because homotetramer chain labels can permute; only same-chain pairs retained','depths_in_seeds_per_protocol':DEPTHS,'focal_nominal_cohort':'first 20 ordered seeds × models 1–5 = 100 intended trajectories per protocol','focal_depth_in_seeds_per_protocol':FOCAL_DEPTH,'focal_qc_retained_trajectories':dict(zip(retention_summary.protocol,retention_summary.retained_trajectories.astype(int))),'representative_rule':'one structure per retained seed/model trajectory: latest recycle surviving final QC','ordering':'fixed nominal seed/model cohort; deterministic, not timestamped chronology; missing QC trajectories are not replaced','primary_inference':'Mann-Whitney comparison of within-seed IQR from one representative per trajectory; BH FDR within depth','overall_breadth_test':'Mann-Whitney on one median normalized intrachain-distance IQR per seed','sensitivity_tests':'trajectory-level Brown-Forsythe per distance, vanilla rarefaction to masked retained count, and 1000 random-seed saturation subsets per budget','separation':'Wasserstein distance between one representative per retained trajectory'}
    (TAB/'l403a_all_distance_sampling_run_summary.json').write_text(json.dumps(audit,indent=2)+'\n'); return {'results':results,'summary':summary,'stability':stable,'audit':audit}

if __name__=='__main__':
    r=run(); print(json.dumps(r['audit'],indent=2)); print(r['summary'].to_string(index=False)); print(r['stability'].to_string(index=False))
