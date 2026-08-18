"""Equal-depth sampling analysis for QC-passing Kv2.1 L403A ensembles."""
from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from shared.plotting import apply_kv21_style, KV21_PALETTE
from kv21.run_l403a_conformational_validation import load_inputs

OUT=ROOT/'kv21'/'dataExtra'/'conformation_analysis'/'early_sampling'
FIG=OUT/'figures'; TAB=OUT/'tables'; FIG.mkdir(parents=True,exist_ok=True); TAB.mkdir(parents=True,exist_ok=True)
DEPTHS=[250,500,1000,2000]
METRICS=['kink_angle_deg','whole_s6_rotation_vs_8SD3_deg','I401_azimuth_deg','I405_azimuth_deg']
ANGLES=set(METRICS[1:]); N_BOOT=1000; RNG=np.random.default_rng(1000)
LABELS={'kink_angle_deg':'PIP/S6 kink','whole_s6_rotation_vs_8SD3_deg':'Whole-S6 rotation','I401_azimuth_deg':'I401 azimuth','I405_azimuth_deg':'I405 azimuth'}

def wrap(x): return (np.asarray(x,float)+180)%360-180
def circ_center(x):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    return np.degrees(np.angle(np.mean(np.exp(1j*np.radians(x))))) if len(x) else np.nan
def dispersion(x,angular=False):
    x=np.asarray(x,float); x=x[np.isfinite(x)]
    if not len(x): return np.nan
    if angular: return float(np.median(np.abs(wrap(x-circ_center(x)))))
    return float(np.percentile(x,75)-np.percentile(x,25))
def savefig(fig,name):
    fig.savefig(FIG/f'{name}.png',dpi=300,bbox_inches='tight',facecolor='white')
    fig.savefig(FIG/f'{name}.pdf',bbox_inches='tight',facecolor='white'); plt.close(fig)

def ordered_l403a():
    exp,_,selected=load_inputs()
    d=selected[(selected.source_type=='prediction')&(selected.condition=='l403a')].copy()
    # One row per structure/subunit. This is a deterministic index-order proxy,
    # not a timestamp: seed, then AF model, then recycle.
    d['recycle_order']=pd.to_numeric(d.recycle,errors='coerce').fillna(999)
    structures=(d[['structure_id','protocol','seed','model_number','recycle_order','recycle_label','source_path']]
                .drop_duplicates('structure_id').sort_values(['protocol','seed','model_number','recycle_order','structure_id']))
    structures['sampling_index']=structures.groupby('protocol').cumcount()+1
    d=d.merge(structures[['structure_id','sampling_index']],on='structure_id',validate='many_to_one')
    return exp,d,structures

def depth_label(n,total): return 'Full QC' if n>=total else f'First {n}'

def bootstrap_ratio(a,b,metric):
    # Whole seeds are resampled; all structures/recycles/subunits in a seed stay together.
    aa=[g[metric].dropna().to_numpy(float) for _,g in a.groupby('seed')]
    bb=[g[metric].dropna().to_numpy(float) for _,g in b.groupby('seed')]
    vals=[]
    for _ in range(N_BOOT):
        av=np.concatenate([aa[i] for i in RNG.integers(0,len(aa),len(aa))])
        bv=np.concatenate([bb[i] for i in RNG.integers(0,len(bb),len(bb))])
        da=dispersion(av,metric in ANGLES); db=dispersion(bv,metric in ANGLES)
        vals.append(np.log2((db+1e-9)/(da+1e-9)))
    vals=np.asarray(vals)
    # Plus-one correction avoids reporting an impossible exact zero from a
    # finite Monte Carlo bootstrap (minimum two-sided value is ~2/(B+1)).
    p=min(1.0,2*(min((vals<=0).sum(),(vals>=0).sum())+1)/(len(vals)+1))
    return *np.percentile(vals,[2.5,97.5]),p

def bh_fdr(p):
    p=np.asarray(p,float); n=len(p); order=np.argsort(p); ranked=p[order]
    q=np.minimum.accumulate((ranked*n/np.arange(1,n+1))[::-1])[::-1]
    out=np.empty(n); out[order]=np.minimum(q,1); return out

def dispersion_analysis(d):
    rows=[]
    totals=d.groupby('protocol').structure_id.nunique().to_dict()
    for depth in DEPTHS+[max(totals.values())]:
      for sub in 'ABCD':
       for metric in METRICS:
        groups={p:d[(d.protocol==p)&(d.canonical_subunit==sub)&(d.sampling_index<=min(depth,totals[p]))] for p in ['vanilla','masked']}
        va=dispersion(groups['vanilla'][metric],metric in ANGLES); ma=dispersion(groups['masked'][metric],metric in ANGLES)
        lo,hi,p=bootstrap_ratio(groups['vanilla'],groups['masked'],metric)
        rows.append(dict(depth='Full QC' if depth>=max(totals.values()) else depth,canonical_subunit=sub,metric=metric,
            dispersion_definition='circular median absolute deviation (deg)' if metric in ANGLES else 'IQR (physical units)',
            vanilla_dispersion=va,masked_dispersion=ma,masked_vanilla_ratio=ma/(va+1e-9),log2_ratio=np.log2((ma+1e-9)/(va+1e-9)),
            log2_ratio_ci_low=lo,log2_ratio_ci_high=hi,masked_broader=ma>va,
            bootstrap_p_two_sided=p,
            vanilla_structures=groups['vanilla'].structure_id.nunique(),masked_structures=groups['masked'].structure_id.nunique(),
            vanilla_seeds=groups['vanilla'].seed.nunique(),masked_seeds=groups['masked'].seed.nunique()))
    out=pd.DataFrame(rows)
    out['bootstrap_q_within_depth']=out.groupby('depth',sort=False).bootstrap_p_two_sided.transform(bh_fdr)
    out.to_csv(TAB/'l403a_sampling_depth_dispersion.csv',index=False); return out

def target_scores(exp,d):
    targets=exp.set_index('canonical_subunit')
    z=d.copy(); errcols=[]
    for metric in METRICS:
        target=z.canonical_subunit.map(targets[f'8SDA__{metric}'])
        e=np.abs(wrap(z[metric]-target)) if metric in ANGLES else (z[metric]-target).abs()
        col=f'error__{metric}'; z[col]=e; errcols.append(col)
    # Rank each physical component in the pooled full-QC population; lower is closer.
    for col in errcols: z[col+'_pct']=z.groupby('canonical_subunit')[col].rank(pct=True)
    bd=z[z.canonical_subunit.isin(['B','D'])]
    score=(bd.groupby(['structure_id','protocol','seed','sampling_index'],as_index=False)
           [[c+'_pct' for c in errcols]].mean())
    score['experimental_like_score']=score[[c+'_pct' for c in errcols]].mean(axis=1)
    score['experimental_like_top5pct']=score.experimental_like_score.le(score.experimental_like_score.quantile(.05))
    score.to_csv(TAB/'l403a_structure_experimental_like_scores.csv',index=False)
    return score

def bootstrap_hit_difference(v,m):
    vv=v.groupby('seed').experimental_like_top5pct.agg(['sum','count']).to_numpy(float)
    mm=m.groupby('seed').experimental_like_top5pct.agg(['sum','count']).to_numpy(float)
    vals=[]
    for _ in range(N_BOOT):
        a=vv[RNG.integers(0,len(vv),len(vv))].sum(axis=0)
        b=mm[RNG.integers(0,len(mm),len(mm))].sum(axis=0)
        vals.append(b[0]/b[1]-a[0]/a[1])
    return np.percentile(vals,[2.5,97.5])

def enrichment_analysis(score):
    totals=score.groupby('protocol').size().to_dict(); rows=[]
    for depth in DEPTHS+[max(totals.values())]:
        g={p:score[(score.protocol==p)&(score.sampling_index<=min(depth,totals[p]))] for p in ['vanilla','masked']}
        lo,hi=bootstrap_hit_difference(g['vanilla'],g['masked'])
        rows.append(dict(depth='Full QC' if depth>=max(totals.values()) else depth,
            vanilla_hit_fraction=g['vanilla'].experimental_like_top5pct.mean(),masked_hit_fraction=g['masked'].experimental_like_top5pct.mean(),
            hit_fraction_difference=g['masked'].experimental_like_top5pct.mean()-g['vanilla'].experimental_like_top5pct.mean(),
            difference_ci_low=lo,difference_ci_high=hi,
            vanilla_best_score=g['vanilla'].experimental_like_score.min(),masked_best_score=g['masked'].experimental_like_score.min(),
            vanilla_hits=int(g['vanilla'].experimental_like_top5pct.sum()),masked_hits=int(g['masked'].experimental_like_top5pct.sum()),
            vanilla_seeds=g['vanilla'].seed.nunique(),masked_seeds=g['masked'].seed.nunique()))
    out=pd.DataFrame(rows); out.to_csv(TAB/'l403a_sampling_depth_experimental_like_enrichment.csv',index=False); return out

def figures(disp,enrich):
    apply_kv21_style(); order=[250,500,1000,2000,'Full QC']; x=np.arange(len(order))
    fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True)
    for ax,(metric,g) in zip(axs.flat,disp.groupby('metric',sort=False)):
        for sub,ls in zip('ABCD',['-','--','-.',':']):
            z=g[g.canonical_subunit==sub].set_index('depth').reindex(order)
            ax.plot(x,z.log2_ratio,marker='o',label=f'Subunit {sub}',linestyle=ls,color=KV21_PALETTE['L403A_HM'])
            ax.fill_between(x,z.log2_ratio_ci_low,z.log2_ratio_ci_high,color=KV21_PALETTE['L403A_VAN'],alpha=.12)
        ax.axhline(0,color='.25',lw=.8); ax.set_title(LABELS[metric]); ax.set_ylabel('log2(masked/vanilla dispersion)')
        ax.set_xticks(x,[str(v) for v in order],rotation=25)
    axs[0,1].legend(ncol=2,fontsize=8); fig.suptitle('L403A equal-depth sampling: does masking broaden S6 geometry earlier?'); fig.tight_layout(); savefig(fig,'l403a_sampling_depth_dispersion')
    z=enrich.set_index('depth').reindex(order); fig,ax=plt.subplots(figsize=(8,4.8))
    ax.plot(x,z.vanilla_hit_fraction,marker='o',color=KV21_PALETTE['L403A_VAN'],label='Vanilla')
    ax.plot(x,z.masked_hit_fraction,marker='o',color=KV21_PALETTE['L403A_HM'],label='Masked')
    ax.set_xticks(x,[str(v) for v in order]); ax.set_xlabel('Retained-structure depth (deterministic order)'); ax.set_ylabel('Fraction in pooled top 5%\n8SDA-like B/D S6 score'); ax.legend(); ax.set_title('Does masking find experimentally shifted S6 geometry earlier?'); fig.tight_layout(); savefig(fig,'l403a_sampling_depth_experimental_like_enrichment')

def run():
    exp,d,structures=ordered_l403a(); disp=dispersion_analysis(d); score=target_scores(exp,d); enrich=enrichment_analysis(score); figures(disp,enrich)
    audit=(structures.groupby('protocol').agg(qc_structures=('structure_id','nunique'),seeds=('seed','nunique'),min_seed=('seed','min'),max_seed=('seed','max')).reset_index())
    audit['order_definition']='seed, model_number, recycle; deterministic proxy, not timestamped chronology'; audit.to_csv(TAB/'l403a_sampling_depth_audit.csv',index=False)
    summary={'bootstrap_replicates':N_BOOT,'depths':DEPTHS+['Full QC'],'primary_depth':1000,'ordering':'seed/model/recycle; not timestamped chronology','experimental_like_definition':'lowest 5% pooled full-QC mean percentile error across kink, S6 rotation, I401 and I405 azimuth in B/D'}
    (TAB/'l403a_sampling_depth_run_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); return {'dispersion':disp,'enrichment':enrich,'audit':audit,'summary':summary}

if __name__=='__main__':
    r=run(); print(r['audit'].to_string(index=False)); print(r['enrichment'].to_string(index=False))
