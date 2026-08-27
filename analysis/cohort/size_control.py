#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
size_control.py - separate the AM-only comparison from any confound with sample size.

Four training sets are compared on identical held-out altermagnetic rows:
  full        the whole training fold          (large, mixed composition)
  am          training fold INTERSECT ALTERMAGNET   (small, altermagnetic composition)
  sub_row     a random draw from the training fold with the same row count as am
              (small, mixed composition)
  sub_parent  whole parents drawn at random until the row count is closest to am
              (small size and reduced parent diversity, mixed composition)

  am vs sub_*   -> the composition effect at fixed size
  full vs sub_* -> the size effect at fixed composition

Each condition is fitted both with the full 52 features and with p_metric withheld, so the
drop-column loss for MSBI is obtained alongside.
"""
import os, argparse
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool

P = dict(learning_rate=0.02096418735206606, max_depth=7, min_child_weight=1,
         subsample=0.755991902684371, colsample_bytree=0.7282976917303419,
         reg_lambda=0.4606236274881072, reg_alpha=0.05239830776073256,
         n_estimators=300, objective='reg:squarederror', tree_method='hist',
         random_state=42, n_jobs=1)

FO = ['avg_bond_length','max_bond_length','min_bond_length','std_bond_length',
'center_max_angle','center_min_angle','center_avg_angle','center_std_angle',
'nonmag_max_angle','nonmag_min_angle','nonmag_std_angle','labelled_1st',
'labelled_2nd','labelled_3rd','global_1st','global_2nd','global_3rd',
'avg_long_axis','avg_short_axis','avg_axis_ratio','avg_s','avg_delta',
'motif0_nonmag_count','magnetic_atomic_number','magnetic_electronegativity',
'nonmagnetic_atomic_number','nonmagnetic_electronegativity',
'hungarian_rotation_angle_deg','dimension','avg_motif_measure',
'unit_cell_volume','packing_fraction','p_metric','p_metric_std','d_orb_e',
'p_orb_e_non','d_lone_pair','proxy_M_magnet','delta_chi','abs_delta_chi',
'delta_Z','abs_delta_Z','pd_ratio','ax_eq_gap','bond_range','bond_cv',
'center_angle_spread','nonmag_angle_spread','delta_chi_times_axeq',
'd_global_local_1st','d_global_local_2nd','d_global_local_3rd']

SEEDS = [11,22,33,44,55,66,77,88,99,111,123,234,345,456,567,678,789,890,901,1012]
COHORTS = ['full','am','sub_row','sub_parent']
DF=None; Y=None; SSE=None; G=None; ISAM=None

def gsplit(seed,k=5):
    idx=np.arange(len(DF)); g=G[idx]; ug=np.array(sorted(set(g)))
    rng=np.random.default_rng(seed); rng.shuffle(ug)
    f={x:i%k for i,x in enumerate(ug)}; fa=np.array([f[x] for x in g])
    return [(idx[fa!=i], idx[fa==i]) for i in range(k)]

def r2(y,p): return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()

def train_idx(cohort, tr, seed, fold):
    if cohort=='full': return tr
    tr_am = tr[ISAM[tr]]
    if cohort=='am': return tr_am
    n = len(tr_am); rng = np.random.default_rng(seed*1000+fold)
    if cohort=='sub_row':
        return np.sort(rng.choice(tr, size=n, replace=False))
    ug = np.array(sorted(set(G[tr]))); rng.shuffle(ug)
    cnt = pd.Series(G[tr]).value_counts()
    tot=0; keep=[]
    for p_ in ug:
        keep.append(p_); tot += int(cnt[p_])
        if tot >= n: break
    if abs(tot-n) > abs(tot-int(cnt[keep[-1]])-n) and len(keep)>1:
        keep.pop()                                  # keep whichever total is closer to n
    m = np.isin(G[tr], keep)
    return tr[m]

def job(a):
    cohort, feat, seed, fold = a
    tr, te = gsplit(seed)[fold]
    ti = train_idx(cohort, tr, seed, fold)
    cols = FO if feat is None else [f for f in FO if f != feat]
    m = xgb.XGBRegressor(**P).fit(DF[cols].values[ti], Y[ti])
    te_am = te[ISAM[te]]
    return dict(cohort=cohort, feature=feat or '_none_', seed=seed, fold=fold,
                n_tr=len(ti), n_par=len(set(G[ti])), frac_am=float(ISAM[ti].mean()),
                idx=te_am.tolist(), pred=m.predict(DF[cols].values[te_am]).tolist())

def pool_oof(recs, keys):
    df=pd.DataFrame(recs); out=[]
    for k,g in df.groupby(keys):
        idx=np.concatenate([np.asarray(x,dtype=int) for x in g['idx']])
        pr=np.concatenate([np.asarray(x,dtype=float) for x in g['pred']])
        o=np.argsort(idx); idx,pr=idx[o],pr[o]
        d=dict(zip(keys, k if isinstance(k,tuple) else (k,)))
        d.update(r2_log=r2(Y[idx],pr), r2_eV=r2(SSE[idx],np.expm1(pr)),
                 n_tr=float(g.n_tr.mean()), n_par=float(g.n_par.mean()),
                 frac_am=float(g.frac_am.mean()))
        out.append(d)
    return pd.DataFrame(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default='fin_data.csv')
    ap.add_argument('--sym', default='raw/magnetic_symmetry_all.csv')
    ap.add_argument('--outdir', default='size_out')
    ap.add_argument('--workers', type=int, default=max(1,(os.cpu_count() or 2)-2))
    a=ap.parse_args()
    global DF,Y,SSE,G,ISAM
    DF=pd.read_csv(a.data)
    sym=pd.read_csv(a.sym)[['filename','verdict']]
    DF['fkey']=DF.filename.str.replace(r'^POSCAR_','',regex=True)
    DF['verdict']=DF.fkey.map(dict(zip(sym.filename,sym.verdict)))
    assert DF.verdict.isna().sum()==0
    pat=DF.filename.str.extract(r'^POSCAR_([A-Za-z0-9]+)_([A-Za-z0-9]+?)(?:_(?:st|x|y|z)\d+)?$')
    DF['parent']=pat[0]+'_'+pat[1]
    SSE=DF.sse.values.astype(float); Y=np.log1p(SSE); G=DF.parent.values
    ISAM=(DF.verdict=='ALTERMAGNET').values
    os.makedirs(a.outdir, exist_ok=True)

    jobs=[(c,f,s,k) for c in COHORTS for f in (None,'p_metric') for s in SEEDS for k in range(5)]
    print(f'{len(jobs)} fits, workers={a.workers}', flush=True)
    with Pool(a.workers) as pool:
        recs=pool.map(job, jobs, chunksize=2)
    res=pool_oof(recs,['cohort','feature','seed'])
    res.to_csv(f'{a.outdir}/size_control_seeds.csv', index=False)

    base=res[res.feature=='_none_']
    print('\n=== performance on held-out ALTERMAGNET rows (20 seeds) ===', flush=True)
    print(f"  {'condition':<11}{'rows':>7}{'parent':>8}{'AMfrac':>8}   {'R2_log':>16}   {'R2_eV':>16}", flush=True)
    for c in COHORTS:
        g=base[base.cohort==c]
        print(f'  {c:<11}{g.n_tr.mean():7.0f}{g.n_par.mean():8.1f}{g.frac_am.mean():8.2f}   '
              f'{g.r2_log.mean():.4f} ± {g.r2_log.std(ddof=1):.4f}   '
              f'{g.r2_eV.mean():.4f} ± {g.r2_eV.std(ddof=1):.4f}', flush=True)

    piv=base.pivot(index='seed',columns='cohort',values='r2_log')
    print('\n=== paired differences (20 seeds, R2_log) ===', flush=True)
    for x,y,lab in [('am','sub_row','composition effect (size fixed, row-matched)'),
                    ('am','sub_parent','composition effect (size fixed, parent-matched)'),
                    ('sub_row','full','size effect (composition fixed, row-matched)'),
                    ('sub_parent','full','size effect (composition fixed, parent-matched)'),
                    ('am','full','the restriction as a whole (AM-only - full)')]:
        d=piv[x]-piv[y]
        print(f'  {lab:<48} {x:>10} - {y:<10} {d.mean():+.4f} +- {d.std(ddof=1):.4f}   negative in {int((d<0).sum())}/20', flush=True)

    print('\n=== drop-column loss for MSBI (p_metric), per condition ===', flush=True)
    bl=base.set_index(['cohort','seed']).r2_log
    dc=res[res.feature=='p_metric'].copy()
    dc['d']=dc.r2_log.values - bl.loc[list(zip(dc.cohort,dc.seed))].values
    for c in COHORTS:
        g=dc[dc.cohort==c]; sd=base[base.cohort==c].r2_log.std(ddof=1)
        print(f'  {c:<11} dR2_log {g.d.mean():+.4f} ± {g.d.std(ddof=1):.4f}   '
              f'|d|/sd {abs(g.d.mean())/sd:5.1f}   negative in {int((g.d<0).sum())}/20', flush=True)
    print('\ndone', flush=True)

if __name__=='__main__':
    main()
