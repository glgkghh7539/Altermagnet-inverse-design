#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stability_selection_100_parallel.py
===================================
Nested grouped CV + stability selection, 20 seeds x 5 outer folds = 100 runs.

Outer folds are independent, so this is embarrassingly parallel: each worker
handles whole (seed, fold) jobs. Set --workers to the number of PHYSICAL cores.

Usage on TGM:
    chmod +x stability_selection_100_parallel.py
    ./stability_selection_100_parallel.py fin_data.csv --workers 20

Runtime: ~1.5-2 h single-core; ~5-10 min on 20 cores.

IMPORTANT: XGBoost must run single-threaded inside each worker (n_jobs=1 below),
otherwise workers oversubscribe the cores and everything slows down. The same
applies to BLAS -- the env vars are set before numpy is imported.

Outputs
-------
stability_100.csv   per-feature selection frequency out of 100
nested_folds_100.csv per-fold nested outer R2 (the honest performance estimate
                     for the SELECTION PROCEDURE, not for a hand-picked subset)
"""
import os, sys, argparse
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"

import numpy as np, pandas as pd, xgboost as xgb, warnings, collections, json
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
P=dict(learning_rate=0.02096418735206606,max_depth=7,min_child_weight=1,
       subsample=0.755991902684371,colsample_bytree=0.7282976917303419,
       reg_lambda=0.4606236274881072,reg_alpha=0.05239830776073256,
       n_estimators=300,objective='reg:squarederror',tree_method='hist',random_state=42,n_jobs=1)
FO=['avg_bond_length','max_bond_length','min_bond_length','std_bond_length','center_max_angle','center_min_angle','center_avg_angle','center_std_angle','nonmag_max_angle','nonmag_min_angle','nonmag_std_angle','labelled_1st','labelled_2nd','labelled_3rd','global_1st','global_2nd','global_3rd','avg_long_axis','avg_short_axis','avg_axis_ratio','avg_s','avg_delta','motif0_nonmag_count','magnetic_atomic_number','magnetic_electronegativity','nonmagnetic_atomic_number','nonmagnetic_electronegativity','hungarian_rotation_angle_deg','dimension','avg_motif_measure','unit_cell_volume','packing_fraction','p_metric','p_metric_std','d_orb_e','p_orb_e_non','d_lone_pair','proxy_M_magnet','delta_chi','abs_delta_chi','delta_Z','abs_delta_Z','pd_ratio','ax_eq_gap','bond_range','bond_cv','center_angle_spread','nonmag_angle_spread','delta_chi_times_axeq','d_global_local_1st','d_global_local_2nd','d_global_local_3rd']
import sys
_ap=argparse.ArgumentParser()
_ap.add_argument('data',nargs='?',default='fin_data.csv')
_ap.add_argument('--workers',type=int,default=max(1,(os.cpu_count() or 1)))
_ARGS=_ap.parse_args()
df=pd.read_csv(_ARGS.data)
pat=df.filename.str.extract(r'^POSCAR_([A-Za-z0-9]+)_([A-Za-z0-9]+?)(?:_(?:st|x|y|z)\d+)?$')
df['parent']=pat[0]+'_'+pat[1]
Y=np.log1p(df.sse.values); G=df.parent.values
SIZES=[52,42,34,28,22,18,14,11,9,7,5,3]

def gsplit(idx,seed,k):
    g=G[idx]; ug=np.array(sorted(set(g))); rng=np.random.default_rng(seed); rng.shuffle(ug)
    f={x:i%k for i,x in enumerate(ug)}; fa=np.array([f[x] for x in g])
    return [(idx[fa!=i],idx[fa==i]) for i in range(k)]

def cv_r2(cols,folds):
    p={}; 
    for tr,te in folds:
        m=xgb.XGBRegressor(**P).fit(df[cols].values[tr],Y[tr])
        for j,v in zip(te,m.predict(df[cols].values[te])): p[j]=v
    ii=np.array(sorted(p)); pr=np.array([p[j] for j in ii]); yy=Y[ii]
    return 1-((yy-pr)**2).sum()/((yy-yy.mean())**2).sum()


SEEDS=[11,22,33,44,55,66,77,88,99,111,123,234,345,456,567,678,789,890,901,1012]

def one_job(job):
    """Run a single (seed, outer-fold): inner RFE -> 1-SE pick -> outer score."""
    seed, oi = job
    folds = gsplit(np.arange(len(df)), seed, 5)
    tr_idx, te_idx = folds[oi]
    inner = gsplit(tr_idx, seed+7, 3)
    cols = list(FO); curve = []
    for s in SIZES:
        if s < len(cols):
            m = xgb.XGBRegressor(**P).fit(df[cols].values[tr_idx], Y[tr_idx])
            imp = pd.Series(m.feature_importances_, index=cols).sort_values(ascending=False)
            cols = list(imp.index[:s])
        curve.append((len(cols), cv_r2(cols, inner), list(cols)))
    best = max(c[1] for c in curve)
    se = np.std([c[1] for c in curve])/np.sqrt(len(curve))
    pick = min([c for c in curve if c[1] >= best-se], key=lambda c: c[0])   # 1-SE rule
    chosen = pick[2]
    yt = Y[te_idx]
    pe = xgb.XGBRegressor(**P).fit(df[chosen].values[tr_idx], Y[tr_idx]).predict(df[chosen].values[te_idx])
    pf = xgb.XGBRegressor(**P).fit(df[FO].values[tr_idx],   Y[tr_idx]).predict(df[FO].values[te_idx])
    r = lambda p: 1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum()
    return dict(seed=seed, fold=oi, n=len(chosen), inner=pick[1],
                outer=r(pe), full=r(pf), chosen=chosen)

if __name__ == '__main__':
    import collections
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = [(s, f) for s in SEEDS for f in range(5)]
    W = max(1, min(_ARGS.workers, len(jobs)))
    print(f"[run] {len(jobs)} outer folds on {W} worker(s)", flush=True)
    if W == 1:
        results = [one_job(j) for j in jobs]
    else:
        results = []
        with ProcessPoolExecutor(max_workers=W) as ex:
            futs = {ex.submit(one_job, j): j for j in jobs}
            for k, fu in enumerate(as_completed(futs), 1):
                rr = fu.result(); results.append(rr)
                print(f"  [{k:3d}/{len(jobs)}] seed{rr['seed']} fold{rr['fold']}: "
                      f"{rr['n']:2d} feats | outer={rr['outer']:.4f} full52={rr['full']:.4f}", flush=True)

    sel = collections.Counter()
    for rr in results: sel.update(rr['chosen'])
    n = len(results)
    out = np.array([rr['outer'] for rr in results])
    ful = np.array([rr['full']  for rr in results])
    siz = np.array([rr['n']     for rr in results])
    print(f"\n=== NESTED result over {n} outer folds ===")
    print(f"  selected-subset outer R2 = {out.mean():.4f} +- {out.std():.4f}")
    print(f"  full-52         outer R2 = {ful.mean():.4f} +- {ful.std():.4f}")
    print(f"  subset size: median={int(np.median(siz))} range={siz.min()}-{siz.max()}")
    print("\n=== STABILITY SELECTION frequency ===")
    for f, c in sel.most_common():
        print(f"  {f:32s} {c:3d}/{n}")
    pd.DataFrame([{k: v for k, v in r.items() if k != 'chosen'} for r in results]
                 ).to_csv('nested_folds_100.csv', index=False)
    pd.Series(sel).reindex(FO).fillna(0).astype(int).sort_values(ascending=False
                 ).to_csv('stability_100.csv', header=['selected_out_of_%d' % n])
    print("\n[saved] stability_100.csv, nested_folds_100.csv")
