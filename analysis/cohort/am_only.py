#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
am_only.py - quantitative answer to "why not refit on the altermagnets alone?"

The comparison is made on *identical held-out altermagnetic rows*.
  - Folds are drawn on the full cohort (3,845 rows) with GroupKFold(5) by parent, 20 seeds.
  - Evaluation set = (held-out fold) INTERSECT (ALTERMAGNET), so it is identical for both models.
  - Only the training set differs:
        full : the whole training fold (altermagnets + conventional AFM + FiM candidates)
        am   : training fold INTERSECT ALTERMAGNET

dR2 = R2(am) - R2(full) therefore answers directly whether removing the symmetry
controls from training improves prediction for altermagnets.
"""
import os, sys, argparse, json
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

DF = None; Y = None; SSE = None; G = None; ISAM = None

def gsplit(seed, k=5):
    idx = np.arange(len(DF)); g = G[idx]; ug = np.array(sorted(set(g)))
    rng = np.random.default_rng(seed); rng.shuffle(ug)
    f = {x: i % k for i, x in enumerate(ug)}
    fa = np.array([f[x] for x in g])
    return [(idx[fa != i], idx[fa == i]) for i in range(k)]

def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def _fit(cols, tr):
    return xgb.XGBRegressor(**P).fit(DF[cols].values[tr], Y[tr])

def job(a):
    """cohort x (feature|None) x seed x fold -> predictions for the held-out altermagnetic rows."""
    cohort, feat, seed, fold = a
    tr, te = gsplit(seed)[fold]
    if cohort == 'am':
        tr = tr[ISAM[tr]]
    cols = FO if feat is None else [f for f in FO if f != feat]
    m = _fit(cols, tr)
    te_am = te[ISAM[te]]                      # evaluation is always on altermagnetic rows only
    out = dict(cohort=cohort, feature=feat or '_none_', seed=seed, fold=fold,
               idx=te_am.tolist(), pred=m.predict(DF[cols].values[te_am]).tolist())
    if feat is None:                          # also record performance on the control rows, for reference
        te_ct = te[~ISAM[te]]
        out['idx_ctrl'] = te_ct.tolist()
        out['pred_ctrl'] = m.predict(DF[cols].values[te_ct]).tolist()
    return out

def job_shap(a):
    cohort, seed, fold = a
    tr, te = gsplit(seed)[fold]
    if cohort == 'am':
        tr = tr[ISAM[tr]]
    m = _fit(FO, tr)
    te_am = te[ISAM[te]]
    c = m.get_booster().predict(xgb.DMatrix(DF[FO].values[te_am], feature_names=FO),
                                pred_contribs=True)[:, :len(FO)]
    return dict(cohort=cohort, seed=seed, fold=fold,
                n=len(te_am), s=np.abs(c).sum(axis=0).tolist())

def pool_oof(recs, keys):
    df = pd.DataFrame(recs); out = []
    for k, g in df.groupby(keys):
        idx = np.concatenate([np.asarray(x, dtype=int) for x in g['idx']])
        pr  = np.concatenate([np.asarray(x, dtype=float) for x in g['pred']])
        o = np.argsort(idx); idx, pr = idx[o], pr[o]
        d = dict(zip(keys, k if isinstance(k, tuple) else (k,)))
        d.update(r2_log=r2(Y[idx], pr), r2_eV=r2(SSE[idx], np.expm1(pr)), n=len(idx))
        out.append(d)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='fin_data.csv')
    ap.add_argument('--sym',  default='raw/magnetic_symmetry_all.csv')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument('--skip-dropcol', action='store_true')
    a = ap.parse_args()

    global DF, Y, SSE, G, ISAM
    DF = pd.read_csv(a.data)
    sym = pd.read_csv(a.sym)[['filename', 'verdict']]
    DF['fkey'] = DF.filename.str.replace(r'^POSCAR_', '', regex=True)
    m = dict(zip(sym.filename, sym.verdict))
    DF['verdict'] = DF.fkey.map(m)
    assert DF.verdict.isna().sum() == 0, f'{DF.verdict.isna().sum()} rows failed to match a symmetry label'
    pat = DF.filename.str.extract(r'^POSCAR_([A-Za-z0-9]+)_([A-Za-z0-9]+?)(?:_(?:st|x|y|z)\d+)?$')
    DF['parent'] = pat[0] + '_' + pat[1]
    SSE = DF.sse.values.astype(float); Y = np.log1p(SSE); G = DF.parent.values
    ISAM = (DF.verdict == 'ALTERMAGNET').values
    os.makedirs(a.outdir, exist_ok=True)

    print(f'rows={len(DF)}  parents={DF.parent.nunique()}  altermagnet={ISAM.sum()}', flush=True)
    print(DF.groupby('verdict').sse.agg(['size','median','max']).to_string(), flush=True)
    # number of parents that survive when only altermagnetic rows are kept
    print(f"parents containing altermagnetic rows = {DF[ISAM].parent.nunique()} / {DF.parent.nunique()}", flush=True)

    with Pool(a.workers) as pool:
        # --- 1. full-52 baseline for both cohorts ---
        jobs = [(c, None, s, f) for c in ('full','am') for s in SEEDS for f in range(5)]
        print(f'[1/3] baseline: {len(jobs)} fits', flush=True)
        recs = pool.map(job, jobs, chunksize=2)
        base = pool_oof(recs, ['cohort','seed'])
        base.to_csv(f'{a.outdir}/cohort_baseline_seeds.csv', index=False)

        piv = base.pivot(index='seed', columns='cohort')
        d_log = (piv[('r2_log','am')] - piv[('r2_log','full')])
        d_eV  = (piv[('r2_eV','am')]  - piv[('r2_eV','full')])
        print('\n=== performance on held-out ALTERMAGNET rows (20 seeds) ===', flush=True)
        for c in ('full','am'):
            g = base[base.cohort == c]
            print(f'  {c:<5} R2_log {g.r2_log.mean():.4f} ± {g.r2_log.std(ddof=1):.4f}   '
                  f'R2_eV {g.r2_eV.mean():.4f} ± {g.r2_eV.std(ddof=1):.4f}   n={g.n.iloc[0]}', flush=True)
        print(f'  dR2_log (am - full) = {d_log.mean():+.4f} ± {d_log.std(ddof=1):.4f}   '
              f'negative in {int((d_log<0).sum())}/20', flush=True)
        print(f'  dR2_eV  (am - full) = {d_eV.mean():+.4f} ± {d_eV.std(ddof=1):.4f}   '
              f'negative in {int((d_eV<0).sum())}/20', flush=True)

        # performance of the full model on the control rows (for reference)
        cr = [r for r in recs if r['cohort']=='full']
        rows=[]
        for s in SEEDS:
            gi = np.concatenate([np.asarray(r['idx_ctrl'],dtype=int) for r in cr if r['seed']==s])
            gp = np.concatenate([np.asarray(r['pred_ctrl'],dtype=float) for r in cr if r['seed']==s])
            rows.append(dict(seed=s, r2_log=r2(Y[gi],gp), r2_eV=r2(SSE[gi],np.expm1(gp)),
                             mae_eV=np.abs(SSE[gi]-np.expm1(gp)).mean(), n=len(gi)))
        ctrl=pd.DataFrame(rows); ctrl.to_csv(f'{a.outdir}/cohort_control_rows.csv', index=False)
        print(f'\n  [reference] full model on non-altermagnetic held-out rows: '
              f'MAE {ctrl.mae_eV.mean()*1000:.1f} meV, n={ctrl.n.iloc[0]}', flush=True)

        # --- 2. SHAP ranking ---
        jobs = [(c, s, f) for c in ('full','am') for s in SEEDS for f in range(5)]
        print(f'\n[2/3] SHAP: {len(jobs)} fits', flush=True)
        sr = pd.DataFrame(pool.map(job_shap, jobs, chunksize=2))
        out=[]
        for c in ('full','am'):
            g = sr[sr.cohort==c]
            tot = np.vstack(g.s.values).sum(axis=0); n = g.n.sum()
            sh = tot/n; sh = sh/sh.sum()
            for f_, v in zip(FO, sh): out.append(dict(cohort=c, feature=f_, share=v))
        sh = pd.DataFrame(out)
        sh['rank'] = sh.groupby('cohort')['share'].rank(ascending=False).astype(int)
        sh.sort_values(['cohort','rank']).to_csv(f'{a.outdir}/cohort_shap.csv', index=False)
        print('\n=== SHAP ranking (on held-out altermagnetic rows) ===', flush=True)
        for c in ('full','am'):
            t = sh[sh.cohort==c].nsmallest(6,'rank')
            print(f'  {c:<5} ' + '  '.join(f'{r.feature}({r.share*100:.1f}%)' for r in t.itertuples()), flush=True)

        # --- 3. drop-column over all 52 features, both cohorts ---
        if not a.skip_dropcol:
            jobs = [(c, f_, s, k) for c in ('full','am') for f_ in FO for s in SEEDS for k in range(5)]
            print(f'\n[3/3] drop-column: {len(jobs)} fits', flush=True)
            recs = pool.map(job, jobs, chunksize=4)
            dc = pool_oof(recs, ['cohort','feature','seed'])
            bl = base.set_index(['cohort','seed'])
            dc['d_log'] = dc.r2_log.values - bl.loc[list(zip(dc.cohort,dc.seed)),'r2_log'].values
            dc['d_eV']  = dc.r2_eV.values  - bl.loc[list(zip(dc.cohort,dc.seed)),'r2_eV'].values
            agg = dc.groupby(['cohort','feature']).agg(
                d_log_mean=('d_log','mean'), d_log_sd=('d_log','std'),
                d_eV_mean=('d_eV','mean'), neg=('d_log', lambda x:int((x<0).sum()))).reset_index()
            agg.sort_values(['cohort','d_log_mean']).to_csv(f'{a.outdir}/cohort_dropcol.csv', index=False)
            print('\n=== five largest drop-column losses, per cohort ===', flush=True)
            for c in ('full','am'):
                sd = base[base.cohort==c].r2_log.std(ddof=1)
                t = agg[agg.cohort==c].nsmallest(5,'d_log_mean')
                print(f'  --- {c} (run-to-run sd = {sd:.4f}) ---', flush=True)
                for r in t.itertuples():
                    print(f'    {r.feature:<32} dR2_log {r.d_log_mean:+.4f} ± {r.d_log_sd:.4f}  '
                          f'|d|/sd {abs(r.d_log_mean)/sd:5.1f}  negative in {r.neg}/20', flush=True)
                nx = agg[(agg.cohort==c) & (agg.d_log_mean < -2*sd)]
                print(f'    -> features exceeding twice the run-to-run sd: {len(nx)}', flush=True)
    print('\ndone', flush=True)

if __name__ == '__main__':
    main()
