#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ablation_grouped.py - response to reviewer Point 3 (npj Comput. Mater.)

The reviewer asks for:
  "compare a composition-only model, a structure-only model, the proposed
   three-feature model, the full model without MSBI, and the full 52-feature
   model under the same grouped splits. Drop-column or grouped permutation
   importance would be more informative here than SHAP alone. The p/d ratio,
   n_p, and n_d are algebraically related, so their separate SHAP shares should
   not be given a direct physical interpretation without treating them as a
   correlated group."

The protocol matches stability_selection_100_parallel.py:
  - GroupKFold(5) by parent, groups shuffled per seed: 20 seeds x 5 folds = 100 splits
  - fixed hyperparameters (no tuning, so that the model configurations remain comparable)
  - target log1p(sse)

R2 is reported both in log1p space and in eV space (after the expm1 inverse transform),
because the manuscript headline R2 = 0.6951 is the eV figure and the metric space must be stated.

Outputs (--outdir):
  ablation_folds.csv    R2 per (config, seed, fold) - the raw fold-to-fold variation
  ablation_summary.csv  pooled-OOF R2 mean and sd per config, and dR2 against the full model
  dropcol_single.csv    drop-column dR2 for each of the 52 features individually
  perm_grouped.csv      grouped permutation importance (no refit; the group is shuffled as one block)
"""
import os, sys, argparse, itertools, json
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from multiprocessing import Pool

# the same hyperparameters as stability_selection_100_parallel.py
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

# ---- feature group definitions ------------------------------------------
# composition: element and electron-count descriptors with no geometric dependence (13 of them)
COMPOSITION = ['magnetic_atomic_number','magnetic_electronegativity',
 'nonmagnetic_atomic_number','nonmagnetic_electronegativity','d_orb_e',
 'p_orb_e_non','d_lone_pair','proxy_M_magnet','delta_chi','abs_delta_chi',
 'delta_Z','abs_delta_Z','pd_ratio']
STRUCTURE = [f for f in FO if f not in COMPOSITION]          # 39 features
MSBI_FAMILY = ['p_metric','p_metric_std','hungarian_rotation_angle_deg']
MPF = ['packing_fraction']
# the reviewer notes that p/d, n_p and n_d are algebraically related and must be treated as a correlated group
PD_FAMILY = ['pd_ratio','p_orb_e_non','d_orb_e']
THREE = ['p_metric','packing_fraction','pd_ratio']

GROUPS = {'MSBI_family': MSBI_FAMILY, 'MPF': MPF, 'pd_family': PD_FAMILY,
          'composition': COMPOSITION, 'structure': STRUCTURE}

def build_configs():
    c = {}
    c['full52']            = list(FO)
    c['composition_only']  = list(COMPOSITION)
    c['structure_only']    = list(STRUCTURE)
    c['three_feature']     = list(THREE)
    c['full_minus_MSBI']   = [f for f in FO if f not in MSBI_FAMILY]
    c['full_minus_MPF']    = [f for f in FO if f not in MPF]
    c['full_minus_pd']     = [f for f in FO if f not in PD_FAMILY]
    c['MSBI_only']         = list(MSBI_FAMILY)
    c['three_minus_MSBI']  = [f for f in THREE if f != 'p_metric']
    return c

SEEDS = [11,22,33,44,55,66,77,88,99,111,123,234,345,456,567,678,789,890,901,1012]

# ---- globals, inherited by the workers through fork ----------------------
DF = None; Y = None; SSE = None; G = None

def gsplit(idx, seed, k=5):
    """The same splitting rule as gsplit in stability_selection_100_parallel.py."""
    g = G[idx]; ug = np.array(sorted(set(g)))
    rng = np.random.default_rng(seed); rng.shuffle(ug)
    f = {x: i % k for i, x in enumerate(ug)}
    fa = np.array([f[x] for x in g])
    return [(idx[fa != i], idx[fa == i]) for i in range(k)]

def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

def fit_predict(cols, tr, te):
    m = xgb.XGBRegressor(**P).fit(DF[cols].values[tr], Y[tr])
    return m.predict(DF[cols].values[te])

# ---- task 1: per-fold performance of each model configuration ------------
def job_model(a):
    name, cols, seed, fold = a
    tr, te = gsplit(np.arange(len(DF)), seed)[fold]
    p = fit_predict(cols, tr, te)
    return dict(config=name, seed=seed, fold=fold, n_feat=len(cols),
                idx=te.tolist(), pred=p.tolist())

# ---- task 2: single-feature drop-column ---------------------------------
def job_dropcol(a):
    feat, seed, fold = a
    cols = [f for f in FO if f != feat]
    tr, te = gsplit(np.arange(len(DF)), seed)[fold]
    p = fit_predict(cols, tr, te)
    return dict(feature=feat, seed=seed, fold=fold,
                idx=te.tolist(), pred=p.tolist())

# ---- task 3: grouped permutation (no refit) ------------------------------
def job_perm(a):
    gname, cols, seed, fold, rep = a
    tr, te = gsplit(np.arange(len(DF)), seed)[fold]
    m = xgb.XGBRegressor(**P).fit(DF[FO].values[tr], Y[tr])
    Xte = DF[FO].values[te].copy()
    base = r2(Y[te], m.predict(Xte))
    ci = [FO.index(c) for c in cols]
    rng = np.random.default_rng(seed * 1000 + fold * 10 + rep)
    perm = rng.permutation(len(te))
    Xp = Xte.copy()
    Xp[:, ci] = Xte[np.ix_(perm, ci)]      # shuffle the whole group as a single block
    return dict(group=gname, seed=seed, fold=fold, rep=rep,
                r2_base=base, r2_perm=r2(Y[te], m.predict(Xp)))

def pool_oof(recs, key):
    """Pool the five folds per (config|feature, seed) and return pooled OOF R2 in both spaces."""
    out = []
    df = pd.DataFrame(recs)
    for (name, seed), g in df.groupby([key, 'seed']):
        idx = np.concatenate([np.array(x) for x in g['idx']])
        pr  = np.concatenate([np.array(x) for x in g['pred']])
        o = np.argsort(idx); idx = idx[o]; pr = pr[o]
        out.append({key: name, 'seed': seed,
                    'r2_log': r2(Y[idx], pr),
                    'r2_eV':  r2(SSE[idx], np.expm1(pr)),
                    'n_feat': int(g['n_feat'].iloc[0]) if 'n_feat' in g else len(FO) - 1})
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('data', nargs='?', default='fin_data.csv')
    ap.add_argument('--outdir', default='.')
    ap.add_argument('--workers', type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--perm-reps', type=int, default=5)
    ap.add_argument('--skip-dropcol', action='store_true')
    a = ap.parse_args()

    global DF, Y, SSE, G
    DF = pd.read_csv(a.data)
    pat = DF.filename.str.extract(r'^POSCAR_([A-Za-z0-9]+)_([A-Za-z0-9]+?)(?:_(?:st|x|y|z)\d+)?$')
    DF['parent'] = pat[0] + '_' + pat[1]
    SSE = DF.sse.values.astype(float)
    Y = np.log1p(SSE)
    G = DF.parent.values
    os.makedirs(a.outdir, exist_ok=True)

    seeds = SEEDS[:a.seeds]
    cfg = build_configs()
    missing = [f for f in FO if f not in DF.columns]
    print(f'rows={len(DF)}  parents={DF.parent.nunique()}  missing_features={missing}', flush=True)
    assert not missing, f'missing features: {missing}'

    with Pool(a.workers) as pool:
        # --- 1. model configurations ---
        jobs = [(n, c, s, f) for n, c in cfg.items() for s in seeds for f in range(5)]
        print(f'[1/3] model configs: {len(cfg)} x {len(seeds)} seeds x 5 folds = {len(jobs)} fits', flush=True)
        recs = pool.map(job_model, jobs, chunksize=4)
        pm = pool_oof(recs, 'config')
        pm.to_csv(f'{a.outdir}/ablation_folds.csv', index=False)

        base = pm[pm.config == 'full52'].set_index('seed')
        rows = []
        for name, g in pm.groupby('config'):
            g = g.set_index('seed')
            rows.append(dict(config=name, n_feat=int(g.n_feat.iloc[0]),
                r2_log_mean=g.r2_log.mean(), r2_log_sd=g.r2_log.std(ddof=1),
                r2_eV_mean=g.r2_eV.mean(),  r2_eV_sd=g.r2_eV.std(ddof=1),
                d_r2_log_vs_full=(g.r2_log - base.r2_log).mean(),
                d_r2_log_sd=(g.r2_log - base.r2_log).std(ddof=1),
                d_r2_eV_vs_full=(g.r2_eV - base.r2_eV).mean()))
        pd.DataFrame(rows).sort_values('r2_log_mean', ascending=False)\
          .to_csv(f'{a.outdir}/ablation_summary.csv', index=False)
        print(open(f'{a.outdir}/ablation_summary.csv').read(), flush=True)

        # --- 2. single-feature drop-column ---
        if not a.skip_dropcol:
            jobs = [(f, s, k) for f in FO for s in seeds for k in range(5)]
            print(f'[2/3] drop-column: {len(FO)} features x {len(seeds)} x 5 = {len(jobs)} fits', flush=True)
            recs = pool.map(job_dropcol, jobs, chunksize=4)
            pd_ = pool_oof(recs, 'feature')
            rows = []
            for name, g in pd_.groupby('feature'):
                g = g.set_index('seed')
                rows.append(dict(feature=name,
                    d_r2_log=(g.r2_log - base.r2_log).mean(),
                    d_r2_log_sd=(g.r2_log - base.r2_log).std(ddof=1),
                    d_r2_eV=(g.r2_eV - base.r2_eV).mean()))
            pd.DataFrame(rows).sort_values('d_r2_log')\
              .to_csv(f'{a.outdir}/dropcol_single.csv', index=False)

        # --- 3. grouped permutation ---
        jobs = [(n, c, s, k, r) for n, c in GROUPS.items() for s in seeds
                for k in range(5) for r in range(a.perm_reps)]
        print(f'[3/3] grouped permutation: {len(jobs)} evaluations', flush=True)
        recs = pd.DataFrame(pool.map(job_perm, jobs, chunksize=4))
        recs['drop'] = recs.r2_base - recs.r2_perm
        recs.groupby('group')['drop'].agg(['mean', 'std', 'count'])\
            .rename(columns={'mean': 'perm_dR2_log', 'std': 'sd'})\
            .sort_values('perm_dR2_log', ascending=False)\
            .to_csv(f'{a.outdir}/perm_grouped.csv')
        print(open(f'{a.outdir}/perm_grouped.csv').read(), flush=True)

    with open(f'{a.outdir}/ablation_meta.json', 'w') as f:
        json.dump(dict(data=os.path.abspath(a.data), rows=len(DF),
                       parents=int(DF.parent.nunique()), seeds=seeds,
                       hyperparams=P, groups={k: v for k, v in GROUPS.items()},
                       configs={k: len(v) for k, v in cfg.items()}), f, indent=1)
    print('done', flush=True)

if __name__ == '__main__':
    main()
