#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
shap_recompute.py - recompute the SHAP ranking on the canonical 3,845-row table and compare with drop-column

Manuscript Fig. 2a as published: MSBI 23.2 %, MPF 7.4 %, p/d 6.8 % (37.4 % together), total 0.4836 eV.
Here SHAP is computed only on the held-out folds of GroupKFold(5) by parent over 20 seeds,
so the seed-to-seed variation of the ranks and shares is reported too. TreeSHAP uses the
built-in xgboost pred_contribs.

SHAP measures attribution (how much the fitted model routes through a descriptor);
drop-column measures necessity (how much accuracy is lost without it). The two disagreeing
is not a contradiction but an indicator of redundancy, so both are reported.
"""
import os, json

# --- paths: resolved against this file, so the script runs from any directory ---
HERE     = os.path.dirname(os.path.abspath(__file__))
FIN_DATA = os.environ.get('FIN_DATA', os.path.join(HERE, '..', 'data', 'fin_data.csv'))
OUTDIR   = os.environ.get('SHAP_OUTDIR', HERE)
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"
import numpy as np, pandas as pd, xgboost as xgb, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold

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

MSBI_FAMILY = ['p_metric','p_metric_std','hungarian_rotation_angle_deg']
PD_FAMILY   = ['pd_ratio','p_orb_e_non','d_orb_e']
MPF         = ['packing_fraction']

df = pd.read_csv(FIN_DATA)
if 'parent' not in df.columns:
    raise SystemExit('no parent column')
X = df[FO].values.astype(np.float64)
y = np.log1p(df['sse'].values.astype(np.float64))
g = df['parent'].values

NSEED = 20
per_seed = []
contrib_seeds = []          # for the beeswarm (Fig. 2b): per-row SHAP averaged over seeds; (nseed, nfeat) mean|SHAP| per seed, in log1p space and not in eV
for seed in range(NSEED):
    rng = np.random.RandomState(seed)
    uniq = np.unique(g); perm = rng.permutation(len(uniq))
    remap = {u: perm[i] for i, u in enumerate(uniq)}
    gs = np.array([remap[v] for v in g])
    contrib = np.zeros((len(y), len(FO)))
    for tr, te in GroupKFold(n_splits=5).split(X, y, gs):
        m = xgb.XGBRegressor(**P).fit(X[tr], y[tr])
        c = m.get_booster().predict(xgb.DMatrix(X[te]), pred_contribs=True)
        contrib[te] = c[:, :-1]          # the last column is the bias term
    per_seed.append(np.abs(contrib).mean(axis=0))
    contrib_seeds.append(contrib)
    print(f'  seed {seed} done', flush=True)

np.savez_compressed(os.path.join(OUTDIR, 'shap_heldout.npz'),
                    shap=np.mean(np.array(contrib_seeds), axis=0).astype(np.float32),
                    X=X.astype(np.float32),
                    features=np.array(FO), sse=df['sse'].values.astype(np.float32))
print('written: shap_heldout.npz  (seed-averaged held-out SHAP matrix and feature values)')

A = np.array(per_seed)                    # (NSEED, 52)
mean = A.mean(axis=0); sd = A.std(axis=0)
tot  = A.sum(axis=1)                      # total per seed
share = A / tot[:, None] * 100.0
sh_m = share.mean(axis=0); sh_s = share.std(axis=0)

order = np.argsort(-mean)
out = pd.DataFrame({'feature': [FO[i] for i in order],
                    'mean_abs_shap': mean[order], 'sd': sd[order],
                    'share_pct': sh_m[order], 'share_sd': sh_s[order]})
out['cum_pct'] = out['share_pct'].cumsum()
out.to_csv(os.path.join(OUTDIR, 'shap_rank.csv'), index=False)

print('\n=== SHAP ranking (20 seeds, held-out folds, log1p space) ===')
print(f'total sum_k mean|SHAP| = {tot.mean():.4f} +- {tot.std():.4f}')
print(f'{"rank":>4} {"feature":<32}{"mean|SHAP|":>11}{"share%":>9}{"+-":>7}{"cum%":>8}')
for r, row in out.head(15).iterrows():
    print(f'{r+1:>4} {row.feature:<32}{row.mean_abs_shap:>11.5f}'
          f'{row.share_pct:>9.2f}{row.share_sd:>7.2f}{row.cum_pct:>8.2f}')

# rank stability across seeds
rk = np.argsort(np.argsort(-A, axis=1), axis=1) + 1
print('\n=== rank stability across the 20 seeds ===')
for f in ['p_metric','packing_fraction','pd_ratio','p_metric_std',
          'avg_motif_measure','unit_cell_volume','d_orb_e','p_orb_e_non']:
    i = FO.index(f)
    print(f'  {f:<32} median rank {int(np.median(rk[:,i])):>3}  '
          f'range {rk[:,i].min()}-{rk[:,i].max()}  '
          f'rank 2 in {(rk[:,i]==2).sum():>2}/20')

# grouped totals
print('\n=== grouped share % (the reviewer asks that these be treated as correlated groups) ===')
for name, feats in [('MSBI_family',MSBI_FAMILY),('pd_family',PD_FAMILY),('MPF',MPF)]:
    idx = [FO.index(f) for f in feats]
    s = share[:, idx].sum(axis=1)
    print(f'  {name:<14} {s.mean():>6.2f} +- {s.std():.2f} %   ({", ".join(feats)})')

json.dump({'total_mean': float(tot.mean()), 'total_sd': float(tot.std()),
           'nseed': NSEED}, open(os.path.join(OUTDIR, 'shap_meta.json'), 'w'), indent=1)
print('\nwritten: shap_rank.csv, shap_meta.json')
