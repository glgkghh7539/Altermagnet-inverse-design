#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_si3_parity.py - SI Fig. S3, the parity plot under grouped cross-validation.

The reviewer cited this figure directly: "Figure S3 shows pronounced underprediction in the
high-SSE tail, which is exactly the region used for optimization." The observation is correct,
so the figure is kept but redrawn on the canonical 3,845-row table under the pinned environment
(XGBoost 3.4.1), with a decile-bias sub-panel added to quantify the bias.

Out-of-fold predictions from GroupKFold(5) by parent are averaged over 20 seeds.
Outputs: SI_3.pdf/.png, si3_oof_predictions.csv, si3_decile.csv
"""
import os
import sys, numpy as np, pandas as pd, xgboost as xgb

# --- paths: resolved against this file, so the script runs from any directory ---
HERE     = os.path.dirname(os.path.abspath(__file__))
FIN_DATA = os.environ.get('FIN_DATA', os.path.join(HERE, '..', '..', 'data', 'fin_data.csv'))
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ[v]="1"
from sklearn.model_selection import GroupKFold
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _fonts


# Outputs land beside this script, not in whatever directory it was launched from.
OUTDIR = os.environ.get('REF_OUTDIR', os.path.dirname(os.path.abspath(__file__)))

P = dict(learning_rate=0.0210, max_depth=7, min_child_weight=1, subsample=0.7560,
         colsample_bytree=0.7283, reg_lambda=0.4606, reg_alpha=0.0524,
         n_estimators=300, objective='reg:squarederror', tree_method='hist', n_jobs=1)
FO=['avg_bond_length','max_bond_length','min_bond_length','std_bond_length','center_max_angle',
'center_min_angle','center_avg_angle','center_std_angle','nonmag_max_angle','nonmag_min_angle',
'nonmag_std_angle','labelled_1st','labelled_2nd','labelled_3rd','global_1st','global_2nd','global_3rd',
'avg_long_axis','avg_short_axis','avg_axis_ratio','avg_s','avg_delta','motif0_nonmag_count',
'magnetic_atomic_number','magnetic_electronegativity','nonmagnetic_atomic_number',
'nonmagnetic_electronegativity','hungarian_rotation_angle_deg','dimension','avg_motif_measure',
'unit_cell_volume','packing_fraction','p_metric','p_metric_std','d_orb_e','p_orb_e_non','d_lone_pair',
'proxy_M_magnet','delta_chi','abs_delta_chi','delta_Z','abs_delta_Z','pd_ratio','ax_eq_gap',
'bond_range','bond_cv','center_angle_spread','nonmag_angle_spread','delta_chi_times_axeq',
'd_global_local_1st','d_global_local_2nd','d_global_local_3rd']

df = pd.read_csv(FIN_DATA)
X = df[FO].values.astype(float); y = df['sse'].values.astype(float); g = df['parent'].values
NSEED = 20
oof = np.zeros((NSEED, len(y)))
for s in range(NSEED):
    rng = np.random.RandomState(s); u = np.unique(g); pm = rng.permutation(len(u))
    gs = np.array([{v: pm[i] for i, v in enumerate(u)}[v] for v in g])
    for tr, te in GroupKFold(5).split(X, y, gs):
        m = xgb.XGBRegressor(**P, random_state=42).fit(X[tr], np.log1p(y[tr]))
        oof[s, te] = m.predict(X[te])
    print(f'  seed {s} done', flush=True)
pred = np.expm1(oof.mean(axis=0))
r2 = 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()
print(f'\nOOF R²(eV) = {r2:.4f}   MAE = {np.abs(y-pred).mean()*1000:.1f} meV')
pd.DataFrame({'filename': df.filename, 'sse_dft': y, 'sse_pred': pred}).to_csv(
    'si3_oof_predictions.csv', index=False)

q = pd.qcut(y, 10, labels=False, duplicates='drop')
dec = pd.DataFrame({'decile': np.arange(1, q.max()+2),
                    'n': np.bincount(q),
                    'sse_mean': [y[q==i].mean() for i in range(q.max()+1)],
                    'pred_mean': [pred[q==i].mean() for i in range(q.max()+1)]})
dec['bias_pct'] = (dec.pred_mean-dec.sse_mean)/dec.sse_mean*100
dec.to_csv('si3_decile.csv', index=False)
print(dec.to_string(index=False))

_fonts.use()
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.4, 4.2), width_ratios=[1, 0.85])
lim = max(y.max(), pred.max())*1.04
a1.plot([0, lim], [0, lim], '--', color='0.45', lw=1)
a1.scatter(y, pred, s=5, alpha=0.35, linewidths=0, color='#4C72B0', rasterized=True)
a1.set_xlim(0, lim); a1.set_ylim(0, lim)
a1.set_xlabel('DFT SSE (eV)', fontsize=11); a1.set_ylabel('Predicted SSE (eV)', fontsize=11)
a1.text(0.04, 0.95, f'$R^2$ = {r2:.3f}\nMAE = {np.abs(y-pred).mean()*1000:.0f} meV\n$n$ = {len(y)}',
        transform=a1.transAxes, va='top', fontsize=9)
a1.set_title('a', loc='left', fontsize=13, fontweight='bold')
a2.axhline(0, color='0.45', lw=1)
a2.bar(dec.decile, dec.bias_pct, color=['#C44E52' if b < 0 else '#4C72B0' for b in dec.bias_pct],
       edgecolor='black', linewidth=0.5)
for d, b in zip(dec.decile, dec.bias_pct):
    a2.text(d, b + (6 if b > 0 else -12), f'{b:+.0f}', ha='center', fontsize=7)
a2.set_xticks(dec.decile); a2.set_xlabel('SSE decile', fontsize=11)
a2.set_ylabel('mean prediction bias (%)', fontsize=11)
a2.set_title('b', loc='left', fontsize=13, fontweight='bold')
for ax in (a1, a2):
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'SI_3.pdf'), bbox_inches='tight'); plt.savefig(os.path.join(OUTDIR, 'SI_3.png'), dpi=300, bbox_inches='tight')
print('\nwritten: SI_3.pdf, SI_3.png, si3_oof_predictions.csv, si3_decile.csv')
