#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_shap_figure.py - manuscript Fig. 2 (a: top-12 bars, b: beeswarm) and Table 1.

Two things differ from the original (see `README.md`):
  - values are reported in log1p space; the local-linearization conversion to eV is not used;
  - SHAP is computed only on held-out folds (the original used in-sample values from a model fitted to all rows).

Inputs:  shap_rank.csv, shap_heldout.npz   (both produced by `shap_recompute.py`)
Outputs: figure2.pdf/.png, table_shap_top12.csv
"""
import os, sys

# --- paths: shap_recompute.py writes these next to itself, in figures/ ---------
HERE        = os.path.dirname(os.path.abspath(__file__))
SHAP_DIR    = os.environ.get('SHAP_OUTDIR', os.path.join(HERE, '..'))
RANK_CSV    = os.path.join(SHAP_DIR, 'shap_rank.csv')
HELDOUT_NPZ = os.path.join(SHAP_DIR, 'shap_heldout.npz')

# Outputs land beside this script, not in whatever directory it was launched from.
OUTDIR = os.environ.get('REF_OUTDIR', HERE)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _fonts
from matplotlib.colors import Normalize

for f in (RANK_CSV, HELDOUT_NPZ):
    if not os.path.exists(f):
        sys.exit(f"{f} not found - run `python shap_recompute.py` first.")

rank = pd.read_csv(RANK_CSV)
z = np.load('shap_heldout.npz', allow_pickle=True)
SH, X, FEAT = z['shap'], z['X'], list(z['features'])

NTOP = 12
top = rank.head(NTOP)
idx = [FEAT.index(f) for f in top.feature]

# Mapping to the manuscript symbols. labelled_* is the distance between the two magnetic
# sites (motif centre to centre); global_* is the nearest metal-metal distance including
# periodic self-images (descriptor.ipynb cell 3). The SHAP ranks of the two (labelled 6th
# above global 11th) reproduce the manuscript ordering d_CC 5th above d_MM 8th, which supports the mapping.
PRETTY = {'p_metric': 'MSBI', 'packing_fraction': 'MPF', 'pd_ratio': '$p/d$ ratio',
          'p_metric_std': r'$\sigma_{\mathrm{inhom}}$',
          'labelled_1st': r'$d_{CC}^{(1)}$', 'global_1st': r'$d_{MM}^{(1)}$',
          'center_std_angle': r'$XMX_{\mathrm{std}}$',
          'center_avg_angle': r'$XMX_{\mathrm{avg}}$',
          'center_max_angle': r'$XMX_{\mathrm{max}}$',
          'p_orb_e_non': r'$n_{p,X}$', 'd_orb_e': r'$n_{d,M}$',
          'd_lone_pair': r'$n_{\mathrm{unpaired}}$', 'proxy_M_magnet': r'$\mu_{\mathrm{spin}}$'}
lab = [PRETTY.get(f, f.replace('_', ' ')) for f in top.feature]

_fonts.use()
plt.rcParams['mathtext.fontset'] = 'cm'
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), width_ratios=[1, 1.35])

# ---- (a) top-12 bars ----
y = np.arange(NTOP)[::-1]
ax1.barh(y, top.mean_abs_shap, color='#4C72B0', edgecolor='black', linewidth=0.6, height=0.72)
for yy, v, p in zip(y, top.mean_abs_shap, top.share_pct):
    ax1.text(v + top.mean_abs_shap.max()*0.015, yy, f'{p:.1f}%', va='center', fontsize=8)
ax1.set_yticks(y); ax1.set_yticklabels(lab, fontsize=9)
ax1.set_xlabel(r'mean $|$SHAP$|$   (log1p space)', fontsize=10)
ax1.set_xlim(0, top.mean_abs_shap.max() * 1.18)
ax1.tick_params(axis='x', labelsize=8)
for sp in ('top', 'right'): ax1.spines[sp].set_visible(False)
ax1.set_title('a', loc='left', fontsize=13, fontweight='bold')

# ---- (b) beeswarm ----
rng = np.random.default_rng(0)
for k, (j, yy) in enumerate(zip(idx, y)):
    v = SH[:, j]; f = X[:, j]
    lo, hi = np.nanpercentile(f, [1, 99])
    c = np.clip((f - lo) / (hi - lo + 1e-12), 0, 1)
    # vertical jitter proportional to local density - an approximation to a beeswarm
    h, edges = np.histogram(v, bins=90)
    dens = h[np.clip(np.digitize(v, edges) - 1, 0, len(h) - 1)]
    jit = rng.uniform(-1, 1, len(v)) * 0.34 * (dens / (dens.max() + 1e-12)) ** 0.5
    ax2.scatter(v, yy + jit, c=c, cmap='coolwarm', norm=Normalize(0, 1),
                s=2.0, linewidths=0, alpha=0.65, rasterized=True)
ax2.axvline(0, color='0.4', lw=0.7, zorder=0)
ax2.set_yticks(y); ax2.set_yticklabels([])
ax2.set_xlabel('SHAP value   (log1p space)', fontsize=10)
ax2.tick_params(axis='x', labelsize=8)
for sp in ('top', 'right', 'left'): ax2.spines[sp].set_visible(False)
ax2.set_title('b', loc='left', fontsize=13, fontweight='bold')
cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, 1), cmap='coolwarm'),
                  ax=ax2, pad=0.02, fraction=0.035)
cb.set_ticks([0, 1]); cb.set_ticklabels(['low', 'high']); cb.set_label('feature value', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'figure2.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, 'figure2.png'), dpi=300, bbox_inches='tight')

out = top[['feature', 'mean_abs_shap', 'share_pct', 'cum_pct']].copy()
out.columns = ['feature', 'mean_abs_SHAP_log1p', 'ratio_pct', 'cumulative_pct']
out.to_csv('table_shap_top12.csv', index=False)
print(f"total sum mean|SHAP| = {rank.mean_abs_shap.sum():.4f} (log1p)")
print(f"top three = {rank.share_pct.head(3).sum():.2f} %   top twelve cumulative = {top.cum_pct.iloc[-1]:.2f} %")
print(out.to_string(index=False))
print("\nwritten: figure2.pdf, figure2.png, table_shap_top12.csv")
