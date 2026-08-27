#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_fig3abc.py - panels a-c of manuscript Fig. 3 (SSE vs MSBI / MPF / p-d).

The published caption colours the points by "mean SHAP value (eV)". That eV conversion has
been withdrawn (see README.md), so the points are coloured by the held-out SHAP value in log1p space.
Panels d-e (CuO and FeSi structures and bands) do not involve SHAP and need no redraw.

Input:  shap_heldout.npz  (produced by shap_recompute.py)
Output: figure3abc.pdf/.png
"""
import os
import sys, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _fonts


# Outputs land beside this script, not in whatever directory it was launched from.
OUTDIR = os.environ.get('REF_OUTDIR', os.path.dirname(os.path.abspath(__file__)))
from matplotlib.colors import TwoSlopeNorm
from matplotlib.ticker import LogLocator, NullFormatter

if not os.path.exists('shap_heldout.npz'):
    sys.exit("shap_heldout.npz not found - run `python shap_recompute.py` first.")
z = np.load('shap_heldout.npz', allow_pickle=True)
SH, X, FEAT, SSE = z['shap'], z['X'], list(z['features']), z['sse']

PANELS = [('p_metric', 'MSBI', 'a'),
          ('packing_fraction', 'MPF', 'b'),
          ('pd_ratio', r'$p/d$ electron ratio', 'c')]

_fonts.use()
plt.rcParams['mathtext.fontset'] = 'cm'
fig, axes = plt.subplots(1, 3, figsize=(13, 4.0))

for ax, (col, lab, tag) in zip(axes, PANELS):
    j = FEAT.index(col)
    x, c = X[:, j], SH[:, j]
    m = x > 0                      # log axis, so exclude zeros (some rows have MSBI exactly 0)
    lim = np.nanpercentile(np.abs(c), 99)
    sc = ax.scatter(x[m], SSE[m], c=c[m], cmap='coolwarm',
                    norm=TwoSlopeNorm(vcenter=0, vmin=-lim, vmax=lim),
                    s=5, linewidths=0, alpha=0.75, rasterized=True)
    ax.set_xscale('log')
    ax.set_xlabel(lab, fontsize=11)
    ax.tick_params(labelsize=9)
    # label only powers of ten, to stop minor-tick labels from colliding (notably in the p/d panel)
    ax.xaxis.set_major_locator(LogLocator(base=10))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(tag, loc='left', fontsize=13, fontweight='bold')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.045)
    cb.set_label('SHAP (log1p)', fontsize=8); cb.ax.tick_params(labelsize=7)
    if tag == 'a':
        ax.axvline(0.4, color='0.35', ls='--', lw=0.9)
        ax.annotate('MSBI = 0.4', xy=(0.4, ax.get_ylim()[1]*0.90),
                    xytext=(-4, 0), textcoords='offset points',
                    fontsize=8, color='0.3', ha='right', va='top')
    n_excl = int((~m).sum())
    if n_excl: print(f"  {col}: {n_excl} rows with x=0 excluded from the log axis")

axes[0].set_ylabel('SSE (eV)', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'figure3abc.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTDIR, 'figure3abc.png'), dpi=300, bbox_inches='tight')
print(f"plotted from {len(SSE)} rows. written: figure3abc.pdf, figure3abc.png")
print("note: panels d-e (CuO, FeSi) do not involve SHAP and are kept as published"
      " (archive/plotband_{CuO,FeSi}_original.ipynb).")
