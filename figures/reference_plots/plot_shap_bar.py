#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_shap_bar.py - the SHAP bar chart of manuscript Fig. 2.

The original (`archive/shap_bar_original.ipynb`) had the three values written in by hand
(`shap_values = [0.0328, 0.0359, 0.1122]`, in eV). They had been computed in a different
notebook and copied across, so the figure could not be reproduced from the plotting code
alone. This script reads `shap_rank.csv`, which `shap_recompute.py` generates.

A second change: the original converted the SHAP values from log1p space to eV by local
linearization (`scale = np.exp(z_pred)` and `shap_sse_run = shap_z * scale` in
`archive/KPS_DFT_free_BO_SGK_original.ipynb`). Additivity holds only in the model output
space, so that conversion is not used; mean|SHAP| is plotted in the training space (log1p).

Usage:
    python shap_recompute.py          # produces shap_rank.csv (needs fin_data.csv)
    python plot_shap_bar.py           # produces shap_bar.pdf / .png
"""
import os, sys
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# the _cohort / _fonts / _save helpers live beside this file; make that true
# however the script is invoked, not just via `python path/to/this.py`
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import _fonts


# Outputs land beside this script, not in whatever directory it was launched from.
OUTDIR = os.environ.get('REF_OUTDIR', os.path.dirname(os.path.abspath(__file__)))

CSV = os.environ.get('SHAP_RANK_CSV',
                     os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'shap_rank.csv'))
if not os.path.exists(CSV):
    sys.exit(f"{CSV} not found - run `python shap_recompute.py` first.")

df = pd.read_csv(CSV)
LABEL = {'p_metric': 'MSBI',
         'packing_fraction': 'MPF',
         'pd_ratio': '$p/d$ ratio'}
sel = df[df.feature.isin(LABEL)].copy()
sel['label'] = sel.feature.map(LABEL)
sel = sel.sort_values('mean_abs_shap')          # smallest at the bottom

_fonts.use()
plt.rcParams['mathtext.fontset'] = 'cm'
fig, ax = plt.subplots(figsize=(2, 2))
ax.barh(range(len(sel)), sel.mean_abs_shap,
        color='white', height=0.75, edgecolor='black', linewidth=1)
ax.set_yticks(range(len(sel)))
ax.set_yticklabels(sel.label, fontsize=7)
ax.set_xlabel('mean |SHAP|  (log1p space)', fontsize=7, labelpad=2)
ax.set_xlim(0, sel.mean_abs_shap.max() * 1.15)
ax.tick_params(axis='x', labelsize=7, length=2, pad=2)
for sp in ('top', 'right', 'left'):
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_linewidth(0.5)
plt.tight_layout(pad=0.3)
plt.savefig(os.path.join(OUTDIR, 'shap_bar.png'), dpi=600, bbox_inches='tight', transparent=True)
plt.savefig(os.path.join(OUTDIR, 'shap_bar.pdf'), bbox_inches='tight', transparent=True)

print("written: shap_bar.pdf, shap_bar.png")
print(sel[['feature', 'mean_abs_shap', 'share_pct']].to_string(index=False))
