#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
msbi_threshold_bootstrap.py - reviewer's additional point (b)

  "The statement that large SSE is 'essentially confined' to MSBI above 0.4 should be
   qualified because about 15% of the high-SSE entries lie below that value. Give
   sensitivity, precision, and confidence intervals using **parents** rather than
   correlated strain variants."

Rows are correlated within a parent (strain variants), so counting is done per parent.
High-SSE means the top 10 % of per-parent maximum SSE. The bootstrap resamples *parents*.

Outputs:
  msbi_threshold_bootstrap.csv   per-resample statistics
  stdout                         point estimates and 95 % CIs

Usage: python msbi_threshold_bootstrap.py [fin_data.csv] [--thr 0.4] [--top 0.10]
       [--nboot 2000] [--seed 1034]
"""
import sys, argparse
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('data', nargs='?', default='fin_data.csv')
ap.add_argument('--thr',   type=float, default=0.4,  help='MSBI threshold')
ap.add_argument('--top',   type=float, default=0.10, help='top fraction counted as high-SSE')
ap.add_argument('--nboot', type=int,   default=2000)
ap.add_argument('--seed',  type=int,   default=1034)
a = ap.parse_args()

df = pd.read_csv(a.data)
for c in ('parent', 'sse', 'p_metric'):
    if c not in df.columns: sys.exit(f"missing column: {c}")

# per-parent representative: max SSE and the MSBI of that row
g = df.loc[df.groupby('parent')['sse'].idxmax(), ['parent', 'sse', 'p_metric']]
g = g.reset_index(drop=True)
n = len(g)
k = max(1, int(round(n * a.top)))
cut = g['sse'].nlargest(k).min()
g['high'] = g['sse'] >= cut
g['above'] = g['p_metric'] > a.thr
print(f"{n} parents; top {a.top:.0%} by SSE -> {int(g['high'].sum())} parents (SSE >= {cut:.4f} eV)")

def metrics(sub):
    hi, ab = sub['high'].values, sub['above'].values
    if hi.sum() == 0 or ab.sum() == 0: return (np.nan,)*3
    below = 1 - ab[hi].mean()          # fraction of high-SSE parents with MSBI < thr (the reviewer's ~15 %)
    prec  = hi[ab].mean()              # fraction of MSBI > thr parents that are high-SSE
    rec   = ab[hi].mean()              # fraction of high-SSE parents with MSBI > thr
    return below*100, prec*100, rec*100

pt = metrics(g)
rng = np.random.default_rng(a.seed)
boot = np.array([metrics(g.iloc[rng.integers(0, n, n)]) for _ in range(a.nboot)])
lo, hi = np.nanpercentile(boot, [2.5, 97.5], axis=0)

names = [f'high-SSE parents with MSBI < {a.thr}', 'precision', 'sensitivity (recall)']
print(f"\n{a.nboot} bootstrap resamples over parents (seed {a.seed})")
print(f"{'quantity':<36}{'estimate':>9}{'95 % CI':>20}")
for nm, p, l, h in zip(names, pt, lo, hi):
    print(f"{nm:<36}{p:>8.1f}%   [{l:>5.1f}, {h:>5.1f}]")

pd.DataFrame(boot, columns=['below_pct', 'precision_pct', 'recall_pct']).to_csv(
    'msbi_threshold_bootstrap.csv', index=False)
print("\nwritten: msbi_threshold_bootstrap.csv")
print("Note: with few high-SSE parents the interval is wide. If its upper limit contains"
      " the reviewer's ~15 %, we cannot claim that 15 % is incorrect.")
