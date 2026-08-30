#!/usr/bin/env python
"""Reproduce the prototype-matching numbers of Table 2 of the manuscript.

Rule (Methods; Supplementary Information, prototype matching)
------------------------------------------------------------
1. Restrict the reference pool to prototypes of the requested coordination class.
   Square-planar : motif0_nonmag_count == 4 and center_max_angle > 170 deg  (59 structures)
   Octahedral    : motif0_nonmag_count == 6 and center_max_angle > 170 deg  (2,340 structures)
2. Rank by cosine similarity in a z-score-normalized space of the 38 purely structural
   descriptors below (the 52 model features less the 14 that depend on composition).
   The scaler is fitted on the full table, not on the restricted pool.
3. Break ties by Euclidean distance.

The optimizer's descriptor vectors are read from optimization_results_resumable.json.

Note on the reference table
---------------------------
The matching was performed on the pre-deduplication table, released here as
optimization/fin_data_bo.csv (3,851 rows), which is the default input of this script and
reproduces every published value exactly.  The audited table data/fin_data.csv (3,845 rows; six
duplicate rows removed, see the response letter, section 2.7) leaves the rank-1 prototype of every
candidate unchanged, but shifts the cosine similarities by about 0.001 and the square-planar pool
from 59 to 58 structures.  Pass a table explicitly to check either one:

    python reproduce_similarity.py                       # optimization/fin_data_bo.csv (default)
    python reproduce_similarity.py ../data/fin_data.csv  # audited 3,845-row table
"""
import json, os, sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

F38 = ['avg_bond_length','max_bond_length','min_bond_length','std_bond_length','center_max_angle',
'center_min_angle','center_avg_angle','center_std_angle','nonmag_max_angle','nonmag_min_angle',
'nonmag_std_angle','labelled_1st','labelled_2nd','labelled_3rd','global_1st','global_2nd','global_3rd',
'avg_long_axis','avg_short_axis','avg_axis_ratio','avg_s','avg_delta','motif0_nonmag_count',
'hungarian_rotation_angle_deg','dimension','avg_motif_measure','unit_cell_volume','packing_fraction',
'p_metric','p_metric_std','ax_eq_gap','bond_range','bond_cv','center_angle_spread',
'nonmag_angle_spread','d_global_local_1st','d_global_local_2nd','d_global_local_3rd']

# trial number -> (label, coordination, published c, published d)   [Table 2]
TABLE2 = {42747:('Fe-S',4,0.694, 9.17), 28382:('Ni-S',4,0.555,12.93),
          27992:('Co-O',4,0.634,10.95), 76985:('Cr-S',4,0.662, 9.75),
          24595:('Ni-S',6,0.897,20.72), 30070:('Co-S',6,0.905,24.35),
          81507:('Cu-S',6,0.904,21.63), 21953:('Fe-As',6,0.893,21.00)}

here = os.path.dirname(os.path.abspath(__file__))
table = sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, 'fin_data_bo.csv')
df = pd.read_csv(table)
res = json.load(open(os.path.join(here, 'optimization_results_resumable.json')))
trials = {t['trial_number']: t for c in ('top5_4coordination','top5_6coordination') for t in res[c]}

scaler = StandardScaler()
X = scaler.fit_transform(df[F38].values)
means = df[F38].mean().values

print('table: %s  (%d rows)' % (os.path.normpath(table), len(df)))
print('%-6s %-3s %6s  %-24s %8s %8s   %8s %8s  %s'
      % ('M-X','Nx','pool','rank-1 prototype','c','d','c(pub)','d(pub)','match'))
ok_all = True
for n, (lab, nx, pc, pd_pub) in TABLE2.items():
    pool = df.index[(df.motif0_nonmag_count == nx) & (df.center_max_angle > 170)]
    feats = trials[n]['features']
    v = np.array([feats.get(c, np.nan) for c in F38], float)
    nan = np.isnan(v); v[nan] = means[nan]          # not reached for the deposited trials
    vs = scaler.transform(v.reshape(1, -1))[0]
    Xs, names = X[pool], df.filename.values[pool]
    sims = (Xs @ vs) / (np.linalg.norm(Xs, axis=1) * np.linalg.norm(vs))
    d2 = ((Xs - vs) ** 2).sum(1)
    i = np.lexsort((d2, -sims))[0]
    c, d = float(sims[i]), float(np.sqrt(d2[i]))
    ok = (round(c, 3) == pc) and (round(d, 2) == pd_pub)
    ok_all &= ok
    print('%-6s %-3d %6d  %-24s %8.4f %8.3f   %8.3f %8.2f  %s'
          % (lab, nx, len(pool), names[i], c, d, pc, pd_pub, 'OK' if ok else 'differs'))
print('\nall eight rows reproduced exactly' if ok_all else
      '\nrank-1 prototypes as published; similarity values differ in the third decimal '
      '(see the note on the reference table in the docstring)')
