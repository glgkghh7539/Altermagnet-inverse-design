#!/usr/bin/env python
"""Run the descriptor cells of descriptor.ipynb verbatim to produce the 52 features.

The cells are top-level scripts that glob POSCAR_* from the working directory, so we chdir
into the target directory and exec the original code unchanged.
"""
import os, sys, io, contextlib, warnings
import pandas as pd, numpy as np
warnings.filterwarnings('ignore')

target, celldir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.chdir(target)
ns = {'__name__': '__main__'}

CELLS = [('01','BOND_df'), ('03','NN_df'), ('05','VOL_df'), ('10','ELONG_df'),
         ('12','ATOM_df'), ('15','ROT_df'), ('16','P_METRIC_df')]
for c, var in CELLS:
    src = open(os.path.join(celldir, f'cell{c}.py'), errors='replace').read()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, f'cell{c}', 'exec'), ns)
    df = ns.get(var)
    print(f'  cell{c} -> {var}: {0 if df is None else len(df)} rows', flush=True)
    if df is None or len(df) == 0:
        print(buf.getvalue()[-800:]); sys.exit(f'cell{c} failed')

# lookup tables
exec(compile(open(os.path.join(celldir,'cell19.py'),errors='replace').read(),'cell19','exec'), ns)

# merge and derived-feature logic of cell 21, excluding only the final external CSV merge (absolute path)
src21 = open(os.path.join(celldir,'cell21.py'),errors='replace').read()
src21 = src21.split("AL_df = pd.read_csv")[0]
exec(compile(src21, 'cell21', 'exec'), ns)
m = ns['merge_df']

FEATURE_ORDER = ['avg_bond_length','max_bond_length','min_bond_length','std_bond_length',
'center_max_angle','center_min_angle','center_avg_angle','center_std_angle','nonmag_max_angle',
'nonmag_min_angle','nonmag_std_angle','labelled_1st','labelled_2nd','labelled_3rd','global_1st',
'global_2nd','global_3rd','avg_long_axis','avg_short_axis','avg_axis_ratio','avg_s','avg_delta',
'motif0_nonmag_count','magnetic_atomic_number','magnetic_electronegativity',
'nonmagnetic_atomic_number','nonmagnetic_electronegativity','hungarian_rotation_angle_deg',
'dimension','avg_motif_measure','unit_cell_volume','packing_fraction','p_metric','p_metric_std',
'd_orb_e','p_orb_e_non','d_lone_pair','proxy_M_magnet','delta_chi','abs_delta_chi','delta_Z',
'abs_delta_Z','pd_ratio','ax_eq_gap','bond_range','bond_cv','center_angle_spread',
'nonmag_angle_spread','delta_chi_times_axeq','d_global_local_1st','d_global_local_2nd',
'd_global_local_3rd']
missing = [c for c in FEATURE_ORDER if c not in m.columns]
if missing: sys.exit(f'missing columns: {missing}')
m['filename'] = m['filename'].astype(str).str.replace(r'^\./','',regex=True)
res = m[['filename']+FEATURE_ORDER].copy()
nan_rows = res[res[FEATURE_ORDER].isna().any(axis=1)]['filename'].tolist()
if nan_rows: print('  WARNING rows containing NaN:', nan_rows)
res.to_csv(out, index=False)
print(f'\nwritten: {out}   {len(res)} rows x {len(FEATURE_ORDER)} features')
print('  structures:', ', '.join(sorted(res.filename)))
