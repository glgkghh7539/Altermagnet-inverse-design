#!/usr/bin/env python
import os
import numpy as np, pandas as pd, xgboost as xgb, json, sys
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
feat, train, model, bojson = sys.argv[1:5]
F=pd.read_csv(feat); T=pd.read_csv(train)
b=xgb.Booster(); b.load_model(model)
pred=lambda X: np.expm1(b.predict(xgb.DMatrix(np.asarray(X,float), feature_names=FO)))

# --- check 1: do the matched rows reproduce the parent values stored in fin_data? ---
MATCH={'FeS':'POSCAR_Cu2O2_1_st950','CoO_sp':'POSCAR_Mn2N2_3_z950','CrS':'POSCAR_Mn2N2_3_z950',
       'NiS':'POSCAR_Cr2S2_1_y950','CoS':'POSCAR_Cr2S2_1_x050','FeAs':'POSCAR_Cr2S2_1_y950'}
print('='*78); print('check: recomputed matched descriptors vs the values stored in fin_data'); print('='*78)
Ti=T.set_index('filename')
for c,p in MATCH.items():
    r=F[F.filename==f'POSCAR_{c}__2matched']
    if r.empty or p not in Ti.index: continue
    a=r[FO].values[0]; t=Ti.loc[p,FO].values.astype(float)
    d=np.abs(a-t); rel=d/(np.abs(t)+1e-9)
    print(f'  {c:8s} <- {p:22s} max abs diff {d.max():.3e}  max rel diff {rel.max():.3e}  '
          f'{"match" if d.max()<1e-6 else "mismatched columns: "+str([FO[i] for i in np.where(d>1e-6)[0]][:4])}')

# --- Williams leverage ---
X=T[FO].values.astype(float)
XtX_inv=np.linalg.pinv(X.T@X)
lev=lambda x: float(np.asarray(x,float)@XtX_inv@np.asarray(x,float))
hstar=3*len(FO)/len(T)
print(f'\nWilliams h* = 3p/n = 3*{len(FO)}/{len(T)} = {hstar:.4f}')

# --- the BO descriptor vectors ---
BO=json.load(open(bojson))
CAND={42747:'FeS',27992:'CoO_sp',76985:'CrS',24595:'NiS',30070:'CoS',21953:'FeAs'}
bo={}
for cat in ['top5_4coordination','top5_6coordination']:
    for t in BO[cat]:
        if t['trial_number'] in CAND: bo[CAND[t['trial_number']]]=t

DFT={'NiS':0.8707,'CoS':1.0824,'CoO_sp':0.9735,'FeS':1.3548,'FeAs':1.4023,'CrS':0.4106}
CONV={'NiS':0.886,'FeAs':1.549,'CoO_sp':1.026,'FeS':1.355}   # converged values from the Point 2 tests
STAGE=[('2matched','(2) matched prototype'),('3generated','(3) generated'),('4relaxed','(4) relaxed (ISIF=2)'),
       ('5fullrelax','(5) fully relaxed (ISIF=3)'),('5fullrelax_PARTIAL','(5) fully relaxed (running)')]
rows=[]
print('\n'+'='*104)
print('TABLE 2 rebuild - the model re-evaluated on the descriptors of each stage')
print('='*104)
for c in ['NiS','CoS','FeAs','FeS','CoO_sp','CrS']:
    t=bo[c]; x=np.array([t['features'][k] for k in FO])
    dft=CONV.get(c,DFT[c])
    print(f'\n### {c}   DFT (converged) = {dft:.3f} eV' + (f'  [base {DFT[c]:.3f}]' if c in CONV else ''))
    print(f'    {"stage":22s} {"MSBI":>7s} {"MPF":>7s} {"V":>7s} {"pred SSE":>8s} {"err %":>8s} {"leverage":>9s} {"AD":>4s}')
    pr=float(pred([x])[0]); h=lev(x)
    print(f'    {"(1) BO vector":22s} {x[FO.index("p_metric")]:7.4f} {x[FO.index("packing_fraction")]:7.4f} '
          f'{x[FO.index("unit_cell_volume")]:7.2f} {pr:8.3f} {100*(pr-dft)/dft:+8.1f} {h:9.4f} {"OUT" if h>hstar else "in":>4s}')
    rows.append(dict(cand=c,stage='BO',pred=pr,dft=dft,lev=h))
    for tag,lab in STAGE:
        r=F[F.filename==f'POSCAR_{c}__{tag}']
        if r.empty: continue
        xx=r[FO].values[0]; pp=float(pred([xx])[0]); hh=lev(xx)
        print(f'    {lab:22s} {xx[FO.index("p_metric")]:7.4f} {xx[FO.index("packing_fraction")]:7.4f} '
              f'{xx[FO.index("unit_cell_volume")]:7.2f} {pp:8.3f} {100*(pp-dft)/dft:+8.1f} {hh:9.4f} {"OUT" if hh>hstar else "in":>4s}')
        rows.append(dict(cand=c,stage=lab,pred=pp,dft=dft,lev=hh))
R=pd.DataFrame(rows)
print('\n'+'='*78); print('mean absolute relative error per stage'); print('='*78)
for s in R.stage.unique():
    g=R[R.stage==s]
    print(f'  {s:24s} n={len(g)}  MRE = {100*np.abs((g.pred-g.dft)/g.dft).mean():6.1f}%   '
          f'median leverage {g.lev.median():.4f}  outside AD {int((g.lev>hstar).sum())}/{len(g)}')
R.to_csv(os.environ.get('TABLE2_OUT', 'table2_rebuilt.csv'), index=False)
