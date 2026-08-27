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
feat,model,bojson,train = sys.argv[1:5]
F=pd.read_csv(feat); T=pd.read_csv(train)
b=xgb.Booster(); b.load_model(model)
pred=lambda X: float(np.expm1(b.predict(xgb.DMatrix(np.asarray([X],float), feature_names=FO)))[0])
BO=json.load(open(bojson))
CAND={42747:('FeS',2),27992:('CoO_sp',2),76985:('CrS',2),24595:('NiS',3),30070:('CoS',3),21953:('FeAs',3)}
bo={}
for cat in ['top5_4coordination','top5_6coordination']:
    for t in BO[cat]:
        if t['trial_number'] in CAND: bo[CAND[t['trial_number']][0]]=t
i_mpf=FO.index('packing_fraction')
t2=T[T.dimension==2]['packing_fraction']
print(f'2D MPF in the training table: median {t2.median():.4f}  25-75% {t2.quantile(.25):.4f}-{t2.quantile(.75):.4f}  (n={len(t2)})')
print()
print(f'{"candidate":9s} {"dim":>3s} {"BO MPF":>8s} {"struc MPF":>9s} {"ratio":>6s} {"pred(BO)":>9s} {"pred(MPF fixed)":>15s} {"delta(meV)":>11s}')
for c,(nm,dim) in [(v[0],v) for v in CAND.values()]:
    pass
for tn,(nm,dim) in CAND.items():
    t=bo[nm]; x=[t['features'][k] for k in FO]
    row=F[F.filename==f'POSCAR_{nm}__3generated']
    if row.empty: continue
    true_mpf=float(row['packing_fraction'].values[0])
    p0=pred(x)
    x2=list(x); x2[i_mpf]=true_mpf
    p1=pred(x2)
    print(f'{nm:9s} {dim:3d} {x[i_mpf]:8.4f} {true_mpf:9.4f} {true_mpf/x[i_mpf]:6.2f} {p0:9.3f} {p1:12.3f} {1000*(p1-p0):+9.1f}')
