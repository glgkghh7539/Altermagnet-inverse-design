import numpy as np, pandas as pd, xgboost as xgb, json, sys, hashlib
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
src,dst,csvp = sys.argv[1],sys.argv[2],sys.argv[3]
b=xgb.Booster(); b.load_model(src)
print('original feature_names:', b.feature_names, ' num_features:', b.num_features())
X=pd.read_csv(csvp)[FO].values.astype(float)
p_before=b.predict(xgb.DMatrix(X))              # predict positionally, without names
b.feature_names=list(FO)
b.feature_types=['float']*len(FO)
b.save_model(dst)
b2=xgb.Booster(); b2.load_model(dst)
print('re-saved feature_names:', (b2.feature_names or [])[:3], '...', len(b2.feature_names or []))
p_after=b2.predict(xgb.DMatrix(X, feature_names=FO))
d=np.abs(p_before-p_after)
print(f'prediction agreement check: max absolute difference {d.max():.3e}  (n={len(d)})')
assert d.max()<1e-9, 'predictions changed'
# check that the guard against a scrambled feature order actually fires
try:
    bad=FO[1:]+FO[:1]
    b2.predict(xgb.DMatrix(X, feature_names=bad)); print('  WARNING a wrong order was accepted')
except Exception as e:
    print('  OK a wrong feature order is now rejected:', str(e)[:80])
print('md5', hashlib.md5(open(dst,'rb').read()).hexdigest())
print('xgboost', xgb.__version__)
