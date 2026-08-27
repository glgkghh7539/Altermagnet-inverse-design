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
model_path, csv_path = sys.argv[1], sys.argv[2]
b=xgb.Booster(); b.load_model(model_path)
print('model feature count:', b.num_features())
df=pd.read_csv(csv_path)
X=df[FO].values.astype(float)
p=b.predict(xgb.DMatrix(X, feature_names=FO))
y=np.log1p(df.sse.values.astype(float))
r2=1-((y-p)**2).sum()/((y-y.mean())**2).sum()
print(f'{len(df)} rows   R2 in log1p space (in-sample) = {r2:.4f}')
pe=np.expm1(p); ye=df.sse.values.astype(float)
r2e=1-((ye-pe)**2).sum()/((ye-ye.mean())**2).sum()
print(f'             R2 in eV space (in-sample) = {r2e:.4f}')
print(f'                 MAE(eV) = {np.abs(ye-pe).mean():.4f}')
# contrast with a shuffled feature order, to confirm the order is correct
rng=np.random.default_rng(0); idx=rng.permutation(52)
ps=b.predict(xgb.DMatrix(X[:,idx], feature_names=FO))
r2s=1-((y-ps)**2).sum()/((y-y.mean())**2).sum()
print(f'  [control] with the feature order shuffled, R2 = {r2s:.4f}  (confirms the order matters)')
