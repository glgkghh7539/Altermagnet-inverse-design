#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_model.py - report what the deposited surrogate artifact is and what it reproduces.

Why verify rather than refit. The hyperparameters themselves are recorded in SI Table S3
(`tab:xgb_hyperparams`): learning_rate 0.0210, max_depth 7, min_child_weight 1,
min_child_weight 1, subsample 0.7560, colsample_bytree 0.7283, reg_lambda 0.4606,
subsample 0.756, colsample_bytree 0.728, reg_lambda 0.461, reg_alpha 0.0524. But the model
file itself carries no training parameters (`learner_train_param` is empty), and the XGBoost
version and seed of that particular fit were not logged. Refitting from Table S3 with the
artifact's 1,015 trees reproduces it to a correlation of 0.996 and a mean difference of
17.5 meV but is not bit-identical; the difference is the same whichever table (3,845 or
3,851) or seed (42 or 0) is used, so the residual looks like a library-version effect. This
script therefore verifies what the artifact reproduces rather than refitting it.

Recoverable from the file:
  - the 52 features and their order
  - 1,015 trees, all of depth exactly 7, matching the max_depth=7 used by the analysis scripts
  - base_score 0.2724, serialization version XGBoost 3.4.1
Not recoverable:
  - learning_rate, subsample, colsample_bytree, reg_lambda, reg_alpha, the early-stopping schedule

Every analysis in the paper and in this revision (ablation, drop-column, stability, held-out,
SHAP) uses models fitted afresh with the hyperparameters written into the scripts in `model/`
and `figures/`, not this artifact. The artifact is used only for the Table 2 predictions.

Usage: python verify_model.py [fin_data.csv] [final_model_all_named.json]
"""
import sys, os, json
import numpy as np, pandas as pd, xgboost as xgb

DATA  = sys.argv[1] if len(sys.argv) > 1 else 'fin_data.csv'
MODEL = sys.argv[2] if len(sys.argv) > 2 else 'final_model_all_named.json'
for f in (DATA, MODEL):
    if not os.path.exists(f):
        sys.exit(f"file not found: {f}")

meta = json.load(open(MODEL))['learner']
names = meta['feature_names']
ntree = int(meta['gradient_booster']['model']['gbtree_model_param']['num_trees'])
print(f"model  : {MODEL}")
print(f"  {len(names)} features, {ntree} trees, serialization {json.load(open(MODEL)).get('version')}")
print(f"  training hyperparameters recorded: {'none' if not meta.get('learner_train_param') else meta['learner_train_param']}")

df = pd.read_csv(DATA)
missing = [n for n in names if n not in df.columns]
if missing:
    sys.exit(f"{len(missing)} features absent from the data: {missing[:5]}")
X = df[names].values.astype(np.float64)          # select by name, so a mismatched order cannot occur
y = df['sse'].values.astype(np.float64)

bst = xgb.Booster(); bst.load_model(MODEL)
pred_log = bst.predict(xgb.DMatrix(X, feature_names=names))
pred_eV  = np.expm1(pred_log)

def r2(t, p): return 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
print(f"\ndata   : {DATA}  ({len(df)} rows)")
print(f"  in-sample R² (log1p) = {r2(np.log1p(y), pred_log):.4f}")
print(f"  in-sample R² (eV)    = {r2(y, pred_eV):.4f}")
print(f"  MAE (eV)             = {np.abs(y - pred_eV).mean()*1000:.1f} meV")
print("\nNote: the values above are in-sample and are not a performance estimate."
      " The reported performance comes from parent-grouped GroupKFold and is reproduced by"
      " `model/ablation_grouped.py`.")

# check that a scrambled feature order is rejected - this was the purpose of re-saving the model
try:
    shuffled = names[::-1]
    bst.predict(xgb.DMatrix(df[shuffled].values, feature_names=shuffled))
    print("\nWARNING a reversed feature order was accepted - the name check is not working.")
except Exception as e:
    print(f"\nfeature-order check: passed (a reversed order is rejected - {type(e).__name__})")
