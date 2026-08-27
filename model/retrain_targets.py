#!/usr/bin/env python3
"""Reviewer Point 2-1: do the conclusions survive retraining on more robust SSE definitions?

The evaluation protocol matches cell 11 of 20251020_KPS_DFT_free_BO.ipynb:
  * GroupKFold OOF (no outer hold-out); the group is the parent, i.e. filename with the strain suffix removed
  * groups are sorted by median target before GroupKFold, to stratify
  * target transformed with log1p; sample weight = 1/group size, to offset strain over-representation
  * early stopping on an inner GroupKFold split
Only the target is varied, between sse, sse_p95 and sse_mean_w.

Usage: retrain_targets.py [n_trials] [outdir]
"""
import os, re, sys, json

# --- paths: resolved against this file, so the script runs from any directory ---
HERE         = os.path.dirname(os.path.abspath(__file__))
FIN_DATA     = os.environ.get('FIN_DATA', os.path.join(HERE, '..', 'data', 'fin_data.csv'))
SSE_VARIANTS = os.environ.get('SSE_VARIANTS',
                              os.path.join(HERE, '..', 'data', 'raw', 'sse_variants_all.csv'))
import numpy as np, pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr
import optuna, shap

N_HP_TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
OUT = sys.argv[2] if len(sys.argv) > 2 else 'retrain_out'
ONLY = sys.argv[3] if len(sys.argv) > 3 else None      # run only this target tag
N_JOBS = int(sys.argv[4]) if len(sys.argv) > 4 else 1  # parallel Optuna trials
os.makedirs(OUT, exist_ok=True)

# ---- the same settings as the notebook ----
TRANSFORM_KIND = 'log1p'; USE_SMEARING = False
N_SPLITS = 10; INNER_SPLITS = 5; SEED_BASE = 1034
N_HP_RUNS = 3; N_OOF_RUNS_BEST = 3

f_t   = lambda y: np.log1p(np.maximum(y, 0))
finv_t = lambda z: np.expm1(z)

INT_KEYS = {"max_depth", "min_child_weight", "n_estimators"}
FLOAT_KEYS = {"learning_rate", "subsample", "colsample_bytree", "reg_lambda", "reg_alpha"}

def cast_params(p):
    return {k: (int(v) if k in INT_KEYS else float(v) if k in FLOAT_KEYS else v) for k, v in p.items()}

def build_model(params, n_estimators=None, random_state=42):
    p = cast_params(dict(params))
    if n_estimators is not None: p['n_estimators'] = int(n_estimators)
    p.setdefault('n_estimators', 5000)
    return XGBRegressor(**p, n_jobs=1, random_state=int(random_state),
                        tree_method='hist', verbosity=0,
                        eval_metric='rmse', early_stopping_rounds=100)

def make_inner_group_split(train_groups_vec, inner_splits=5):
    ug = np.array(pd.unique(train_groups_vec))
    n_s = min(max(2, inner_splits), len(ug))
    gkf = GroupKFold(n_splits=n_s)
    tr_g_idx, va_g_idx = next(gkf.split(ug, groups=ug))
    return train_groups_vec.isin(set(ug[tr_g_idx])).values, train_groups_vec.isin(set(ug[va_g_idx])).values

def get_stratified_group_order(groups_series, target_series, n_bins=10, seed=42):
    g = pd.DataFrame({'group': groups_series, 'target': target_series})
    med = g.groupby('group')['target'].median()
    nb = min(n_bins, len(med))
    rng = np.random.default_rng(seed)
    if nb < 2:
        idx = med.index.to_numpy(); rng.shuffle(idx); return idx
    bins = pd.qcut(med.rank(method='first'), nb, labels=False)
    out = []
    for b in range(nb):
        m = med.index[bins == b].to_numpy(); rng.shuffle(m); out.append(m)
    order = []
    for i in range(max(len(x) for x in out)):
        for x in out:
            if i < len(x): order.append(x[i])
    return np.array(order)

SUFFIX = ('st','x','y','z','a','b','c','strain','eps','ea','eb','ec','scale')
def seed_id(s):
    toks = os.path.splitext(os.path.basename(str(s)))[0].split('_')
    while len(toks) > 1 and re.match(r'^(?:' + '|'.join(SUFFIX) + r')[\+\-]?\d*(?:\.\d+)?$', toks[-1], re.I):
        toks.pop()
    return '_'.join(toks)


def oof_run(params, X_all, y_all, groups, sw, n_runs, seed_base):
    n_groups = groups.nunique()
    n_splits_eff = min(N_SPLITS, n_groups)
    runs = []
    best_iters = []
    oof_last = None
    for run_idx in range(n_runs):
        order_groups = get_stratified_group_order(groups, pd.Series(y_all, index=groups.index),
                                                  n_bins=n_splits_eff, seed=seed_base + 1000*run_idx)
        ord_map = pd.Series(np.arange(len(order_groups)), index=order_groups)
        idx_sorted = np.argsort(ord_map[groups.values].values)
        gkf = GroupKFold(n_splits=n_splits_eff)
        s = np.zeros(len(y_all)); c = np.zeros(len(y_all), dtype=int)
        for tr_s, te_s in gkf.split(idx_sorted, y_all[idx_sorted], groups=groups.iloc[idx_sorted]):
            tr, te = idx_sorted[tr_s], idx_sorted[te_s]
            g_tr = groups.iloc[tr]
            m_in, v_in = make_inner_group_split(g_tr, INNER_SPLITS)
            model = build_model(params, n_estimators=5000)
            model.fit(X_all[tr][m_in], f_t(y_all[tr][m_in]), sample_weight=sw[tr][m_in],
                      eval_set=[(X_all[tr][v_in], f_t(y_all[tr][v_in]))], verbose=False)
            bi = int(getattr(model, 'best_iteration', 0)) + 1
            best_iters.append(bi)
            s[te] += finv_t(model.predict(X_all[te], iteration_range=(0, bi))); c[te] += 1
        valid = c > 0
        pred = np.zeros_like(s); pred[valid] = s[valid]/c[valid]
        runs.append(dict(r2=r2_score(y_all[valid], pred[valid]),
                         mae=mean_absolute_error(y_all[valid], pred[valid]),
                         rho=spearmanr(y_all[valid], pred[valid]).statistic))
        oof_last = pred
    return runs, int(np.median(best_iters)), oof_last


def run_for_target(df, target, feats, tag):
    d = df.dropna(subset=[target]).reset_index(drop=True)
    groups = d['filename'].astype(str).map(seed_id)
    X = d[feats].astype(float).values
    y = d[target].astype(float).values
    sw = groups.map(1.0/groups.value_counts()).values
    print('[%s] n=%d, groups=%d, features=%d, y range %.3f-%.3f' % (tag, len(d), groups.nunique(), len(feats), y.min(), y.max()))

    def objective(trial):
        p = dict(learning_rate=trial.suggest_float('learning_rate', 0.01, 0.08),
                 max_depth=trial.suggest_int('max_depth', 4, 9),
                 min_child_weight=trial.suggest_int('min_child_weight', 1, 7),
                 subsample=trial.suggest_float('subsample', 0.6, 1.0),
                 colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                 reg_lambda=trial.suggest_float('reg_lambda', 1e-2, 10.0, log=True),
                 reg_alpha=trial.suggest_float('reg_alpha', 1e-3, 1.0, log=True))
        runs, _, _ = oof_run(p, X, y, groups, sw, N_HP_RUNS, SEED_BASE + 100000*trial.number)
        return float(np.mean([r['r2'] for r in runs]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED_BASE))
    st.optimize(objective, n_trials=N_HP_TRIALS, n_jobs=N_JOBS, show_progress_bar=False)
    best = st.best_params
    runs, bi, oof = oof_run(best, X, y, groups, sw, N_OOF_RUNS_BEST, SEED_BASE)
    res = dict(target=target, n=len(d), n_groups=int(groups.nunique()),
               r2_mean=float(np.mean([r['r2'] for r in runs])), r2_std=float(np.std([r['r2'] for r in runs])),
               mae_mean=float(np.mean([r['mae'] for r in runs])), rho_mean=float(np.mean([r['rho'] for r in runs])),
               best_params=best, best_iter=bi)
    # final model and SHAP
    fm = build_model(best, n_estimators=bi)
    fm.set_params(early_stopping_rounds=None)
    fm.fit(X, f_t(y), sample_weight=sw, verbose=False)
    sv = shap.TreeExplainer(fm).shap_values(X)
    imp = pd.DataFrame({'feature': feats, 'mean_abs_shap': np.abs(sv).mean(0)}).sort_values('mean_abs_shap', ascending=False)
    imp['rank'] = np.arange(1, len(imp)+1)
    imp.to_csv(os.path.join(OUT, 'shap_%s.csv' % tag), index=False)
    pd.DataFrame({'filename': d['filename'], 'y': y, 'oof': oof}).to_csv(os.path.join(OUT, 'oof_%s.csv' % tag), index=False)
    return res, imp


if __name__ == '__main__':
    BOPY = os.environ.get('BOPY', os.path.join(HERE, '..', 'optimization', 'BO.py'))
    src = open(BOPY).read()
    i = src.index('FEATURE_ORDER = ['); j = src.index(']', i)
    FEATS = [x.strip().strip("'\"") for x in src[i+len('FEATURE_ORDER = ['):j].replace('\n',' ').split(',') if x.strip()]

    fin = pd.read_csv(FIN_DATA)
    var = pd.read_csv(SSE_VARIANTS)[['name','sse_max','sse_p95','sse_mean_w']]
    df = fin.merge(var, left_on='filename', right_on='name', how='left')
    df['sse_recomputed'] = df['sse_max']

    results = {}; imps = {}
    todo = [('sse','sse_orig'), ('sse_recomputed','sse_max_new'),
            ('sse_p95','p95'), ('sse_mean_w','bzmean')]
    if ONLY: todo = [t for t in todo if t[1] == ONLY]
    for target, tag in todo:
        r, im = run_for_target(df, target, FEATS, tag)
        results[tag] = r; imps[tag] = im
        print('   -> R2 %.4f±%.4f  MAE %.1f meV  Spearman %.4f' % (r['r2_mean'], r['r2_std'], r['mae_mean']*1000, r['rho_mean']))
    sfx = ('_' + ONLY) if ONLY else ''
    json.dump(results, open(os.path.join(OUT,'summary%s.json' % sfx),'w'), indent=2)
    if len(imps) < 2:
        print('(single-target run; the ranking comparison is done after merging)'); raise SystemExit(0)
    print('\n=== comparison of the top-12 SHAP ranking ===')
    tab = pd.DataFrame({t: imps[t].set_index('feature')['rank'] for t in imps})
    tab['orig_rank'] = tab['sse_orig']
    print(tab.sort_values('orig_rank').head(12).to_string())
    tab.to_csv(os.path.join(OUT,'shap_rank_compare.csv'))
