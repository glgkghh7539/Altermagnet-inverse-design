#!/usr/bin/env python3
"""
Nested hyperparameter validation (reduced protocol) — Reviewer Point 3, §3.5.

Outer loop : GroupKFold(5) by parent  -> honest performance estimate
Inner loop : per outer-train set, Optuna TPE (N_TRIALS) over the SAME search
             space as the original 200-trial search, scored by GroupKFold(5)
             by parent on the outer-train set only.
Final fit  : best inner params + median inner best_iteration, trained on the
             full outer-train set, evaluated once on the untouched outer-test.

Outputs (in --outdir):
  nested_hp_outer_results.csv   one row per outer fold: params, inner score,
                                outer R2_log, best_iter, optimism
  nested_hp_summary.txt         pooled OOF R2_log / R2_eV, mean +/- sd,
                                comparison line vs published 0.7548
  nested_hp_oof.csv             filename, y_log, oof prediction

Parallelism: outer folds run as independent processes (joblib). Each fold's
Optuna study is stored in its own SQLite file, so a killed job RESUMES from
where it stopped (safe to requeue on TGM).

!! FILL IN: SEARCH_SPACE below mirrors the recovered optimum with standard
ranges. Before running, replace the ranges with the exact ones from the KPS
notebook hyperparameter-search cell so the inner protocol matches the
original 200-trial search verbatim.
"""
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = ['avg_bond_length','max_bond_length','min_bond_length','std_bond_length',
 'center_max_angle','center_min_angle','center_avg_angle','center_std_angle',
 'nonmag_max_angle','nonmag_min_angle','nonmag_std_angle',
 'labelled_1st','labelled_2nd','labelled_3rd','global_1st','global_2nd','global_3rd',
 'avg_long_axis','avg_short_axis','avg_axis_ratio','avg_s','avg_delta','motif0_nonmag_count',
 'magnetic_atomic_number','magnetic_electronegativity',
 'nonmagnetic_atomic_number','nonmagnetic_electronegativity',
 'hungarian_rotation_angle_deg','dimension','avg_motif_measure',
 'unit_cell_volume','packing_fraction','p_metric','p_metric_std',
 'd_orb_e','p_orb_e_non','d_lone_pair','proxy_M_magnet',
 'delta_chi','abs_delta_chi','delta_Z','abs_delta_Z','pd_ratio','ax_eq_gap',
 'bond_range','bond_cv','center_angle_spread','nonmag_angle_spread',
 'delta_chi_times_axeq','d_global_local_1st','d_global_local_2nd','d_global_local_3rd']

SEED_BASE = 1034          # keep consistent with stability_selection_100_parallel.py
N_OUTER   = 5
N_INNER   = 5
N_TRIALS  = 50            # reduced from the original 200
MAX_TREES = 5000
ES_ROUNDS = 100

# ------------------------------------------------------------------ FILL IN
# Replace with the exact ranges of the KPS-notebook Optuna search.
def SEARCH_SPACE(trial):
    return dict(
        learning_rate    = trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
        max_depth        = trial.suggest_int('max_depth', 3, 10),
        min_child_weight = trial.suggest_int('min_child_weight', 1, 10),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        reg_lambda       = trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        reg_alpha        = trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
    )
# ---------------------------------------------------------------------------

def load(csv):
    df = pd.read_csv(csv)
    suf = re.compile(r'_(?:x|y|z|st)(?:025|050|950|975|05)$')
    df['parent'] = df['filename'].str.replace(suf, '', regex=True)
    X = df[FEATURES].values
    y = np.log1p(df['sse'].values)
    return df, X, y

def inner_cv_score(params, X, y, groups, n_jobs):
    """Mean R2_log over inner GroupKFold; returns (score, median best_iter)."""
    gkf = GroupKFold(n_splits=N_INNER)
    scores, iters = [], []
    for tr, va in gkf.split(X, y, groups):
        m = xgb.XGBRegressor(**params, n_estimators=MAX_TREES,
                             objective='reg:squarederror', tree_method='hist',
                             early_stopping_rounds=ES_ROUNDS,
                             n_jobs=n_jobs, random_state=42)
        m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], verbose=False)
        scores.append(r2_score(y[va], m.predict(X[va])))
        iters.append(m.best_iteration + 1)
    return float(np.mean(scores)), int(np.median(iters))

def run_outer_fold(k, tr, te, df, X, y, outdir, n_jobs):
    storage = f'sqlite:///{outdir}/optuna_fold{k}.db'
    study = optuna.create_study(direction='maximize',
                                sampler=optuna.samplers.TPESampler(seed=SEED_BASE + k),
                                study_name=f'fold{k}', storage=storage,
                                load_if_exists=True)
    Xtr, ytr, gtr = X[tr], y[tr], df.iloc[tr]['parent'].values

    def objective(trial):
        params = SEARCH_SPACE(trial)
        score, best_iter = inner_cv_score(params, Xtr, ytr, gtr, n_jobs)
        trial.set_user_attr('best_iter', best_iter)
        return score

    remaining = N_TRIALS - len(study.trials)
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)

    best = study.best_trial
    params = {p: best.params[p] for p in best.params}
    n_est = best.user_attrs['best_iter']

    final = xgb.XGBRegressor(**params, n_estimators=n_est,
                             objective='reg:squarederror', tree_method='hist',
                             n_jobs=n_jobs, random_state=42)
    final.fit(Xtr, ytr, verbose=False)
    pred = final.predict(X[te])
    outer_r2 = r2_score(y[te], pred)

    row = dict(fold=k, inner_cv_r2=best.value, outer_r2_log=outer_r2,
               optimism=best.value - outer_r2, n_estimators=n_est, **params)
    return row, te, pred

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='fin_data.csv')
    ap.add_argument('--outdir', default='nested_hp_out')
    ap.add_argument('--n-procs', type=int, default=5,
                    help='outer folds run in parallel; set to 1 for serial')
    ap.add_argument('--xgb-jobs', type=int, default=8,
                    help='threads per xgboost fit (total cores ~= n_procs * xgb_jobs)')
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    df, X, y = load(a.csv)
    gkf = GroupKFold(n_splits=N_OUTER)
    splits = list(gkf.split(X, y, df['parent'].values))

    results = Parallel(n_jobs=a.n_procs)(
        delayed(run_outer_fold)(k, tr, te, df, X, y, a.outdir, a.xgb_jobs)
        for k, (tr, te) in enumerate(splits))

    rows, oof = [], np.full(len(y), np.nan)
    for row, te, pred in results:
        rows.append(row); oof[te] = pred
    res = pd.DataFrame(rows).sort_values('fold')
    res.to_csv(f'{a.outdir}/nested_hp_outer_results.csv', index=False)
    pd.DataFrame({'filename': df['filename'], 'y_log': y, 'oof': oof}
                 ).to_csv(f'{a.outdir}/nested_hp_oof.csv', index=False)

    r2_log = r2_score(y, oof)
    r2_ev  = r2_score(np.expm1(y), np.expm1(oof))
    mean, sd = res['outer_r2_log'].mean(), res['outer_r2_log'].std()
    opt_m, opt_s = res['optimism'].mean(), res['optimism'].std()
    with open(f'{a.outdir}/nested_hp_summary.txt', 'w') as f:
        f.write(f'Nested HP validation (outer {N_OUTER}-fold, inner {N_INNER}-fold, '
                f'{N_TRIALS} TPE trials/fold)\n'
                f'pooled OOF  R2_log = {r2_log:.4f}   R2_eV = {r2_ev:.4f}\n'
                f'outer folds R2_log = {mean:.4f} +/- {sd:.4f}\n'
                f'optimism (inner-CV minus outer) = {opt_m:.4f} +/- {opt_s:.4f}\n'
                f'published non-nested reference: R2_log 0.7548 (fold sd ~0.011)\n')
    print(open(f'{a.outdir}/nested_hp_summary.txt').read())
    print(res.to_string(index=False))

if __name__ == '__main__':
    main()
