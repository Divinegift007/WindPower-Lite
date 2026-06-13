#!/usr/bin/env python3
"""evaluate_holdout.py
Run a simple holdout evaluation of the saved hybrid model against pseudo-targets
(created from manufacturer power curve) using `windpower_results.csv`.
Outputs RMSE / MAE / R2 and saves a small CSV with predictions for the holdout set.
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "windpower_results.csv"
MODEL_PATH = ROOT / "models" / "hybrid_model.joblib"
OUT_DIR = ROOT / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} not found. Run pipeline first to produce results CSV.")
    sys.exit(2)

if not MODEL_PATH.exists():
    print(f"Error: model not found at {MODEL_PATH}. Train the hybrid model first.")
    sys.exit(2)

sys.path.insert(0, str(ROOT / 'src'))
from utils import load_turbine_specs
from hybrid_model import (
    make_pseudo_targets_from_power_curve,
    prepare_features,
    load_model,
    train_hybrid_model,
)

# load data
df = pd.read_csv(CSV_PATH, parse_dates=['timestamp'])
# load turbine specs (use first turbine in file)
specs = load_turbine_specs()
_tname = list(specs.keys())[0]
T = specs[_tname]

# ensure v_hub_mps present
if 'v_hub_mps' not in df.columns:
    raise RuntimeError('v_hub_mps missing from CSV; cannot create pseudo-targets')

# create pseudo targets from power curve
pc = T.get('power_curve')
if pc is None:
    raise RuntimeError(f'No power_curve for turbine {_tname}')

df2 = make_pseudo_targets_from_power_curve(df, pc, v_col='v_hub_mps', out_col='P_true_w')
# compute physics column presence
if 'P_physics_w' not in df2.columns:
    raise RuntimeError('P_physics_w missing; run physics pipeline first')

# add time features
if 'hour' not in df2.columns:
    df2['hour'] = df2['timestamp'].dt.hour
if 'month' not in df2.columns:
    df2['month'] = df2['timestamp'].dt.month

# split: use last 20% as holdout (time-ordered)
n = len(df2)
hold_n = max(1, int(n * 0.2))
train_idx = slice(0, n - hold_n)
hold_idx = slice(n - hold_n, n)

train = df2.iloc[train_idx].reset_index(drop=True)
hold = df2.iloc[hold_idx].reset_index(drop=True)

# prepare features for train and holdout
X_tr, y_tr, feature_cols = prepare_features(train, physics_power_col='P_physics_w', v_col='v_hub_mps', temp_col='temp_2m_c', pressure_col='surface_pressure_pa', time_cols=['hour','month'])
X_hold, y_hold, _ = prepare_features(hold, physics_power_col='P_physics_w', v_col='v_hub_mps', temp_col='temp_2m_c', pressure_col='surface_pressure_pa', time_cols=['hour','month'])

# Retrain a fresh model on the training split so holdout is truly unseen
print('Training model on training split for holdout evaluation...')
model, train_metrics = train_hybrid_model(X_tr, y_tr, model_type='rf', cv_folds=5, n_jobs=-1)
print('CV RMSE (train): {:.4f}, MAE: {:.4f}, R2: {:.4f}'.format(train_metrics['rmse_cv_mean'], train_metrics['mae_cv_mean'], train_metrics['r2_cv_mean']))

# predict residuals and hybrid power on holdout
pred_res = model.predict(X_hold)
P_phys = hold['P_physics_w'].values
P_pred = P_phys + pred_res
P_pred = np.where(np.isnan(P_pred), np.nan, np.maximum(0.0, P_pred))

# metrics against pseudo-targets (P_true_w)
y_true = hold['P_true_w'].values
rmse = np.sqrt(mean_squared_error(y_true, P_pred))
mae = mean_absolute_error(y_true, P_pred)
r2 = r2_score(y_true, P_pred)

print('Holdout evaluation (last {:.0f}% of data):'.format(hold_n / n * 100))
print(f'Rows holdout: {len(hold)}')
print(f'RMSE (W): {rmse:.3f}')
print(f'MAE  (W): {mae:.3f}')
print(f'R2       : {r2:.4f}')

# save predictions
out_df = hold[['timestamp','v_hub_mps','P_physics_w','P_true_w']].copy()
out_df['P_hybrid_pred_w'] = P_pred
out_csv = OUT_DIR / 'holdout_predictions.csv'
out_df.to_csv(out_csv, index=False)
print('Saved holdout predictions to', out_csv)

print('\nFeature columns used:', feature_cols)

sys.exit(0)
