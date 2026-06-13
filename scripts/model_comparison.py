"""
model_comparison.py
-------------------
Compare Random Forest against Gradient Boosting, Ridge Regression,
and SVR using 5-Fold Cross-Validation on the V90-2.0MW residual
learning task at Ilorin 2023.

Saves results to results/model_comparison.csv

Run:
    python scripts/model_comparison.py
"""

import os
import sys
import math
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from fetch_meteo import fetch_hourly_nasa_power
from physics_model import (
    air_density_from_ps_temp,
    adjust_wind_speed_power_law,
    power_from_power_curve,
)
from utils import load_turbine_specs

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
LAT         = 8.4966
LON         = 4.5421
START_DATE  = datetime(2023,  1,  1, tzinfo=timezone.utc)
END_DATE    = datetime(2023, 12, 31, tzinfo=timezone.utc)
TURBINE     = "V90-2.0MW"
SHEAR_ALPHA = 0.14
REF_HEIGHT  = 50.0


def main():
    print("=" * 65)
    print("  MODEL COMPARISON — WINDPOWER LITE")
    print("=" * 65)
    print(f"  Site    : Ilorin, Kwara State")
    print(f"  Turbine : {TURBINE}")
    print(f"  Period  : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  CV      : 5-Fold Cross-Validation")
    print("=" * 65 + "\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading meteorological data ...")
    df = fetch_hourly_nasa_power(
        lat=LAT, lon=LON,
        start_dt=START_DATE, end_dt=END_DATE,
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"]      = df["timestamp"].dt.hour
    df["month"]     = df["timestamp"].dt.month

    specs        = load_turbine_specs()
    turbine_spec = specs[TURBINE]

    hub_h      = float(turbine_spec.get("hub_height_m",   80))
    rated_w    = float(turbine_spec.get("rated_power_kw", 2000)) * 1000.0
    cut_in_ms  = float(turbine_spec.get("cut_in_ms",  turbine_spec.get("cut_in_mps",  3.5)))
    cut_out_ms = float(turbine_spec.get("cut_out_ms", turbine_spec.get("cut_out_mps", 25.0)))
    rotor_d    = float(turbine_spec.get("rotor_diameter_m", 90))
    A          = math.pi * (rotor_d / 2) ** 2

    # ── Compute features ──────────────────────────────────────────────────────
    print("Computing features and residual target ...")

    df["rho"] = df.apply(
        lambda r: air_density_from_ps_temp(
            r["surface_pressure_pa"], r["temp_2m_c"]
        ), axis=1
    )
    df["v_hub"] = df["wind_speed_50m_mps"].apply(
        lambda v: adjust_wind_speed_power_law(v, REF_HEIGHT, hub_h, SHEAR_ALPHA)
    )

    v = df["v_hub"].values
    rho = df["rho"].values

    # Physics baseline with cut-in / cut-out
    P_base = 0.5 * rho * A * v ** 3 * 0.40
    P_base = np.where(v < cut_in_ms,  0.0,     P_base)
    P_base = np.where(v > cut_out_ms, 0.0,     P_base)
    P_base = np.where(P_base > rated_w, rated_w, P_base)

    # Manufacturer power curve with cut-in / cut-out
    power_curve_df = turbine_spec.get("power_curve")
    P_curve = np.array([
        power_from_power_curve(vi, power_curve_df) for vi in v
    ])
    P_curve = np.where(v < cut_in_ms,  0.0,     P_curve)
    P_curve = np.where(v > cut_out_ms, 0.0,     P_curve)
    P_curve = np.where(P_curve > rated_w, rated_w, P_curve)

    # 3% noise — same seed as training for consistency
    noise_std      = 0.03 * rated_w
    rng            = np.random.default_rng(seed=42)
    noise          = rng.normal(0.0, noise_std, len(P_curve))
    operating_mask = (v >= cut_in_ms) & (v <= cut_out_ms)
    P_curve_noisy  = P_curve.copy().astype(float)
    P_curve_noisy[operating_mask] += noise[operating_mask]
    P_curve_noisy  = np.clip(P_curve_noisy, 0.0, rated_w)

    # Residual target
    y_vals = P_curve_noisy - P_base

    # Feature matrix
    X = pd.DataFrame({
        "P_physics_w":         P_base,
        "v_hub_mps":           v,
        "temp_2m_c":           df["temp_2m_c"].values,
        "surface_pressure_pa": df["surface_pressure_pa"].values,
        "hour":                df["hour"].values,
        "month":               df["month"].values,
    })
    y = pd.Series(y_vals)

    # Drop incomplete rows
    mask = X.notna().all(axis=1) & y.notna()
    X    = X[mask].reset_index(drop=True)
    y    = y[mask].reset_index(drop=True)

    print(f"  Training samples : {len(X)}")
    print(f"  Features         : {list(X.columns)}\n")

    # ── Model definitions ─────────────────────────────────────────────────────
    models = {
        "Random Forest":     RandomForestRegressor(
                                 n_estimators=200,
                                 max_depth=15,
                                 random_state=42,
                                 n_jobs=-1,
                             ),
        "Gradient Boosting": GradientBoostingRegressor(
                                 n_estimators=200,
                                 max_depth=5,
                                 learning_rate=0.05,
                                 random_state=42,
                             ),
        "Ridge Regression":  Ridge(alpha=1.0),
        "SVR":               SVR(kernel="rbf", C=100, epsilon=0.1),
    }

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("Running 5-Fold Cross-Validation for each model ...")
    print(f"  {'Model':<22} {'RMSE (W)':>12}  {'MAE (W)':>12}  {'R²':>8}")
    print(f"  {'-'*22} {'-'*12}  {'-'*12}  {'-'*8}")

    rows = []
    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model),
        ])
        scores = cross_validate(
            pipeline, X, y,
            cv=5,
            scoring=[
                "neg_root_mean_squared_error",
                "neg_mean_absolute_error",
                "r2",
            ],
            n_jobs=-1,
        )
        rmse = -scores["test_neg_root_mean_squared_error"].mean()
        mae  = -scores["test_neg_mean_absolute_error"].mean()
        r2   =  scores["test_r2"].mean()

        rows.append({
            "Model":    name,
            "RMSE (W)": round(rmse, 2),
            "MAE (W)":  round(mae,  2),
            "R²":       round(r2,   4),
        })
        print(f"  {name:<22} {rmse:>12,.2f}  {mae:>12,.2f}  {r2:>8.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    df_results = pd.DataFrame(rows)

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)

    best_r2   = df_results.loc[df_results["R²"].idxmax()]
    best_rmse = df_results.loc[df_results["RMSE (W)"].idxmin()]

    print(f"\n  Best R²   : {best_r2['Model']:<22} R²={best_r2['R²']:.4f}")
    print(f"  Best RMSE : {best_rmse['Model']:<22} RMSE={best_rmse['RMSE (W)']:,.2f} W")

    if best_r2["Model"] == "Random Forest":
        print("\n  ✓ Random Forest is the best model — selection justified empirically.")
    else:
        print(f"\n  ⚠ {best_r2['Model']} outperforms Random Forest.")
        print("    Consider switching to this model or documenting the trade-off.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "model_comparison.csv")
    df_results.to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()