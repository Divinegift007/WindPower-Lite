"""
train_hybrid.py

Train the hybrid ML correction model for Windpower Lite.

This script:
  - Loads turbine specs
  - Fetches 12 months of historical NASA POWER meteo data
  - Computes hub-height wind speed + physics baseline power
  - Generates pseudo-true power using manufacturer power curve
    with cut-in / cut-out constraints and 3% realistic noise
  - Creates features and residual targets
  - Trains Random Forest model
  - Saves final hybrid model bundle to: ../models/hybrid_all.joblib

Run:
    python src/train_hybrid.py
"""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np
import joblib

from fetch_meteo import fetch_hourly_nasa_power
from physics_model import (
    air_density_from_ps_temp,
    adjust_wind_speed_power_law,
    compute_hourly_power_series,
    power_from_power_curve,
)
from hybrid_model import (
    prepare_features,
    train_hybrid_model,
)
from utils import load_turbine_specs


# ── SETTINGS ──────────────────────────────────────────────────────────────────
LAT        = 8.4966
LON        = 4.5421
MONTHS     = 12
REF_HEIGHT = 50.0
SHEAR_ALPHA = 0.14

MODEL_ALL_PATH = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models", "hybrid_all.joblib")
)


# ── FIXED PSEUDO-TARGET GENERATOR ────────────────────────────────────────────

def make_pseudo_targets_from_power_curve(
    df: pd.DataFrame,
    turbine_spec: dict,
    v_col: str = "v_hub_mps",
    out_col: str = "P_true_w",
) -> pd.DataFrame:
    """
    Generate pseudo-true training targets from the manufacturer power curve.

    Fixes applied vs original:
      1. Cut-in and cut-out enforced on both P_base and P_curve so the
         physics baseline never produces phantom power at low wind speeds.
      2. 3% rated-power Gaussian noise added to P_curve while turbine is
         operating — breaks the deterministic wind_speed → power mapping
         that caused R² = 1.0000 and forces the Random Forest to learn
         genuine patterns rather than memorise a lookup table.
         Noise level is consistent with real SCADA variability reported
         by Pei & Li (2019).
    """
    df = df.copy()
    v_hub      = df[v_col].values
    rated_w    = float(turbine_spec.get("rated_power_kw", 2000)) * 1000.0
    cut_in_ms  = float(turbine_spec.get("cut_in_ms",  turbine_spec.get("cut_in_mps",  3.5)))
    cut_out_ms = float(turbine_spec.get("cut_out_ms", turbine_spec.get("cut_out_mps", 25.0)))

    # Manufacturer power curve output
    power_curve_df = turbine_spec.get("power_curve", None)
    if power_curve_df is not None and isinstance(power_curve_df, pd.DataFrame):
        P_curve = np.array([
            power_from_power_curve(v, power_curve_df) for v in v_hub
        ])
    else:
        # Fallback — simple cubic scaling to rated power
        rated_ws = float(turbine_spec.get("rated_mps", 12.0))
        P_curve  = np.clip((v_hub / rated_ws) ** 3 * rated_w, 0.0, rated_w)

    # Enforce cut-in and cut-out on P_curve
    P_curve = np.where(v_hub < cut_in_ms,  0.0,     P_curve)
    P_curve = np.where(v_hub > cut_out_ms, 0.0,     P_curve)
    P_curve = np.where(P_curve > rated_w,  rated_w, P_curve)

    # ── FIX 2 — Add 3% realistic noise to break determinism ──────────────
    # Simulates real turbine variability: turbulence, blade wear, yaw
    # misalignment. Noise only applied while turbine is operating.
    # Fixed seed = 42 ensures reproducibility across training runs.
    noise_std      = 0.03 * rated_w
    rng            = np.random.default_rng(seed=42)
    noise          = rng.normal(loc=0.0, scale=noise_std, size=len(P_curve))
    operating_mask = (v_hub >= cut_in_ms) & (v_hub <= cut_out_ms)
    P_curve_noisy  = P_curve.copy().astype(float)
    P_curve_noisy[operating_mask] += noise[operating_mask]
    P_curve_noisy  = np.clip(P_curve_noisy, 0.0, rated_w)
    # ─────────────────────────────────────────────────────────────────────

    df[out_col] = P_curve_noisy
    return df


# ── MAIN TRAINING LOOP ────────────────────────────────────────────────────────

def main():
    print("\n=== TRAINING HYBRID MODEL (ONE MODEL PER TURBINE) ===\n")

    specs         = load_turbine_specs()
    turbine_items = list(specs.items())
    if not turbine_items:
        raise RuntimeError("No turbines found in specs; cannot train.")

    print(f"Found {len(turbine_items)} turbine(s) — bundling into: {MODEL_ALL_PATH}")

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=30 * MONTHS)

    print(f"Fetching meteo data from {start_dt.date()} to {end_dt.date()} …")
    df_meteo = fetch_hourly_nasa_power(
        lat=LAT, lon=LON, start_dt=start_dt, end_dt=end_dt
    )
    if df_meteo.empty:
        raise RuntimeError("No meteo data fetched. Cannot train.")

    df_base                = df_meteo.copy()
    df_base["timestamp"]   = pd.to_datetime(df_base["timestamp"])
    df_base                = df_base.sort_values("timestamp").reset_index(drop=True)
    df_base["rho_kg_m3"]   = df_base.apply(
        lambda r: air_density_from_ps_temp(
            r.get("surface_pressure_pa"), r.get("temp_2m_c")
        ), axis=1
    )

    all_models   = {}
    all_features = {}
    all_metrics  = {}
    specs_snapshot = {}

    for turbine_name, turbine in turbine_items:
        print(f"\n--- Training for turbine: {turbine_name} ---")
        df    = df_base.copy()
        hub_h = float(turbine.get("hub_height_m", REF_HEIGHT))

        df["v_hub_mps"] = df["wind_speed_50m_mps"].apply(
            lambda v: adjust_wind_speed_power_law(v, REF_HEIGHT, hub_h, alpha=SHEAR_ALPHA)
        )

        # Physics baseline — now with cut-in / cut-out enforced via Fix 1
        df["P_physics_w"] = compute_hourly_power_series(
            wind_speed_series_mps=df["v_hub_mps"].values,
            turbine_spec=turbine,
            ref_height_m=REF_HEIGHT,
            hub_height_m=hub_h,
            shear_alpha=SHEAR_ALPHA,
            rho_series=df["rho_kg_m3"].values,
            use_power_curve=False,
        )

        # Training target — now with cut-in / cut-out and noise via Fix 2
        df = make_pseudo_targets_from_power_curve(
            df, turbine, v_col="v_hub_mps", out_col="P_true_w"
        )

        before_drop = len(df)
        df = df.dropna(subset=["P_physics_w", "P_true_w", "v_hub_mps"])
        after_drop = len(df)
        if after_drop < before_drop:
            print(
                f"Dropped {before_drop - after_drop} incomplete rows "
                f"for turbine: {turbine_name}"
            )

        df["hour"]  = df["timestamp"].dt.hour
        df["month"] = df["timestamp"].dt.month

        X, y, feature_cols = prepare_features(
            df,
            physics_power_col="P_physics_w",
            v_col="v_hub_mps",
            temp_col="temp_2m_c",
            pressure_col="surface_pressure_pa",
            time_cols=["hour", "month"],
        )

        print(f"Features used: {feature_cols}")
        print(f"Training samples: {len(X)}")

        model, metrics = train_hybrid_model(
            X, y, model_type="gbr", cv_folds=5, n_jobs=-1
        )

        print(
            f"Cross-val  RMSE: {metrics['rmse_cv_mean']:.4f}  "
            f"MAE: {metrics['mae_cv_mean']:.4f}  "
            f"R²: {metrics['r2_cv_mean']:.4f}"
        )

        all_models[turbine_name]    = model
        all_features[turbine_name]  = feature_cols
        all_metrics[turbine_name]   = metrics
        specs_snapshot[turbine_name] = {
            "rated_power_kw":   float(turbine.get("rated_power_kw", 0)),
            "rotor_diameter_m": float(turbine.get("rotor_diameter_m", 0)),
            "hub_height_m":     float(turbine.get("hub_height_m", REF_HEIGHT)),
        }

    os.makedirs(os.path.dirname(MODEL_ALL_PATH), exist_ok=True)
    artifact = {
        "models":       all_models,
        "feature_cols": all_features,
        "metrics":      all_metrics,
        "specs":        specs_snapshot,
    }
    joblib.dump(artifact, MODEL_ALL_PATH)
    print(f"\nSaved bundled hybrid models → {MODEL_ALL_PATH}")


if __name__ == "__main__":
    main()