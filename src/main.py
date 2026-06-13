"""
main.py
Pipeline orchestration for Windpower Lite.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np

from fetch_meteo import fetch_hourly_nasa_power
from physics_model import (
    air_density_from_ps_temp,
    adjust_wind_speed_power_law,
    compute_hourly_power_series,
    compute_aep_from_power_series,
)
from utils import load_turbine_specs
from hybrid_model import (
    load_model,
    predict_hybrid,
    load_hybrid_bundle,
    predict_hybrid_from_bundle,
)

# ── FIX 1 — Point to hybrid_all.joblib not hybrid_model.joblib ───────────────
DEFAULT_MODEL_PATH = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "hybrid_all.joblib"
    )
)


def _ensure_datetime(dt_obj: Optional[datetime]) -> Optional[datetime]:
    if dt_obj is None:
        return None
    if not isinstance(dt_obj, datetime):
        raise ValueError("Dates must be python datetime objects")
    return dt_obj


def prepare_site_dataframe(
    df_meteo: pd.DataFrame,
    turbine_spec: Dict[str, Any],
    ref_height_m: float = 50.0,
    shear_alpha: float = 0.14,
) -> pd.DataFrame:
    df = df_meteo.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    def _rho_row(row):
        p = row.get("surface_pressure_pa", None)
        t = row.get("temp_2m_c", None)
        try:
            return air_density_from_ps_temp(p, t)
        except Exception:
            print("[main] Warning: air density fallback to 1.225 kg/m³ for a row")
            return 1.225

    df["rho_kg_m3"] = df.apply(_rho_row, axis=1)

    if "wind_speed_50m_mps" not in df.columns:
        raise ValueError(
            "meteorological dataframe must include 'wind_speed_50m_mps'"
        )

    hub_h = float(turbine_spec.get("hub_height_m", ref_height_m))

    # ── FIX 2 — Pass 50m wind speed to compute_hourly_power_series ───────────
    # Previously v_hub_mps was computed here AND inside compute_hourly_power_series
    # causing double extrapolation. Now we pass the raw 50m wind speed and let
    # compute_hourly_power_series handle all extrapolation and cut-in logic.
    df["v_hub_mps"] = df["wind_speed_50m_mps"].apply(
        lambda v: adjust_wind_speed_power_law(v, ref_height_m, hub_h, alpha=shear_alpha)
    )

    rho_series = df["rho_kg_m3"].values

    # Pass raw 50m wind speed so compute_hourly_power_series applies
    # cut-in / cut-out at the correct hub-height speed
    power_w = compute_hourly_power_series(
        wind_speed_series_mps=df["wind_speed_50m_mps"].values,
        turbine_spec=turbine_spec,
        ref_height_m=ref_height_m,
        hub_height_m=hub_h,
        shear_alpha=shear_alpha,
        rho_series=rho_series,
        use_power_curve=False,
    )
    df["P_physics_w"] = power_w
    return df


def try_load_hybrid_model(model_path: str = DEFAULT_MODEL_PATH):
    model_path = os.path.normpath(model_path)
    if os.path.exists(model_path):
        try:
            try:
                bundle = load_hybrid_bundle(model_path)
                if isinstance(bundle, dict) and "models" in bundle:
                    return bundle
            except Exception:
                pass
            model = load_model(model_path)
            return model
        except Exception as e:
            print(f"[main] Warning: failed to load hybrid model at {model_path}: {e}")
            return None
    else:
        return None


def run_pipeline(
    lat: float = 8.4966,
    lon: float = 4.5421,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    turbine_name: Optional[str] = None,
    turbine_specs_csv: Optional[str] = None,
    apply_hybrid_if_available: bool = True,
    hybrid_model_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    start_dt = _ensure_datetime(start_dt)
    end_dt   = _ensure_datetime(end_dt)

    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365)

    specs = load_turbine_specs(turbine_specs_csv) if turbine_specs_csv else load_turbine_specs()

    if turbine_name is None:
        turbine_name = list(specs.keys())[0]
        print(f"[main] No turbine selected; using default: {turbine_name}")

    if turbine_name not in specs:
        raise ValueError(
            f"Turbine '{turbine_name}' not found in specs. "
            f"Available: {list(specs.keys())}"
        )

    turbine_spec = specs[turbine_name]

    print(
        f"[main] Fetching meteo data for {lat:.4f},{lon:.4f} "
        f"from {start_dt.date()} to {end_dt.date()} ..."
    )
    df_meteo = fetch_hourly_nasa_power(
        lat=lat, lon=lon, start_dt=start_dt, end_dt=end_dt
    )
    if df_meteo.empty:
        raise RuntimeError("No meteo data fetched.")

    df_site  = prepare_site_dataframe(
        df_meteo, turbine_spec, ref_height_m=50.0, shear_alpha=0.14
    )

    dt_hours        = 1.0
    aep_physics_kwh = compute_aep_from_power_series(
        df_site["P_physics_w"].values, dt_hours=dt_hours
    )
    hours      = float(len(df_site)) * dt_hours
    cf_physics = None
    try:
        cf_physics = aep_physics_kwh / (float(turbine_spec["rated_power_kw"]) * hours)
    except Exception:
        cf_physics = None

    summary = {
        "turbine":                  turbine_name,
        "lat":                      lat,
        "lon":                      lon,
        "start":                    start_dt.isoformat(),
        "end":                      end_dt.isoformat(),
        "rows":                     len(df_site),
        "hours":                    hours,
        "aep_physics_kwh":          float(aep_physics_kwh),
        "capacity_factor_physics":  float(cf_physics) if cf_physics is not None else None,
    }

    # ── Diagnostic — confirm cut-in is working ────────────────────────────────
    zero_hours = int((df_site["P_physics_w"] == 0).sum())
    print(f"[main] Hours P_physics_w == 0 : {zero_hours} "
          f"(cut-in active = {zero_hours > 0})")

    hybrid_model = None
    if apply_hybrid_if_available:
        model_path   = hybrid_model_path if hybrid_model_path else DEFAULT_MODEL_PATH
        hybrid_model = try_load_hybrid_model(model_path)

        if hybrid_model is not None:
            df_feat         = df_site.copy()
            df_feat["hour"] = df_feat["timestamp"].dt.hour
            df_feat["month"]= df_feat["timestamp"].dt.month

            feature_cols = ["P_physics_w", "v_hub_mps"]
            if "temp_2m_c" in df_feat.columns:
                feature_cols.append("temp_2m_c")
            if "surface_pressure_pa" in df_feat.columns:
                feature_cols.append("surface_pressure_pa")
            feature_cols += ["hour", "month"]

            for c in feature_cols:
                if c not in df_feat.columns:
                    df_feat[c] = 0.0

            try:
                max_output_w = float(turbine_spec["rated_power_kw"]) * 1000.0

                if isinstance(hybrid_model, dict) and "models" in hybrid_model:
                    df_site["P_hybrid_w"] = predict_hybrid_from_bundle(
                        hybrid_model,
                        turbine_name,
                        df_feat,
                        physics_power_col="P_physics_w",
                        max_output_w=max_output_w,
                    )
                else:
                    df_site["P_hybrid_w"] = predict_hybrid(
                        hybrid_model,
                        df_feat,
                        physics_power_col="P_physics_w",
                        feature_cols=feature_cols,
                        max_output_w=max_output_w,
                    )

                aep_hybrid_kwh = compute_aep_from_power_series(
                    df_site["P_hybrid_w"].values, dt_hours=1.0
                )
                cf_hybrid = aep_hybrid_kwh / (
                    float(turbine_spec["rated_power_kw"]) * hours
                )
                summary["aep_hybrid_kwh"]        = float(aep_hybrid_kwh)
                summary["capacity_factor_hybrid"] = float(cf_hybrid)
                summary["hybrid_model_used"]      = os.path.basename(model_path)

            except Exception as e:
                print(f"[main] Warning: hybrid model prediction failed: {e}")
        else:
            print("[main] No hybrid model found — running physics-only baseline.")
    else:
        print("[main] apply_hybrid_if_available=False — skipping hybrid.")

    return df_site, summary


def run_all_turbines(
    lat: float = 8.4966,
    lon: float = 4.5421,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    turbine_specs_csv: Optional[str] = None,
    apply_hybrid_if_available: bool = True,
    hybrid_model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Run the full pipeline for every turbine in turbine_specs.csv.
    Fetches NASA POWER data once and reuses it for all turbines.
    Returns a DataFrame with one row per turbine.
    """
    end_dt   = _ensure_datetime(end_dt)
    start_dt = _ensure_datetime(start_dt)
    if end_dt is None:
        end_dt = datetime.now(timezone.utc)
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365)

    specs = load_turbine_specs(turbine_specs_csv) if turbine_specs_csv else load_turbine_specs()

    print(
        f"[run_all_turbines] Fetching meteo for {lat:.4f},{lon:.4f} "
        f"from {start_dt.date()} to {end_dt.date()} ..."
    )
    df_meteo = fetch_hourly_nasa_power(
        lat=lat, lon=lon, start_dt=start_dt, end_dt=end_dt
    )
    if df_meteo.empty:
        raise RuntimeError("No meteo data fetched.")
    print(f"[run_all_turbines] {len(df_meteo):,} hourly records fetched.")

    hybrid_model = None
    if apply_hybrid_if_available:
        model_path   = hybrid_model_path if hybrid_model_path else DEFAULT_MODEL_PATH
        hybrid_model = try_load_hybrid_model(model_path)
        if hybrid_model is not None:
            print(f"[run_all_turbines] Hybrid model loaded: {os.path.basename(model_path)}")
        else:
            print("[run_all_turbines] No hybrid model — physics only.")

    results = []

    for turbine_name, turbine_spec in specs.items():
        print(f"[run_all_turbines] Processing: {turbine_name} ...")
        try:
            df_site = prepare_site_dataframe(
                df_meteo, turbine_spec, ref_height_m=50.0, shear_alpha=0.14
            )

            hours           = float(len(df_site))
            aep_physics_kwh = compute_aep_from_power_series(
                df_site["P_physics_w"].values, dt_hours=1.0
            )
            cf_physics = (
                aep_physics_kwh / (float(turbine_spec["rated_power_kw"]) * hours)
            ) * 100

            row = {
                "turbine":                     turbine_name,
                "rated_power_kw":              turbine_spec.get("rated_power_kw"),
                "rotor_diameter_m":            turbine_spec.get("rotor_diameter_m"),
                "hub_height_m":                turbine_spec.get("hub_height_m"),
                "aep_physics_kwh":             round(float(aep_physics_kwh), 0),
                "capacity_factor_physics_pct": round(float(cf_physics), 2),
                "aep_hybrid_kwh":              None,
                "capacity_factor_hybrid_pct":  None,
            }

            if hybrid_model is not None:
                df_feat          = df_site.copy()
                df_feat["hour"]  = df_feat["timestamp"].dt.hour
                df_feat["month"] = df_feat["timestamp"].dt.month

                feature_cols = ["P_physics_w", "v_hub_mps"]
                if "temp_2m_c" in df_feat.columns:
                    feature_cols.append("temp_2m_c")
                if "surface_pressure_pa" in df_feat.columns:
                    feature_cols.append("surface_pressure_pa")
                feature_cols += ["hour", "month"]

                for c in feature_cols:
                    if c not in df_feat.columns:
                        df_feat[c] = 0.0

                max_output_w = float(turbine_spec["rated_power_kw"]) * 1000.0

                try:
                    if isinstance(hybrid_model, dict) and "models" in hybrid_model:
                        df_site["P_hybrid_w"] = predict_hybrid_from_bundle(
                            hybrid_model, turbine_name, df_feat,
                            physics_power_col="P_physics_w",
                            max_output_w=max_output_w,
                        )
                    else:
                        df_site["P_hybrid_w"] = predict_hybrid(
                            hybrid_model, df_feat,
                            physics_power_col="P_physics_w",
                            feature_cols=feature_cols,
                            max_output_w=max_output_w,
                        )

                    aep_hybrid_kwh = compute_aep_from_power_series(
                        df_site["P_hybrid_w"].values, dt_hours=1.0
                    )
                    cf_hybrid = (
                        aep_hybrid_kwh / (float(turbine_spec["rated_power_kw"]) * hours)
                    ) * 100
                    row["aep_hybrid_kwh"]             = round(float(aep_hybrid_kwh), 0)
                    row["capacity_factor_hybrid_pct"] = round(float(cf_hybrid), 2)

                except Exception as e:
                    print(f"[run_all_turbines] Hybrid failed for {turbine_name}: {e}")

            results.append(row)
            print(
                f"  → AEP Physics: {row['aep_physics_kwh']:>12,.0f} kWh  "
                f"CF: {row['capacity_factor_physics_pct']:>6.2f}%  |  "
                f"AEP Hybrid: {row['aep_hybrid_kwh'] or 0:>12,.0f} kWh  "
                f"CF: {row['capacity_factor_hybrid_pct'] or 0:>6.2f}%"
            )

        except Exception as e:
            print(f"[run_all_turbines] ERROR on {turbine_name}: {e}")
            results.append({"turbine": turbine_name, "error": str(e)})

    return pd.DataFrame(results)


if __name__ == "__main__":
    import datetime as _dt
    end   = _dt.datetime(2023, 12, 31, tzinfo=timezone.utc)
    start = _dt.datetime(2023,  1,  1, tzinfo=timezone.utc)

    df_res, summ = run_pipeline(
        lat=8.4966, lon=4.5421,
        start_dt=start, end_dt=end,
        turbine_name="V90-2.0MW",
        apply_hybrid_if_available=True,
        hybrid_model_path=None,
    )
    print("SUMMARY:")
    for k, v in summ.items():
        print(f"  {k}: {v}")
    print("\nSample rows:")
    print(df_res[["timestamp", "wind_speed_50m_mps", "v_hub_mps", "P_physics_w"]].head(10))