"""
benchmark_windpowerlib.py

Three-way comparison:
  Windpower Lite Physics  vs  windpowerlib  vs  Windpower Lite Hybrid

Quantifies the hybrid correction uplift and benchmarks both models
against the windpowerlib standard power curve implementation.

Usage:
    python scripts/benchmark_windpowerlib.py
"""

import os
import sys
from datetime import datetime
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import run_pipeline
from fetch_meteo import fetch_hourly_nasa_power

try:
    from windpowerlib import WindTurbine, ModelChain, get_turbine_types
except ImportError:
    print("ERROR: windpowerlib not installed.  Run: pip install windpowerlib")
    sys.exit(1)


# ── CONFIGURATION ─────────────────────────────────────────────────────────────
SITE_NAME           = "Ilorin, Kwara State"
LATITUDE            = 8.4966
LONGITUDE           = 4.5421
START_DATE          = datetime(2023, 1, 1)
END_DATE            = datetime(2023, 12, 31)
WPL_TURBINE_NAME    = "V90-2.0MW"   # Windpower Lite turbine name
WINDPOWERLIB_TURBINE= "V90/2000"    # windpowerlib database name
HUB_HEIGHT_M        = 80
ROUGHNESS_LENGTH    = 0.03          # Guinea Savannah open terrain


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 75)
    print("  WINDPOWER LITE vs WINDPOWERLIB — THREE-WAY BENCHMARK")
    print("=" * 75)
    print(f"  Site    : {SITE_NAME}")
    print(f"  Coords  : {LATITUDE}°N, {LONGITUDE}°E")
    print(f"  Period  : {START_DATE.date()} → {END_DATE.date()}")
    print(f"  Turbine : {WPL_TURBINE_NAME} / {WINDPOWERLIB_TURBINE}  "
          f"(hub {HUB_HEIGHT_M} m)")
    print("=" * 75 + "\n")

    # ── STEP 1 — Windpower Lite physics baseline ──────────────────────────────
    print("[1] Windpower Lite — physics baseline only ...")
    df_phys, summ_phys = run_pipeline(
        lat=LATITUDE, lon=LONGITUDE,
        start_dt=START_DATE, end_dt=END_DATE,
        turbine_name=WPL_TURBINE_NAME,
        apply_hybrid_if_available=False,
    )
    aep_physics = summ_phys["aep_physics_kwh"]
    cf_physics  = summ_phys["capacity_factor_physics"]

    # Diagnostic — confirm cut-in fix is active
    zero_h = int((df_phys["P_physics_w"] == 0).sum())
    pos_h  = int((df_phys["P_physics_w"] >  0).sum())
    print(f"  AEP Physics         : {aep_physics:>12,.0f} kWh")
    print(f"  CF  Physics         : {cf_physics*100:>12.2f}%")
    print(f"  Hours P_physics == 0: {zero_h}  "
          f"({'cut-in active ✓' if zero_h > 0 else 'cut-in NOT active ✗'})")
    print(f"  Hours P_physics  > 0: {pos_h}\n")

    # ── STEP 2 — Windpower Lite hybrid ────────────────────────────────────────
    print("[2] Windpower Lite — hybrid correction ...")
    df_hyb, summ_hyb = run_pipeline(
        lat=LATITUDE, lon=LONGITUDE,
        start_dt=START_DATE, end_dt=END_DATE,
        turbine_name=WPL_TURBINE_NAME,
        apply_hybrid_if_available=True,
    )
    aep_hybrid = summ_hyb.get("aep_hybrid_kwh")
    cf_hybrid  = summ_hyb.get("capacity_factor_hybrid")
    model_used = summ_hyb.get("hybrid_model_used", "unknown")

    if aep_hybrid:
        uplift = (aep_hybrid - aep_physics) / aep_physics * 100
        print(f"  AEP Hybrid          : {aep_hybrid:>12,.0f} kWh")
        print(f"  CF  Hybrid          : {cf_hybrid*100:>12.2f}%")
        print(f"  Model used          : {model_used}")
        print(f"  Uplift over physics : {uplift:>+12.1f}%\n")
    else:
        print("  Hybrid model not available\n")
        uplift = None

    # ── STEP 3 — Fetch meteo for windpowerlib ─────────────────────────────────
    print("[3] Fetching meteorological data (cached) ...")
    df_meteo = fetch_hourly_nasa_power(
        lat=LATITUDE, lon=LONGITUDE,
        start_dt=START_DATE, end_dt=END_DATE,
    )
    print(f"  Records : {len(df_meteo):,}\n")

    # ── STEP 4 — Build windpowerlib weather DataFrame ─────────────────────────
    # windpowerlib requires MultiIndex columns: (variable_name, height)
    print("[4] Preparing windpowerlib weather DataFrame ...")
    weather_data = {
        ("wind_speed",      50): df_meteo["wind_speed_50m_mps"].values,
        ("temperature",      2): df_meteo["temp_2m_c"].values + 273.15,
        ("pressure",         0): df_meteo["surface_pressure_pa"].values / 100,
        ("roughness_length", 0): np.full(len(df_meteo), ROUGHNESS_LENGTH),
    }
    df_weather         = pd.DataFrame(
        weather_data,
        index=pd.to_datetime(df_meteo["timestamp"].values)
    )
    df_weather.columns = pd.MultiIndex.from_tuples(df_weather.columns)
    df_weather.index.name = "timestamp"
    print(f"  Shape : {df_weather.shape}\n")

    # ── STEP 5 — Run windpowerlib ─────────────────────────────────────────────
    print("[5] Running windpowerlib ModelChain ...")
    aep_wpl_lib = None
    cf_wpl_lib  = None

    try:
        turbine = WindTurbine(
            turbine_type=WINDPOWERLIB_TURBINE,
            hub_height=HUB_HEIGHT_M,
        )
        print(f"  Turbine loaded : {WINDPOWERLIB_TURBINE}")
        print(f"  Rated power    : {turbine.nominal_power / 1000:,.0f} kW")

        mc = ModelChain(power_plant=turbine)
        mc.run_model(df_weather)

        power_kw    = mc.power_output / 1000.0
        aep_wpl_lib = float(power_kw.sum())
        rated_kw    = turbine.nominal_power / 1000.0
        hours       = len(df_weather)
        cf_wpl_lib  = aep_wpl_lib / (rated_kw * hours)

        print(f"  AEP windpowerlib : {aep_wpl_lib:>12,.0f} kWh")
        print(f"  CF  windpowerlib : {cf_wpl_lib*100:>12.2f}%\n")

    except Exception as e:
        print(f"  ERROR running windpowerlib: {e}")
        import traceback
        traceback.print_exc()

    # ── STEP 6 — Three-way comparison table ───────────────────────────────────
    print("\n" + "=" * 75)
    print("  THREE-WAY COMPARISON SUMMARY")
    print("=" * 75)
    print(f"\n  {'Model':<35} {'AEP (kWh)':>12}  {'CF (%)':>8}")
    print(f"  {'-'*35} {'-'*12}  {'-'*8}")
    print(f"  {'Windpower Lite — Physics':<35} {aep_physics:>12,.0f}  "
          f"{cf_physics*100:>8.2f}")
    if aep_wpl_lib:
        print(f"  {'windpowerlib (standard curve)':<35} {aep_wpl_lib:>12,.0f}  "
              f"{cf_wpl_lib*100:>8.2f}")
    if aep_hybrid:
        print(f"  {'Windpower Lite — Hybrid':<35} {aep_hybrid:>12,.0f}  "
              f"{cf_hybrid*100:>8.2f}")

    # ── STEP 7 — Gap analysis ─────────────────────────────────────────────────
    if aep_wpl_lib:
        print(f"\n  {'Gap analysis':}")
        phys_vs_lib = (aep_physics  - aep_wpl_lib) / aep_wpl_lib * 100
        print(f"  Physics  vs windpowerlib : {phys_vs_lib:>+.1f}%")
        if aep_hybrid:
            hyb_vs_lib  = (aep_hybrid   - aep_wpl_lib) / aep_wpl_lib * 100
            hyb_vs_phys = (aep_hybrid   - aep_physics) / aep_physics * 100
            print(f"  Hybrid   vs windpowerlib : {hyb_vs_lib:>+.1f}%")
            print(f"  Hybrid   vs Physics      : {hyb_vs_phys:>+.1f}%")

    # ── STEP 8 — Diagnosis ────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  DIAGNOSIS")
    print("=" * 75)
    if aep_wpl_lib and aep_hybrid:
        hyb_vs_lib = (aep_hybrid - aep_wpl_lib) / aep_wpl_lib * 100
        if abs(hyb_vs_lib) <= 15:
            print("\n  ✓ Hybrid and windpowerlib are within 15%.")
            print("    The hybrid is correctly learning to replicate")
            print("    manufacturer power curve behaviour.")
            print("    The uplift is justified — it corrects the conservative")
            print("    Cp=0.40 baseline toward realistic turbine output.")
        elif hyb_vs_lib > 15:
            print(f"\n  ⚠ Hybrid exceeds windpowerlib by {hyb_vs_lib:.1f}%.")
            print("    The hybrid may still be over-correcting.")
            print("    Consider reviewing the noise level in training targets.")
        else:
            print(f"\n  ⚠ Hybrid is {abs(hyb_vs_lib):.1f}% below windpowerlib.")
            print("    The hybrid is under-correcting.")
            print("    Review feature matrix and residual target generation.")

        phys_vs_lib = (aep_physics - aep_wpl_lib) / aep_wpl_lib * 100
        if phys_vs_lib > 5:
            print(f"\n  ⚠ Physics baseline still {phys_vs_lib:.1f}% above windpowerlib.")
            print("    Cut-in constraint may not be fully active.")
            print("    Check 'Hours P_physics == 0' in Step 1 output above.")
        elif phys_vs_lib < -5:
            print(f"\n  ✓ Physics baseline is {abs(phys_vs_lib):.1f}% below windpowerlib.")
            print("    Conservative Cp=0.40 baseline is working as intended.")
        else:
            print(f"\n  ✓ Physics baseline within 5% of windpowerlib.")

    # ── STEP 9 — Save results ─────────────────────────────────────────────────
    rows = [{"Model": "Windpower Lite — Physics",
             "AEP_kWh": round(aep_physics, 0),
             "CF_pct":  round(cf_physics * 100, 2)}]
    if aep_wpl_lib:
        rows.append({"Model": "windpowerlib",
                     "AEP_kWh": round(aep_wpl_lib, 0),
                     "CF_pct":  round(cf_wpl_lib * 100, 2)})
    if aep_hybrid:
        rows.append({"Model": "Windpower Lite — Hybrid",
                     "AEP_kWh": round(aep_hybrid, 0),
                     "CF_pct":  round(cf_hybrid * 100, 2)})

    out_dir  = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "benchmark_windpowerlib.csv")
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\n  Results saved → {out_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()