"""
physics_model.py
Physics-based baseline functions for Windpower Lite.
"""

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

# Physical constants
R_AIR   = 287.05  # J/(kg·K)
RHO_STD = 1.225   # kg/m³


def air_density_from_ps_temp(pressure_pa: float, temp_c: float) -> float:
    if pressure_pa is None or np.isnan(pressure_pa):
        return RHO_STD
    if temp_c is None or np.isnan(temp_c):
        temp_c = 15.0
    t_k = float(temp_c) + 273.15
    rho = float(pressure_pa) / (R_AIR * t_k)
    return float(rho)


def rotor_swept_area(diameter_m: float) -> float:
    return np.pi * (float(diameter_m) / 2.0) ** 2.0


def adjust_wind_speed_power_law(
    v_ref: float, h_ref: float, h_target: float, alpha: float = 0.14
) -> float:
    if v_ref is None or np.isnan(v_ref):
        return np.nan
    return float(v_ref) * (float(h_target) / float(h_ref)) ** float(alpha)


def adjust_wind_speed_log_profile(
    v_ref: float, h_ref: float, h_target: float, z0: float = 0.03
) -> float:
    if v_ref is None or np.isnan(v_ref):
        return np.nan
    if h_ref <= z0 or h_target <= z0:
        return float(v_ref)
    return float(v_ref) * (
        np.log(float(h_target) / z0) / np.log(float(h_ref) / z0)
    )


def power_from_power_curve(v: float, power_curve: pd.DataFrame) -> float:
    if power_curve is None or power_curve.empty:
        return np.nan
    pc   = power_curve.sort_values("wind_speed_mps").reset_index(drop=True)
    ws   = pc["wind_speed_mps"].values.astype(float)
    p_kw = pc["power_kw"].values.astype(float)
    if v <= ws[0]:
        return 0.0
    if v >= ws[-1]:
        return float(p_kw[-1]) * 1000.0
    p_interp_kw = np.interp(v, ws, p_kw)
    return float(p_interp_kw) * 1000.0


def physics_power_from_cp(
    v: float,
    rho: float,
    rotor_diameter_m: float,
    cp: float = 0.4,
    eta_sys: float = 0.9,
) -> float:
    if v is None or np.isnan(v):
        return np.nan
    A      = rotor_swept_area(rotor_diameter_m)
    P_wind = 0.5 * float(rho) * float(A) * (float(v) ** 3.0)
    P_elec = P_wind * float(cp) * float(eta_sys)
    return float(P_elec)


def baseline_turbine_power(
    v_hub_mps: float,
    turbine_spec: Dict[str, Any],
    rho: float = RHO_STD,
    use_power_curve: bool = True,
) -> float:
    rated_kw  = float(turbine_spec.get("rated_power_kw"))
    rated_w   = rated_kw * 1000.0
    # Support both key naming conventions
    cut_in    = float(turbine_spec.get("cut_in_ms",  turbine_spec.get("cut_in_mps",  3.5)))
    cut_out   = float(turbine_spec.get("cut_out_ms", turbine_spec.get("cut_out_mps", 25.0)))
    rated_ws  = float(turbine_spec.get("rated_mps",  12.0))

    if v_hub_mps is None or np.isnan(v_hub_mps):
        return np.nan
    if v_hub_mps < cut_in:
        return 0.0
    if v_hub_mps >= cut_out:
        return 0.0

    if (
        use_power_curve
        and "power_curve" in turbine_spec
        and isinstance(turbine_spec["power_curve"], pd.DataFrame)
    ):
        p_w = power_from_power_curve(v_hub_mps, turbine_spec["power_curve"])
        return min(p_w, rated_w)

    rotor_d = turbine_spec.get("rotor_diameter_m", None)
    if rotor_d is None:
        raise ValueError(
            "turbine_spec must provide 'rotor_diameter_m' if no power_curve is present"
        )
    cp  = float(turbine_spec.get("cp",      0.4))
    eta = float(turbine_spec.get("eta_sys", 0.9))
    p_w = physics_power_from_cp(v_hub_mps, rho, rotor_d, cp=cp, eta_sys=eta)
    if p_w > rated_w:
        p_w = rated_w
    return float(p_w)


def compute_hourly_power_series(
    wind_speed_series_mps,
    turbine_spec: Dict[str, Any],
    ref_height_m: float = 50.0,
    hub_height_m: Optional[float] = None,
    shear_alpha: float = 0.14,
    rho_series: Optional[np.ndarray] = None,
    use_power_curve: bool = True,
    shear_method: str = "power_law",
) -> np.ndarray:
    ws = np.asarray(wind_speed_series_mps, dtype=float)
    n  = ws.size

    if hub_height_m is None:
        hub_height_m = float(turbine_spec.get("hub_height_m", ref_height_m))

    if rho_series is None:
        rho_arr = np.full(n, RHO_STD, dtype=float)
    else:
        rho_arr = np.asarray(rho_series, dtype=float)
        if rho_arr.shape[0] != n:
            raise ValueError(
                "rho_series must have same length as wind_speed_series_mps"
            )

    # ── FIX 1 — Read cut-in and cut-out once, apply vectorised ───────────
    cut_in_ms  = float(turbine_spec.get("cut_in_ms",  turbine_spec.get("cut_in_mps",  3.5)))
    cut_out_ms = float(turbine_spec.get("cut_out_ms", turbine_spec.get("cut_out_mps", 25.0)))
    rated_w    = float(turbine_spec.get("rated_power_kw", 2000)) * 1000.0
    # ─────────────────────────────────────────────────────────────────────

    power_out = np.zeros(n, dtype=float)

    for i in range(n):
        v_ref = ws[i]
        if shear_method == "power_law":
            v_hub = adjust_wind_speed_power_law(
                v_ref, ref_height_m, hub_height_m, alpha=shear_alpha
            )
        else:
            v_hub = adjust_wind_speed_log_profile(
                v_ref, ref_height_m, hub_height_m,
                z0=float(turbine_spec.get("roughness_length", 0.03)),
            )
        rho_i = rho_arr[i] if not np.isnan(rho_arr[i]) else RHO_STD
        p     = baseline_turbine_power(
            v_hub, turbine_spec, rho=rho_i, use_power_curve=use_power_curve
        )
        power_out[i] = p

    # ── FIX 1 continued — Enforce cut-in / cut-out / rated cap ───────────
    # baseline_turbine_power already applies cut-in/cut-out per call,
    # but we recompute hub-height wind speeds vectorised for the mask.
    v_hub_arr = np.array([
        adjust_wind_speed_power_law(v, ref_height_m, hub_height_m, shear_alpha)
        if shear_method == "power_law"
        else adjust_wind_speed_log_profile(
            v, ref_height_m, hub_height_m,
            z0=float(turbine_spec.get("roughness_length", 0.03))
        )
        for v in ws
    ])
    power_out = np.where(v_hub_arr < cut_in_ms,  0.0,     power_out)
    power_out = np.where(v_hub_arr > cut_out_ms, 0.0,     power_out)
    power_out = np.where(power_out  > rated_w,   rated_w, power_out)
    # ─────────────────────────────────────────────────────────────────────

    return power_out


def compute_aep_from_power_series(
    power_w_array: np.ndarray, dt_hours: float = 1.0
) -> float:
    power_w    = np.asarray(power_w_array, dtype=float)
    total_kwh  = np.nansum(power_w) * float(dt_hours) / 1000.0
    return float(total_kwh)


def capacity_factor(aep_kwh: float, rated_power_kw: float) -> float:
    denom = float(rated_power_kw) * 8760.0
    if denom <= 0:
        return np.nan
    return float(aep_kwh) / denom