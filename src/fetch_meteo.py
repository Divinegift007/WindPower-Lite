"""
fetch_meteo.py

Fetch hourly meteo data (WS50M, T2M, PS) from NASA POWER Hourly API.

Returns pandas.DataFrame with columns:
['timestamp', 'wind_speed_50m_mps', 'temp_2m_c', 'surface_pressure_pa']

Data is automatically cached locally to avoid redundant API calls for the same location/date range.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Optional
import time
import hashlib
from pathlib import Path
import os

# --- Default coordinate (Ilorin, Kwara) ---
DEFAULT_LAT = 8.4966
DEFAULT_LON = 4.5421

# NASA POWER hourly endpoint base
BASE_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Which parameters to request — use WS50M for Option A
PARAMETERS = ["WS50M", "T2M", "PS"]  # WS50M = wind speed at 50m; T2M = temp @2m; PS = surface pressure

# Cache directory for meteorological data
CACHE_DIR = Path(os.path.dirname(__file__)).parent / "meteo_cache"


def _get_cache_key(lat: float, lon: float, start_dt: datetime, end_dt: datetime) -> str:
    """Generate a unique cache filename for a location and date range."""
    key_str = f"{lat:.6f}_{lon:.6f}_{start_dt.date().isoformat()}_{end_dt.date().isoformat()}"
    hash_val = hashlib.md5(key_str.encode()).hexdigest()
    return f"{hash_val}.parquet"


def _dt_to_nasa_date(dt: datetime) -> str:
    """Convert a Python datetime to NASA POWER date string YYYYMMDD."""
    return dt.strftime("%Y%m%d")


def fetch_hourly_nasa_power(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    start_dt: Optional[datetime] = None,
    end_dt: Optional[datetime] = None,
    time_standard: str = "UTC",
    max_retries: int = 3,
    retry_delay: float = 1.0,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Fetch hourly WS50M, T2M, PS from NASA POWER for given lat/lon and datetime range.
    Returns DataFrame with columns: timestamp, wind_speed_50m_mps, temp_2m_c, surface_pressure_pa
    """

    # use timezone-aware UTC now to avoid deprecation warning
    now = datetime.now(timezone.utc)
    if end_dt is None:
        end_dt = now
    if start_dt is None:
        start_dt = end_dt - timedelta(days=365)

    start_str = _dt_to_nasa_date(start_dt)
    end_str = _dt_to_nasa_date(end_dt)

    # ── Check cache ───────────────────────────────────────────────────────────
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / _get_cache_key(lat, lon, start_dt, end_dt)
    
    if cache_file.exists() and not force_refresh:
        print(f"[fetch_meteo] Cache hit: {cache_file.name} — loading from disk")
        df_out = pd.read_parquet(cache_file)
        print(f"[fetch_meteo] Loaded {len(df_out)} rows from cache")
        return df_out

    base_params = {
        "latitude": lat,
        "longitude": lon,
        "parameters": ",".join(PARAMETERS),
        "format": "JSON",
        "time-standard": time_standard,
        "community": "RE",
    }

    def _fetch_single(s_dt: datetime, e_dt: datetime):
        p = base_params.copy()
        p["start"] = _dt_to_nasa_date(s_dt)
        p["end"] = _dt_to_nasa_date(e_dt)
        last_exc = None
        for attempt in range(max_retries):
            try:
                resp = requests.get(BASE_URL, params=p, timeout=30)
                resp.raise_for_status()
                return resp.json()
            except Exception as e:
                last_exc = e
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(retry_delay)

    # Try single fetch; on failure (large range or 422) chunk to 7-day pieces
    try:
        data = _fetch_single(start_dt, end_dt)
    except requests.exceptions.HTTPError as he:
        status = getattr(he, "status_code", None)
        resp_text = getattr(he, "response_text", "")
        total_days = (end_dt - start_dt).days
        if status == 422 or total_days > 7:
            chunk_jsons = []
            cur = start_dt
            any_success = False
            while cur < end_dt:
                chunk_end = min(cur + timedelta(days=7), end_dt)
                try:
                    cj = _fetch_single(cur, chunk_end)
                    chunk_jsons.append(cj)
                    any_success = True
                except Exception:
                    pass
                cur = chunk_end + timedelta(days=1)
            if not any_success:
                raise RuntimeError(f"NASA POWER API returned HTTP {status} for range {start_str}->{end_str}. Response: {resp_text}") from he
            combined = {"properties": {"parameter": {}}}
            for cj in chunk_jsons:
                props = cj.get("properties", {})
                params_blk = props.get("parameter", {})
                for pname, pmap in params_blk.items():
                    if pname not in combined["properties"]["parameter"]:
                        combined["properties"]["parameter"][pname] = {}
                    combined["properties"]["parameter"][pname].update(pmap)
            data = combined
        else:
            raise RuntimeError(f"Failed to fetch NASA POWER data: HTTP {status}. Response: {resp_text}") from he
    except Exception as e:
        raise RuntimeError(f"Failed to fetch NASA POWER data: {e}") from e

    # Parse response
    try:
        parameters_block = data["properties"]["parameter"]
    except KeyError:
        raise RuntimeError("Unexpected NASA POWER response structure. Full response: " + str(data))

    sample_param = None
    for p in PARAMETERS:
        if p in parameters_block:
            sample_param = p
            break
    if sample_param is None:
        raise RuntimeError(f"None of the requested parameters were returned by NASA POWER. Requested: {PARAMETERS}")

    ts_dict = parameters_block[sample_param]
    timestamps = []
    for k in ts_dict.keys():
        if len(k) == 8:
            fmt = "%Y%m%d"
        elif len(k) == 10:
            fmt = "%Y%m%d%H"
        elif len(k) == 12:
            fmt = "%Y%m%d%H%M"
        else:
            try:
                timestamps.append(datetime.strptime(k[:10], "%Y%m%d%H"))
                continue
            except Exception:
                raise RuntimeError(f"Unexpected timestamp format from NASA POWER: '{k}'")
        dt = datetime.strptime(k, fmt)
        timestamps.append(dt)

    df = pd.DataFrame({"timestamp": timestamps})
    df.set_index("timestamp", inplace=False)

    for p in PARAMETERS:
        param_dict = parameters_block.get(p, {})
        dt_map = {}
        for k, v in param_dict.items():
            if len(k) == 10:
                dt = datetime.strptime(k, "%Y%m%d%H")
            elif len(k) == 12:
                dt = datetime.strptime(k, "%Y%m%d%H%M")
            elif len(k) == 8:
                dt = datetime.strptime(k, "%Y%m%d")
            else:
                dt = datetime.strptime(k[:10], "%Y%m%d%H")
            dt_map[dt] = v
        ordered_vals = [dt_map.get(ts, None) for ts in timestamps]
        df[p] = ordered_vals

    # Normalize columns and units
    df_out = pd.DataFrame({"timestamp": timestamps})
    # WS50M should be present, but convert sentinel values to NaN
    if "WS50M" in df.columns:
        df_out["wind_speed_50m_mps"] = df["WS50M"].astype(float)
        # avoid pandas chained assignment warnings by reassigning result
        df_out["wind_speed_50m_mps"] = df_out["wind_speed_50m_mps"].replace([-999, -9999], float("nan"))
    else:
        raise RuntimeError("Wind speed parameter (WS50M) missing from NASA response.")

    # temperature (T2M)
    df_out["temp_2m_c"] = df["T2M"].astype(float) if "T2M" in df.columns else None
    if "temp_2m_c" in df_out.columns:
        df_out["temp_2m_c"] = df_out["temp_2m_c"].replace([-999, -9999], float("nan"))

    # pressure normalization with more robust unit detection
    if "PS" in df.columns:
        ps_series = pd.Series(df["PS"]).astype(float)
        # treat NASA sentinel values as missing
        ps_series = ps_series.replace([-999, -9999], float("nan"))
        if ps_series.dropna().empty:
            df_out["surface_pressure_pa"] = None
        else:
            median_val = ps_series.dropna().median()
            # NASA POWER historically returns pressure in kilopascals (~90–110 kPa).
            # older logic mis-classified these values and left them at ~100 Pa,
            # leading to air density ≈0.001 kg/m³ and absurdly low physics power.
            # Acceptable ranges:
            #   < 200      : assume kPa → convert to Pa
            #   200–2000   : assume hPa → convert to Pa
            #   >=2000     : already in Pa
            if median_val < 200:
                df_out["surface_pressure_pa"] = ps_series * 1000.0
            elif median_val < 2000:
                df_out["surface_pressure_pa"] = ps_series * 100.0
            else:
                df_out["surface_pressure_pa"] = ps_series
    else:
        df_out["surface_pressure_pa"] = None

    # final housekeeping
    df_out["timestamp"] = pd.to_datetime(df_out["timestamp"])
    df_out = df_out.sort_values("timestamp").reset_index(drop=True)

    # ── Save to cache ─────────────────────────────────────────────────────────
    try:
        df_out.to_parquet(cache_file)
        print(f"[fetch_meteo] Cached {len(df_out)} rows to {cache_file.name}")
    except Exception as e:
        print(f"[fetch_meteo] Warning: could not save cache ({e})")

    return df_out


# Example usage guard
if __name__ == "__main__":
    from datetime import datetime
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=365)
    print("Fetching NASA POWER hourly data for Ilorin (default)...")
    df = fetch_hourly_nasa_power(lat=8.4966, lon=4.5421, start_dt=start, end_dt=end)
    print(df.head())
    print(f"Rows fetched: {len(df)}")
