"""
utils.py
Utility functions for loading turbine specifications and power curves.
"""

import os
import json
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TURBINE_CSV = os.path.join(DATA_DIR, "turbine_specs.csv")

def load_turbine_specs(csv_path: str = TURBINE_CSV):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"turbine_specs file not found at: {csv_path}")
    df = pd.read_csv(csv_path)
    required_cols = [
        "turbine_name",
        "rated_power_kw",
        "rotor_diameter_m",
        "hub_height_m",
        "cut_in_mps",
        "rated_mps",
        "cut_out_mps",
        "cp",
        "power_curve_json",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in turbine_specs.csv: {c}")
    specs = {}
    for _, row in df.iterrows():
        name = row["turbine_name"]
        spec = {
            "turbine_name": name,
            "rated_power_kw": float(row["rated_power_kw"]),
            "rotor_diameter_m": float(row["rotor_diameter_m"]),
            "hub_height_m": float(row["hub_height_m"]),
            "cut_in_mps": float(row["cut_in_mps"]),
            "rated_mps": float(row["rated_mps"]),
            "cut_out_mps": float(row["cut_out_mps"]),
            "cp": float(row["cp"]),
            "eta_sys": 0.9,
        }
        pc_json = row["power_curve_json"]
        try:
            pairs = json.loads(pc_json)
            pc_df = pd.DataFrame(pairs, columns=["wind_speed_mps", "power_kw"])
            spec["power_curve"] = pc_df
        except Exception:
            spec["power_curve"] = None
        specs[name] = spec
    return specs
