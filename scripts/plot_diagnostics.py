#!/usr/bin/env python3
"""plot_diagnostics.py
Load windpower_results.csv and create:
 - diagnostics/physics_vs_hybrid.png (scatter P_physics_w vs P_hybrid_w)
 - diagnostics/residual_hist.png (histogram of residuals = P_hybrid_w - P_physics_w)
Prints simple summary stats to stdout.
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "windpower_results.csv"
OUT_DIR = ROOT / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not CSV_PATH.exists():
    print(f"Error: {CSV_PATH} not found.")
    sys.exit(2)

df = pd.read_csv(CSV_PATH, parse_dates=["timestamp"]) 

# Ensure required cols
if "P_physics_w" not in df.columns:
    print("Error: P_physics_w column missing in CSV")
    sys.exit(2)

# If P_hybrid_w missing, create NaN series
if "P_hybrid_w" not in df.columns:
    print("Warning: P_hybrid_w column missing; cannot compute hybrid diagnostics.")
    df["P_hybrid_w"] = np.nan

# compute residuals and drop NaNs for plotting
df["residual_w"] = df["P_hybrid_w"] - df["P_physics_w"]
mask = df["residual_w"].notna()

# Summary stats
count = len(df)
count_nonnull = mask.sum()
mean_phys = float(df["P_physics_w"].mean())
mean_hyb = float(df["P_hybrid_w"].mean())
mean_res = float(df.loc[mask, "residual_w"].mean()) if count_nonnull>0 else float('nan')
std_res = float(df.loc[mask, "residual_w"].std()) if count_nonnull>0 else float('nan')

print(f"Rows in CSV: {count}")
print(f"Non-null residuals (hybrid present): {count_nonnull}")
print(f"Mean P_physics_w: {mean_phys:.3f} W")
print(f"Mean P_hybrid_w:  {mean_hyb:.3f} W")
print(f"Mean residual (hybrid - physics): {mean_res:.3f} W")
print(f"Residual std: {std_res:.3f} W")

# Scatter P_physics vs P_hybrid (use log scale for readability when values span orders)
scatter_path = OUT_DIR / "physics_vs_hybrid.png"
plt.figure(figsize=(7,6))
plt.scatter(df["P_physics_w"], df["P_hybrid_w"], s=6, alpha=0.6)
plt.plot([0, max(df["P_physics_w"].max(), df["P_hybrid_w"].max(), 1)], [0, max(df["P_physics_w"].max(), df["P_hybrid_w"].max(), 1)], color="red", linestyle="--", label="1:1")
plt.xlabel("P_physics_w (W)")
plt.ylabel("P_hybrid_w (W)")
plt.title("Physics vs Hybrid power (W)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(scatter_path)
plt.close()

# Residual histogram (linear and zoomed)
hist_path = OUT_DIR / "residual_hist.png"
plt.figure(figsize=(8,5))
plt.hist(df.loc[mask, "residual_w"], bins=80, color="#2b8cbe", alpha=0.85)
plt.xlabel("Residual (W)")
plt.ylabel("Count")
plt.title("Residual distribution: P_hybrid - P_physics (W)")
plt.grid(axis='y', linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(hist_path)
plt.close()

print(f"Saved scatter: {scatter_path}")
print(f"Saved histogram: {hist_path}")

# exit success
sys.exit(0)
