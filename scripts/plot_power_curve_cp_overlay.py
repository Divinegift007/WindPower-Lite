"""
Plot manufacturer power curve for V90-2.0MW with Cp=0.40 physics baseline overlay.
Saves to figures/figure_power_curve_cp_overlay.png
"""
import os
import sys
import math
import numpy as np
import matplotlib.pyplot as plt

# make src importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_turbine_specs

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
OUT_PATH = os.path.join(OUT_DIR, 'figure_power_curve_cp_overlay.png')
os.makedirs(OUT_DIR, exist_ok=True)

TURBINE = 'V90-2.0MW'
CP_BASE = 0.40
RHO = 1.225  # kg/m3

specs = load_turbine_specs()
if TURBINE not in specs:
    raise KeyError(f'Turbine {TURBINE} not found in specs')
spec = specs[TURBINE]
pc = spec.get('power_curve')
if pc is None or pc.empty:
    raise RuntimeError('Power curve missing for turbine')

# power_curve in kW (as loaded)
ws_curve = pc['wind_speed_mps'].astype(float).values
p_curve_kw = pc['power_kw'].astype(float).values

# create fine wind speed grid
ws = np.linspace(0.0, max(ws_curve.max(), spec.get('cut_out_mps', 25.0)), 400)
# manufacturer curve (interpolate), kW
p_curve_interp_kw = np.interp(ws, ws_curve, p_curve_kw, left=0.0, right=p_curve_kw[-1])

# Cp baseline in kW
d = float(spec.get('rotor_diameter_m', 90.0))
A = math.pi * (d/2.0) ** 2
p_cp_w = 0.5 * RHO * A * (ws ** 3) * CP_BASE
p_cp_kw = p_cp_w / 1000.0

# markers
cut_in = float(spec.get('cut_in_mps', 3.5))
rated_ws = float(spec.get('rated_mps', spec.get('rated_mps', ws_curve.max())))
cut_out = float(spec.get('cut_out_mps', 25.0))
rated_power_kw = float(spec.get('rated_power_kw', 2000.0))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ws, p_curve_interp_kw, label='Manufacturer power curve', color='#0A7E8C', linewidth=2.5)
ax.plot(ws, p_cp_kw, label=f'Cp = {CP_BASE:.2f} physics baseline', color='gray', linestyle='--', linewidth=2)

# vertical lines for cut-in, rated, cut-out
ax.axvline(cut_in, color='k', linestyle=':', linewidth=1)
ax.text(cut_in, ax.get_ylim()[1]*0.05, f'cut-in\n{cut_in:.1f} m/s', ha='center', va='bottom')
ax.axvline(rated_ws, color='k', linestyle=':', linewidth=1)
ax.text(rated_ws, ax.get_ylim()[1]*0.95, f'rated\n{rated_ws:.1f} m/s', ha='center', va='bottom')
ax.axvline(cut_out, color='k', linestyle=':', linewidth=1)
ax.text(cut_out, ax.get_ylim()[1]*0.05, f'cut-out\n{cut_out:.1f} m/s', ha='center', va='bottom')

# Annotate regions
ypos = ax.get_ylim()[1]*0.5
ax.annotate('Region 1\n(Inactive)', xy=(cut_in/2, ypos), xytext=(cut_in/2, ypos), ha='center')
ax.annotate('Region 2\n(Partial to Rated)', xy=((cut_in + rated_ws)/2, ypos), ha='center')
ax.annotate('Region 3\n(Control/Flat)', xy=((rated_ws + cut_out)/2, ypos), ha='center')

ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Power (kW)')
ax.set_title(f'{TURBINE} — Manufacturer power curve with Cp baseline (Cp={CP_BASE:.2f})')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
print('Saved power curve figure to', OUT_PATH)
