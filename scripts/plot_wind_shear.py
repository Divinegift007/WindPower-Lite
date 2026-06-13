"""
Plot a wind shear extrapolation diagram showing reference height (50 m) and
hub heights for all turbines in `data/turbine_specs.csv`.

Saves output to `figures/wind_shear_profile.png`.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
# Ensure project src is importable (same pattern as other scripts)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from utils import load_turbine_specs

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
OUT_PATH = os.path.join(OUT_DIR, 'wind_shear_profile.png')

os.makedirs(OUT_DIR, exist_ok=True)

# Parameters for the power-law wind shear
Z_REF = 50.0  # reference height (m)
V_REF = 5.0   # reference wind speed at Z_REF (m/s) -- illustrative
ALPHA = 0.14  # shear exponent (typical)

# Load turbine specs
specs = load_turbine_specs()
# Collect hub heights and names
turbines = []
for name, spec in specs.items():
    h = float(spec.get('hub_height_m', 0.0))
    turbines.append((name, h))

# Sort by height
turbines.sort(key=lambda x: x[1])
hub_heights = [h for _, h in turbines]
names = [n for n, _ in turbines]

# Height grid
z_max = max(hub_heights) + 30.0
z = np.linspace(0.1, z_max, 400)
# Power-law
v = V_REF * (z / Z_REF) ** ALPHA

# Plot
fig, ax = plt.subplots(figsize=(4.5, 8))
ax.plot(v, z, color='#0A7E8C', linewidth=3)
# Mark reference height
ax.axhline(Z_REF, color='gray', linestyle='--', linewidth=1)
ax.text(v.max()*0.02, Z_REF + 1.5, f'{int(Z_REF)} m reference', color='gray')
# Ground line
ax.axhline(0, color='k', linewidth=1.5)
ax.text(v.max()*0.02, -2.5, 'Ground', va='top')

# Plot hub height lines and labels
for name, h in turbines:
    ax.hlines(h, xmin=0, xmax=v.max()*1.02, color='#666666', linewidth=1)
    ax.text(v.max()*1.04, h, f'{name} ({int(h)} m)', va='center', fontsize=8)

# Styling
ax.set_xlabel('Wind speed (m/s)')
ax.set_ylabel('Height (m)')
ax.set_xlim(left=0)
ax.set_ylim(bottom=-5, top=z_max)
ax.grid(axis='x', linestyle=':', alpha=0.6)
ax.set_title('Wind Shear Extrapolation — Power-law (α=0.14)')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
print('Saved wind shear figure to', OUT_PATH)
