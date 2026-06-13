#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os

def main():
    # Data provided by user (north -> south)
    locations = ['Sokoto', 'Katsina', 'Ilorin', 'Abuja', 'Port Harcourt']
    best_cf_pct = [27.24, 30.04, 12.74, 9.32, 4.42]
    best_turbine = ['V110-2.0MW', 'V110-2.0MW', 'V110-2.0MW', 'V110-2.0MW', 'V150-4.2MW']

    thresh = 25.0  # percent viability threshold

    colors = ['#2ca02c' if v >= thresh else '#b0b0b0' for v in best_cf_pct]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(locations, best_cf_pct, color=colors)

    # Add dashed threshold line
    ax.axhline(thresh, color='green', linestyle='--', linewidth=1.5)
    ax.text(0.98, thresh + 0.8, 'Viability threshold (25%)', color='green', ha='right', va='bottom', fontsize=9, transform=ax.get_yaxis_transform())

    # Annotate bars with percent and turbine
    for i, b in enumerate(bars):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 0.8, f"{h:.2f}%", ha='center', va='bottom', fontsize=9)
        # turbine label under x tick as second line
        # We'll set xticklabels to include turbine

    xticklabels = [f"{loc}\n{turb}" for loc, turb in zip(locations, best_turbine)]
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel('Best Hybrid Capacity Factor (%)')
    ax.set_title('Best Hybrid Capacity Factor — Five Locations (North→South)')
    ax.set_ylim(0, max(best_cf_pct) * 1.25)
    plt.tight_layout()

    outdir = os.path.join('results')
    os.makedirs(outdir, exist_ok=True)
    out_png = os.path.join(outdir, 'figure_capacity_factor_5_locations.png')
    out_svg = os.path.join(outdir, 'figure_capacity_factor_5_locations.svg')
    plt.savefig(out_png, dpi=300)
    plt.savefig(out_svg)
    print(f"Saved {out_png} and {out_svg}")

if __name__ == '__main__':
    main()
