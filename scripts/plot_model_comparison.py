#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    csv_path = os.path.join('results', 'model_comparison.csv')
    df = pd.read_csv(csv_path)

    models = df['Model'].tolist()
    rmse_w = df['RMSE (W)'].values
    r2 = df['R²'].values

    # Convert RMSE to kW for plotting
    rmse_kw = rmse_w / 1000.0

    # Colors: highlight Gradient Boosting
    highlight = 'Gradient Boosting'
    colors = [('#1f77b4' if m == highlight else '#b0b0b0') for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: RMSE (kW)
    ax = axes[0]
    bars = ax.bar(models, rmse_kw, color=colors)
    ax.set_ylabel('RMSE (kW)')
    ax.set_title('Model RMSE')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=30, ha='right')

    # Annotate RMSE bars with formatted kW values
    bar_labels = [f"{v:,.1f} kW" for v in rmse_kw]
    ax.bar_label(bars, labels=bar_labels, padding=3, fontsize=9)

    # Emphasize winner bar
    for i, b in enumerate(bars):
        if models[i] == highlight:
            b.set_edgecolor('black')
            b.set_linewidth(1.5)

    # Right: R^2
    ax2 = axes[1]
    bars2 = ax2.bar(models, r2, color=colors)
    ax2.set_ylabel('R²')
    ax2.set_ylim(0, 1)
    ax2.set_title('Model R²')
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels(models, rotation=30, ha='right')

    # Annotate R^2 bars
    r2_labels = [f"{v:.3f}" for v in r2]
    ax2.bar_label(bars2, labels=r2_labels, padding=3, fontsize=9)
    for i, b in enumerate(bars2):
        if models[i] == highlight:
            b.set_edgecolor('black')
            b.set_linewidth(1.5)

    plt.tight_layout()

    outdir = os.path.join('results')
    os.makedirs(outdir, exist_ok=True)
    outpath_png = os.path.join(outdir, 'figure_model_comparison.png')
    outpath_svg = os.path.join(outdir, 'figure_model_comparison.svg')
    plt.savefig(outpath_png, dpi=300)
    plt.savefig(outpath_svg)
    print(f"Saved {outpath_png} and {outpath_svg}")

if __name__ == '__main__':
    main()
