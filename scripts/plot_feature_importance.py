import os
import sys
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BUNDLE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'hybrid_all.joblib')
OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
OUT_PATH = os.path.join(OUT_DIR, 'figure_feature_importance.png')
TURBINE = 'V90-2.0MW'

os.makedirs(OUT_DIR, exist_ok=True)

try:
    bundle = joblib.load(BUNDLE_PATH)
except Exception as e:
    print('Failed to load bundle:', BUNDLE_PATH)
    print(e)
    sys.exit(1)

try:
    model = bundle['models'][TURBINE]
    feature_cols = bundle['feature_cols'][TURBINE]
except KeyError as e:
    print('Expected keys not found in bundle:', e)
    print('Bundle keys:', list(bundle.keys()))
    sys.exit(1)

# Try to obtain feature importances
try:
    importances = model.named_steps['regressor'].feature_importances_
except Exception as e:
    print('Failed to read feature_importances_ from model:', e)
    sys.exit(1)

if len(importances) != len(feature_cols):
    print('Warning: mismatch between importances length and feature_cols length')

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_cols)), importances[indices], color='#0A7E8C', edgecolor='white')
plt.xticks(range(len(feature_cols)), [feature_cols[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Feature Importance Score')
plt.title(f'Gradient Boosting Feature Importances — {TURBINE}')
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300)
print('Saved figure to', OUT_PATH)
