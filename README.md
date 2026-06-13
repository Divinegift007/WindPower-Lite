# Windpower Lite

Windpower Lite is a lightweight, farm‑scale wind‑power estimation toolkit. It combines a
simple physics‑based baseline with an optional machine‑learning correction layer and
provides a Streamlit user interface for interactive site selection and analysis.

The project is designed for developers, researchers and students who want a
physically interpretable yet flexible estimator that can run entirely from public
meteorological data (NASA POWER) and a small set of turbine specifications.

This project is currently configured for historical/retrospective estimation, not
for operational future weather forecasting.

---

## 🔧 What it does

* Fetches hourly meteorological data from NASA POWER for a chosen site and date range.
* Computes a physics‑based wind power baseline using turbine geometry, air density,
  hub‑height wind speed, and a constant coefficient of performance.
* Optionally applies a hybrid machine learning correction trained on the difference
  between the physics baseline and a manufacturer power curve.
* Computes summary outputs including annual energy production (AEP) and capacity
  factor for both physics and hybrid estimates.
* Provides a Streamlit app for map‑based site selection, turbine selection, and
  visualization of time series and summary statistics.
* Includes a training script for building hybrid correction models from one year
  of historical data.
* Supports command‑line execution, unit tests, and container deployment.

---

## 📁 Repository Layout

```
├── app/                     # Streamlit front end
│   └── app.py
├── data/                    # Input data such as turbine specifications
│   └── turbine_specs.csv
├── models/                  # Trained hybrid model artifacts
│   ├── hybrid_model.joblib      # legacy single-model file
│   └── hybrid_all.joblib        # per-turbine bundle used by UI/library
├── src/                     # Core library modules
│   ├── fetch_meteo.py           # NASA POWER client and data normalization
│   ├── physics_model.py         # physics-based power and aero functions
│   ├── hybrid_model.py          # hybrid correction training and prediction logic
│   ├── main.py                  # main pipeline orchestration
│   ├── train_hybrid.py          # offline hybrid training script
│   └── utils.py                 # turbine spec loading helpers
├── tests/                   # unit and integration tests
├── diagnostics/ …            # output artifacts generated during development
├── README.md                # this file
└── requirements.txt         # Python dependencies
```

---

## 🚀 Quick Start

1. **Create and activate a virtual environment**:

   ```bash
   python -m venv wplvenv
   source wplvenv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the physics-only pipeline** (demo):

   ```bash
   python - <<'PY'
   import os, sys
   sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
   from main import run_pipeline
   from datetime import datetime, timedelta
   end = datetime.utcnow(); start = end - timedelta(days=3)
   df, summary = run_pipeline(start_dt=start, end_dt=end, apply_hybrid_if_available=False)
   print(summary)
   df.to_csv('demo_results.csv', index=False)
   PY
   ```

3. **Train or update hybrid models** (default uses 12 months of historical data):

   ```bash
   python src/train_hybrid.py
   ```

4. **Launch the Streamlit user interface**:

   ```bash
   streamlit run app/app.py
   ```

5. **Run the test suite**:

   ```bash
   python -m unittest -v
   ```

6. **(Optional) Containerized deployment**

   A `Dockerfile` and `docker-compose.yml` are provided to package the application.
   Build and run with:

   ```bash
   docker build -t windpower_lite:latest .
   docker run --rm -p 8501:8501 windpower_lite:latest
   ```

   or

   ```bash
   docker compose up --build
   ```

   Source is mounted by compose so you can iterate without rebuilding.

---

## 📘 How It Works

### 1. Fetch meteorological data

`src/fetch_meteo.py` retrieves hourly NASA POWER data for a given latitude/longitude
and time range. It requests wind speed at 50 m (`WS50M`), 2 m temperature (`T2M`),
and surface pressure (`PS`).

The fetcher handles:
* retries on transient request failures,
* chunking long date ranges if necessary,
* parsing NASA timestamp formats,
* sentinel value cleanup (`-999`, `-9999`),
* pressure unit normalization from kPa/hPa to Pa.

### 2. Compute the physics baseline

`src/main.py` prepares the site dataframe by:
* converting timestamps,
* computing air density from pressure and temperature,
* extrapolating 50 m wind speed to turbine hub height using a power law,
* computing the physics power series with `src/physics_model.py`.

`src/physics_model.py` provides the core equations:
* `air_density_from_ps_temp()` — ideal gas density calculation,
* `rotor_swept_area()` — swept area from rotor diameter,
* `adjust_wind_speed_power_law()` — hub height wind speed scaling,
* `power_from_power_curve()` — manufacturer curve interpolation,
* `physics_power_from_cp()` — aerodynamic power from `0.5 * rho * A * v^3 * Cp * eta`,
* `baseline_turbine_power()` — cut-in/cut-out/rated power logic,
* `compute_hourly_power_series()` — apply the turbine model to a full time series,
* `compute_aep_from_power_series()` — integrate kWh from hourly watts.

### 3. Apply the hybrid correction (optional)

`src/hybrid_model.py` builds a machine learning residual model that corrects the
physics baseline. It supports:
* pseudo-target generation from manufacturer power curves,
* feature construction including physics power, hub wind speed, temperature,
  pressure, hour, and month,
* training a scikit-learn pipeline with `RandomForestRegressor` or
  `GradientBoostingRegressor`,
* saving and loading models with `joblib`,
* predicting hybrid-corrected power either from a single model or a bundled
  per-turbine artifact.

### 4. Summarize results

The pipeline returns:
* hourly data with physics and optional hybrid output,
* annual energy production (`aep_physics_kwh`, `aep_hybrid_kwh`),
* capacity factor computed using the actual number of hours in the dataset,
* optional hybrid model metadata.

### 5. Streamlit interface

`app/app.py` provides the user-facing experience:
* map-based location selection with search and click-to-set coordinates,
* date range input for historical evaluation,
* turbine model selection from `data/turbine_specs.csv`,
* optional hybrid correction toggle,
* summary metrics, time series plots, and download of CSV results.

> Note: the current app uses NASA POWER historical data only. It is not a future
> weather forecast system.

---

## 🧠 Hybrid Modelling Philosophy

Windpower Lite uses a hybrid framework:

* A **simple physics baseline** generates an initial estimate using the
  aerodynamic wind power equation with a constant coefficient of performance.
* A **machine learning residual model** learns the difference between that
  baseline and an idealized turbine power curve.
* This keeps the prediction physically grounded while allowing the system to
  correct for turbine-specific non-linear performance.

This means the hybrid output is intended as a correction to the baseline, not
as a standalone black-box predictor.

---

## 🧩 Current Limitations

* The date picker in the UI is a historical evaluation window, not a future
  forecast window.
* The hybrid training targets are derived from manufacturer power curves,
  not actual operational SCADA measurements.
* The system is therefore best suited for preliminary feasibility screening,
  not bankable operational yield forecasting.

---

## 📦 Extending the Project

* Add or update turbines by editing `data/turbine_specs.csv`.
* Modify the physical model parameters in `src/physics_model.py`.
* Extend `src/train_hybrid.py` or `src/hybrid_model.py` to train alternate
  machine learning approaches.
* Add a new weather provider for forecasting by replacing or extending
  `src/fetch_meteo.py`.

---

## ✅ Dependencies

Key packages in `requirements.txt`:
* `numpy`, `pandas`
* `scikit-learn`
* `streamlit`, `plotly`
* `requests`
* `joblib`
* `folium`, `streamlit_folium`, `geopy`

## 📝 License & Contribution

*(Add licence and contribution instructions here if needed.)*

---

Questions, improvements or bug reports are welcome — feel free to open an issue or
submit a pull request!
 

---

## 🟦 Project state & generated artifacts (current)

This repository contains both source code and a number of generated artifacts
created during development. Important notes about the current state:

- The hybrid models were retrained using a Gradient Boosting regressor and
  bundled into `models/hybrid_all.joblib` (used by the Streamlit UI if present).
- The training workflow now enforces turbine cut-in/cut-out and adds ~3% Gaussian
  noise to pseudo-targets to avoid deterministic memorisation of power curves.
- Diagnostic and plot scripts live in `scripts/` and write outputs to `results/`
  and `figures/` (examples: `scripts/plot_model_comparison.py`,
  `scripts/plot_capacity_factor_5_locations.py`).
- Mermaid diagram sources are stored in `docs/*.mmd`; use `@mermaid-js/mermaid-cli`
  (`mmdc`) to render PNG/SVG for embedding in `docs/figures/`.

If you want reproducible runs, prefer creating a clean venv (see Quick Start)
and avoid checking generated artifacts into source control (see `.gitignore`).

