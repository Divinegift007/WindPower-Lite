"""
hybrid_model.py
Lightweight hybrid correction layer for Windpower Lite.
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import os

def make_pseudo_targets_from_power_curve(df: pd.DataFrame, power_curve_df: pd.DataFrame, v_col: str = "v_hub_mps", out_col: str = "P_true_w") -> pd.DataFrame:
    if v_col not in df.columns:
        raise ValueError(f"{v_col} not found in provided df")
    pc = power_curve_df.sort_values("wind_speed_mps").reset_index(drop=True)
    ws = pc["wind_speed_mps"].values.astype(float)
    p_kw = pc["power_kw"].values.astype(float)
    df_copy = df.copy()
    df_copy[out_col] = np.interp(df_copy[v_col].fillna(0).values, ws, p_kw, left=0.0, right=p_kw[-1]) * 1000.0
    return df_copy

def prepare_features(df: pd.DataFrame, physics_power_col: str = "P_physics_w", v_col: str = "v_hub_mps", temp_col: Optional[str] = "temp_2m_c", pressure_col: Optional[str] = "surface_pressure_pa", time_cols: Optional[List[str]] = None, extra_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if "P_true_w" not in df.columns:
        raise ValueError("DataFrame must contain 'P_true_w' (true or pseudo power in Watts).")
    if time_cols is None:
        df = df.copy()
        if "timestamp" in df.columns:
            df["hour"] = df["timestamp"].dt.hour
            df["month"] = df["timestamp"].dt.month
            time_cols = ["hour", "month"]
        else:
            time_cols = []
    features = []
    if physics_power_col not in df.columns:
        raise ValueError(f"{physics_power_col} not present in dataframe. Run physics model first.")
    features.append(physics_power_col)
    if v_col in df.columns:
        features.append(v_col)
    if temp_col and temp_col in df.columns:
        features.append(temp_col)
    if pressure_col and pressure_col in df.columns:
        features.append(pressure_col)
    features += [c for c in (time_cols or []) if c in df.columns]
    if extra_cols:
        features += [c for c in extra_cols if c in df.columns]
    X = df[features].ffill().fillna(0.0).values
    y = (df["P_true_w"].values - df[physics_power_col].values).astype(float)
    return X, y, features

def get_model_pipeline(model_type: str = "rf", random_state: int = 42, n_jobs: int = -1) -> Pipeline:
    if model_type == "rf":
        reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=random_state, n_jobs=n_jobs)
    elif model_type == "gbr":
        reg = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=random_state)
    else:
        raise ValueError("model_type must be 'rf' or 'gbr'")
    pipeline = Pipeline([("scaler", StandardScaler()), ("regressor", reg)])
    return pipeline

def train_hybrid_model(X: np.ndarray, y: np.ndarray, model_type: str = "rf", cv_folds: int = 5, random_state: int = 42, n_jobs: int = -1, save_path: Optional[str] = None) -> Tuple[Pipeline, Dict[str, float]]:
    model = get_model_pipeline(model_type=model_type, random_state=random_state, n_jobs=n_jobs)
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rmse_scores = []
    mae_scores = []
    r2_scores = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        rmse_scores.append(np.sqrt(mean_squared_error(y_te, y_pred)))
        mae_scores.append(mean_absolute_error(y_te, y_pred))
        r2_scores.append(r2_score(y_te, y_pred))
    model.fit(X, y)
    metrics = {
        "rmse_cv_mean": float(np.mean(rmse_scores)),
        "mae_cv_mean": float(np.mean(mae_scores)),
        "r2_cv_mean": float(np.mean(r2_scores)),
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(model, save_path)
    return model, metrics

def predict_hybrid(model: Pipeline, df: pd.DataFrame, physics_power_col: str = "P_physics_w", feature_cols: Optional[List[str]] = None, max_output_w: Optional[float] = None) -> pd.Series:
    if feature_cols is None:
        raise ValueError("feature_cols must be provided (list of columns used when training).")
    X = df[feature_cols].ffill().fillna(0.0).values
    pred_residual = model.predict(X)
    p_phys = df[physics_power_col].values
    p_hybrid = p_phys + pred_residual
    p_hybrid = np.where(np.isnan(p_hybrid), np.nan, np.maximum(0.0, p_hybrid))
    if max_output_w is not None:
        p_hybrid = np.minimum(p_hybrid, float(max_output_w))
    return pd.Series(p_hybrid, index=df.index)

def save_model(model: Pipeline, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str) -> Pipeline:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)


def load_hybrid_bundle(path: str):
    """Load a bundled hybrid models artifact (saved by train_hybrid as `hybrid_all.joblib`).

    Returns the deserialized artifact (dict) with keys: `models`, `feature_cols`, `metrics`, `specs`.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Hybrid bundle not found: {path}")
    return joblib.load(path)


def predict_hybrid_from_bundle(bundle: dict, turbine_name: str, df: pd.DataFrame, physics_power_col: str = "P_physics_w", max_output_w: Optional[float] = None) -> pd.Series:
    """Convenience wrapper to predict hybrid power for `turbine_name` using a loaded bundle.

    - `bundle` is the dict returned by `load_hybrid_bundle`.
    - `df` must contain the feature columns listed for the turbine in `bundle['feature_cols']`.

    Returns a pandas Series of predicted hybrid power (W) aligned with `df`.
    """
    models = bundle.get("models", {})
    feature_map = bundle.get("feature_cols", {})
    if turbine_name not in models:
        raise KeyError(f"Turbine '{turbine_name}' not found in bundle")
    model = models[turbine_name]
    feature_cols = feature_map.get(turbine_name)
    if feature_cols is None:
        raise KeyError(f"Feature columns for turbine '{turbine_name}' not found in bundle")

    # Ensure df has the feature columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        # create missing columns as zeros to avoid runtime errors
        for c in missing:
            df[c] = 0.0

    X = df[feature_cols].ffill().fillna(0.0).values
    pred_residual = model.predict(X)
    p_phys = df[physics_power_col].values if physics_power_col in df.columns else np.zeros(len(df))
    p_hybrid = p_phys + pred_residual
    p_hybrid = np.where(np.isnan(p_hybrid), np.nan, np.maximum(0.0, p_hybrid))
    if max_output_w is not None:
        p_hybrid = np.minimum(p_hybrid, float(max_output_w))
    return pd.Series(p_hybrid, index=df.index)
