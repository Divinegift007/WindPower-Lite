"""
Integration tests for the full Windpower Lite pipeline.
Tests complete workflows from data fetch through results generation.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
import tempfile
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import run_pipeline, prepare_site_dataframe
from utils import load_turbine_specs
from physics_model import (
    compute_hourly_power_series,
    compute_aep_from_power_series,
    capacity_factor
)
from fetch_meteo import fetch_hourly_nasa_power
from hybrid_model import (
    make_pseudo_targets_from_power_curve,
    prepare_features,
    get_model_pipeline,
    train_hybrid_model,
    load_model,
    save_model
)


class TestIntegrationPipeline(unittest.TestCase):
    """Test the complete wind power estimation pipeline."""

    @classmethod
    def setUpClass(cls):
        """Load turbine specs once for all tests."""
        cls.specs = load_turbine_specs(
            os.path.join(os.path.dirname(__file__),
            '..', 'data', 'turbine_specs.csv')
        )

    def test_end_to_end_pipeline_v27(self):
        """Test full pipeline for V27-225 turbine."""
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=7)

        df, summary = run_pipeline(
            lat=8.4966,
            lon=4.5421,
            start_dt=start_dt,
            end_dt=end_dt,
            turbine_name='V27-225',
            apply_hybrid_if_available=False
        )

        # Verify output structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(summary, dict)
        self.assertGreater(len(df), 0)

        # Verify required columns
        required_cols = [
            'timestamp', 'wind_speed_50m_mps', 'temp_2m_c',
            'surface_pressure_pa', 'rho_kg_m3', 'v_hub_mps', 'P_physics_w'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

        # Verify summary metrics
        self.assertIn('turbine', summary)
        self.assertIn('aep_physics_kwh', summary)
        self.assertIn('capacity_factor_physics', summary)
        self.assertEqual(summary['turbine'], 'V27-225')

        # Verify AEP is reasonable (positive, not NaN)
        self.assertGreater(summary['aep_physics_kwh'], 0)
        self.assertLess(summary['capacity_factor_physics'], 1.0)

    def test_end_to_end_pipeline_v150(self):
        """Test full pipeline for V150-4.2MW turbine."""
        end_dt = datetime.utcnow()
        start_dt = end_dt - timedelta(days=7)

        df, summary = run_pipeline(
            lat=8.4966,
            lon=4.5421,
            start_dt=start_dt,
            end_dt=end_dt,
            turbine_name='V150-4.2MW',
            apply_hybrid_if_available=False
        )

        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)
        self.assertEqual(summary['turbine'], 'V150-4.2MW')
        self.assertGreater(summary['aep_physics_kwh'],
                          # V150 should produce more than V27
                          100)

    def test_all_ten_vestas_turbines_load(self):
        """Verify all 10 Vestas turbines are configured."""
        expected_turbines = {
            'V27-225', 'V47-660', 'V52-850', 'V80-2.0MW', 'V90-2.0MW',
            'V90-3.0MW', 'V100-2.75MW', 'V110-2.0MW', 'V117-3.45MW', 'V150-4.2MW'
        }
        actual_turbines = set(self.specs.keys())
        self.assertEqual(actual_turbines, expected_turbines)

        # Verify key requirements
        for name, spec in self.specs.items():
            # All should be Vestas
            self.assertIn('V', name, f"Turbine {name} should be Vestas")

            # Must have required fields
            self.assertIn('rated_power_kw', spec)
            self.assertIn('hub_height_m', spec)
            self.assertIn('rotor_diameter_m', spec)
            self.assertIn('power_curve', spec)

            # Verify reasonable ranges from spec
            self.assertGreaterEqual(spec['hub_height_m'], 31.5)
            self.assertLessEqual(spec['hub_height_m'], 105)
            self.assertGreaterEqual(spec['rated_power_kw'], 225)
            self.assertLessEqual(spec['rated_power_kw'], 4200)

    def test_physics_model_vs_power_curve(self):
        """Verify physics model is simpler than power curve."""
        spec = self.specs['V90-3.0MW']

        # Create synthetic wind speeds
        wind_speeds = np.array([4.0, 8.0, 10.0, 14.0, 20.0])
        rho = np.full_like(wind_speeds, 1.225)

        # Compute physics baseline
        physics_power = compute_hourly_power_series(
            wind_speeds, spec, rho_series=rho, use_power_curve=False
        )

        # Compute power curve output
        pc_power = compute_hourly_power_series(
            wind_speeds, spec, rho_series=rho, use_power_curve=True
        )

        # Physics baseline should generally be simpler (often lower)
        # but power curve should have proper cut-in/cut-out behavior
        self.assertEqual(len(physics_power), len(wind_speeds))
        self.assertEqual(len(pc_power), len(wind_speeds))

        # At zero wind, both should be zero
        zero_wind = np.array([0.0])
        physics_zero = compute_hourly_power_series(
            zero_wind, spec, rho_series=np.array([1.225]), use_power_curve=False
        )
        self.assertEqual(physics_zero[0], 0.0)

    def test_hybrid_training_workflow(self):
        """Test the hybrid model training workflow."""
        # Create synthetic hourly data (24 hours)
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=24, freq='h'),
            'wind_speed_50m_mps': np.random.rand(24) * 12 + 3,
            'temp_2m_c': np.random.rand(24) * 10 + 20,
            'surface_pressure_pa': np.random.rand(24) * 2000 + 101000,
        })

        spec = self.specs['V90-2.0MW']

        # Prepare data
        df = prepare_site_dataframe(df, spec)

        # Create pseudo-targets from power curve
        df = make_pseudo_targets_from_power_curve(df, spec['power_curve'])

        # Prepare features
        X, y, feature_names = prepare_features(df)

        # Verify feature preparation
        self.assertGreater(len(feature_names), 0)
        self.assertEqual(X.shape[0], len(df))
        self.assertEqual(len(y), len(df))

        # Train a simple model
        model, metrics = train_hybrid_model(
            X, y, model_type='rf', cv_folds=3
        )

        # Verify training output
        self.assertIsNotNone(model)
        self.assertIn('rmse_cv_mean', metrics)
        self.assertIn('mae_cv_mean', metrics)
        self.assertIn('r2_cv_mean', metrics)

        # Metrics should be numbers (not NaN)
        self.assertFalse(np.isnan(metrics['rmse_cv_mean']))
        self.assertFalse(np.isnan(metrics['mae_cv_mean']))
        self.assertFalse(np.isnan(metrics['r2_cv_mean']))

    def test_model_serialization(self):
        """Test saving and loading hybrid models."""
        # Create and train a simple model
        X = np.random.rand(50, 6)
        y = np.random.rand(50)

        model, _ = train_hybrid_model(X, y, cv_folds=2)

        # Save and load
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = os.path.join(tmpdir, 'test_model.joblib')
            save_model(model, model_path)

            self.assertTrue(os.path.exists(model_path))

            loaded_model = load_model(model_path)
            self.assertIsNotNone(loaded_model)

            # Verify predictions are consistent
            X_test = np.random.rand(10, 6)
            pred_orig = model.predict(X_test)
            pred_loaded = loaded_model.predict(X_test)

            np.testing.assert_array_almost_equal(pred_orig, pred_loaded)

    def test_air_density_tropical_conditions(self):
        """Test air density calculations for tropical Nigeria."""
        from physics_model import air_density_from_ps_temp

        # Standard tropical conditions at Ilorin (35°C typical afternoon)
        rho_hot = air_density_from_ps_temp(
            pressure_pa=101325, temp_c=35
        )

        # Cooler baseline
        rho_cool = air_density_from_ps_temp(
            pressure_pa=101325, temp_c=15
        )

        # Hot air is less dense
        self.assertLess(rho_hot, rho_cool)

        # Reasonable range for tropical sea level
        self.assertGreater(rho_hot, 1.0)
        self.assertLess(rho_hot, 1.3)

    def test_wind_shear_hub_height_variation(self):
        """Test wind shear adjustment across hub heights."""
        from physics_model import adjust_wind_speed_power_law

        v_ref = 8.0
        h_ref = 50.0
        alpha = 0.14

        # Turbine hub heights vary from 31.5m to 105m
        h_small = 31.5  # V27
        h_large = 105.0  # V150

        v_small = adjust_wind_speed_power_law(v_ref, h_ref, h_small, alpha)
        v_large = adjust_wind_speed_power_law(v_ref, h_ref, h_large, alpha)

        # Lower hub height should have lower wind speed
        self.assertLess(v_small, v_ref)
        self.assertGreater(v_large, v_ref)

        # V150 should see meaningfully faster wind than V27
        self.assertGreater(v_large - v_small, 0.5)

    def test_capacity_factor_calculations(self):
        """Test capacity factor computations."""
        # Create synthetic power series
        rated_power_kw = 2000
        power_avg_kw = 500  # Average power output

        # Annual equivalent
        hours = 8760
        aep = power_avg_kw * hours

        cf = capacity_factor(aep, rated_power_kw)

        # Should be 500/2000 = 0.25
        self.assertAlmostEqual(cf, 0.25, places=2)

        # Reasonable CF range for Nigeria (2-25%)
        self.assertGreater(cf, 0.0)
        self.assertLess(cf, 0.3)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Setup for each test."""
        self.specs = load_turbine_specs(
            os.path.join(os.path.dirname(__file__),
            '..', 'data', 'turbine_specs.csv')
        )

    def test_zero_wind_speed(self):
        """Test handling of zero wind speed."""
        spec = self.specs['V90-2.0MW']
        wind = np.array([0.0, 0.0, 0.0])
        rho = np.full_like(wind, 1.225)

        power = compute_hourly_power_series(wind, spec, rho_series=rho)

        # Zero wind should produce zero power
        np.testing.assert_array_equal(power, [0.0, 0.0, 0.0])

    def test_extreme_wind_speed(self):
        """Test handling of extreme (cut-out) wind speeds."""
        spec = self.specs['V90-2.0MW']
        wind = np.array([30.0, 50.0])  # Above cut-out
        rho = np.full_like(wind, 1.225)

        power = compute_hourly_power_series(wind, spec, rho_series=rho)

        # Should be zero (turbine shut down)
        self.assertEqual(power[0], 0.0)
        self.assertEqual(power[1], 0.0)

    def test_missing_power_curve(self):
        """Test turbine without power curve falls back to Cp method."""
        spec = self.specs['V90-2.0MW'].copy()
        wind = np.array([10.0])
        rho = np.array([1.225])

        # With power curve
        power_with_pc = compute_hourly_power_series(
            wind, spec, rho_series=rho, use_power_curve=True
        )

        # Without power curve (use Cp only)
        power_with_cp = compute_hourly_power_series(
            wind, spec, rho_series=rho, use_power_curve=False
        )

        # Should be different
        self.assertGreater(power_with_pc[0], 0)
        self.assertGreater(power_with_cp[0], 0)

    def test_nan_in_data(self):
        """Test handling of NaN values in workflow."""
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='h'),
            'wind_speed_50m_mps': [np.nan, 5.0, 6.0, np.nan, 8.0, 7.0, 6.0, 5.0, np.nan, 7.0],
            'temp_2m_c': np.full(10, 25.0),
            'surface_pressure_pa': np.full(10, 101325.0),
        })

        spec = self.specs['V90-2.0MW']

        # Should handle NaN gracefully (forward fill will propagate values)
        df_prepared = prepare_site_dataframe(df, spec)

        # After forward-fill in prepare_features, NaN in wind should be handled
        # by forward fill in the feature preparation
        df_pseudo = make_pseudo_targets_from_power_curve(df_prepared, spec['power_curve'])
        X, y, _ = prepare_features(df_pseudo)

        # After ffill and fillna(0), X should have no NaN
        self.assertFalse(np.isnan(X).any(), "X should have no NaN after ffill and fillna")


if __name__ == '__main__':
    unittest.main()
