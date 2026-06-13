import os
import sys
import unittest
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from physics_model import (
    rotor_swept_area,
    adjust_wind_speed_power_law,
    physics_power_from_cp,
    baseline_turbine_power,
    air_density_from_ps_temp,
)


class TestPhysicsModel(unittest.TestCase):
    def test_rotor_area(self):
        d = 10.0
        A = rotor_swept_area(d)
        self.assertAlmostEqual(A, math.pi * (d / 2.0) ** 2.0)

    def test_adjust_wind_speed_power_law(self):
        v_ref = 5.0
        v = adjust_wind_speed_power_law(v_ref, 50.0, 100.0, alpha=0.14)
        self.assertGreater(v, 0.0)

    def test_physics_power_positive(self):
        p = physics_power_from_cp(5.0, rho=1.225, rotor_diameter_m=6.7, cp=0.39, eta_sys=0.9)
        self.assertGreater(p, 0.0)

    def test_air_density(self):
        # known sea-level conditions should give approx 1.225 kg/m^3
        rho = air_density_from_ps_temp(101325.0, 15.0)
        self.assertAlmostEqual(rho, 1.225, places=3)

    def test_baseline_turbine_power_cutoffs(self):
        spec = {
            'rated_power_kw': 10.0,
            'rotor_diameter_m': 6.7,
            'hub_height_m': 30.0,
            'cut_in_mps': 3.0,
            'rated_mps': 11.0,
            'cut_out_mps': 20.0,
            'cp': 0.39,
            'eta_sys': 0.9,
            'power_curve': None,
        }
        self.assertEqual(baseline_turbine_power(0.0, spec, rho=1.225, use_power_curve=False), 0.0)
        self.assertEqual(baseline_turbine_power(25.0, spec, rho=1.225, use_power_curve=False), 0.0)
        p = baseline_turbine_power(6.0, spec, rho=1.225, use_power_curve=False)
        self.assertGreater(p, 0.0)


if __name__ == '__main__':
    unittest.main()
