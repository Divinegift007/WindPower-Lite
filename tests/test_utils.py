import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils import load_turbine_specs


class TestUtils(unittest.TestCase):
    def test_load_turbine_specs(self):
        specs = load_turbine_specs()
        self.assertIsInstance(specs, dict)
        self.assertGreater(len(specs), 0)
        # Check first spec has expected keys
        first = list(specs.values())[0]
        self.assertIn('rated_power_kw', first)
        self.assertIn('power_curve', first)


if __name__ == '__main__':
    unittest.main()
