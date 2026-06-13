import os
import sys
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
from fetch_meteo import fetch_hourly_nasa_power


class DummyResponse:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data

    def raise_for_status(self):
        # always succeed
        return None


def make_fake_json(ps_values, ws_values, t_values, time_keys):
    # Construct minimal NASA POWER style response
    params = {}
    for key, vals in [('WS50M', ws_values), ('T2M', t_values), ('PS', ps_values)]:
        params[key] = dict(zip(time_keys, vals))
    return {'properties': {'parameter': params}}


class TestFetchMeteo(unittest.TestCase):
    def setUp(self):
        # patch requests.get on the fly
        import fetch_meteo
        self.orig_get = fetch_meteo.requests.get
        fetch_meteo.requests.get = self.fake_get

    def tearDown(self):
        import fetch_meteo
        fetch_meteo.requests.get = self.orig_get

    def fake_get(self, url, params=None, timeout=None):
        # return the dummy response we stored on the instance
        return DummyResponse(self.fake_data)

    def test_pressure_unit_conversion(self):
        # create fake data with PS in kPa (~100), wind speed and temp arbitrary
        time_keys = ['2020010100']
        self.fake_data = make_fake_json(ps_values=[97.0], ws_values=[5.0], t_values=[15.0], time_keys=time_keys)
        df = fetch_hourly_nasa_power(lat=0, lon=0, start_dt=datetime(2020,1,1), end_dt=datetime(2020,1,1))
        # pressure should be converted to Pa (97 kPa -> 97000 Pa)
        self.assertTrue(df['surface_pressure_pa'].iloc[0] > 90000)
        self.assertLess(df['surface_pressure_pa'].iloc[0], 200000)

    def test_sentinel_values_are_nan(self):
        time_keys = ['2020010100', '2020010101']
        # first timestamp has all sentinels, second has a mixture
        self.fake_data = make_fake_json(
            ps_values=[-999.0, 1013.0],
            ws_values=[-999.0, 3.1],
            t_values=[-999.0, -999.0],
            time_keys=time_keys,
        )
        df = fetch_hourly_nasa_power(lat=0, lon=0, start_dt=datetime(2020,1,1), end_dt=datetime(2020,1,2))
        # first row should be NaN everywhere
        self.assertTrue(pd.isna(df['wind_speed_50m_mps'].iloc[0]))
        self.assertTrue(pd.isna(df['temp_2m_c'].iloc[0]))
        self.assertTrue(pd.isna(df['surface_pressure_pa'].iloc[0]))
        # second row still has NaN temp and wind sentinel replaced
        self.assertTrue(pd.isna(df['temp_2m_c'].iloc[1]))
        self.assertFalse(pd.isna(df['wind_speed_50m_mps'].iloc[1]))
        # pressure 1013 should have been interpreted as kPa -> Pa
        self.assertGreater(df['surface_pressure_pa'].iloc[1], 100000)


if __name__ == '__main__':
    unittest.main()
