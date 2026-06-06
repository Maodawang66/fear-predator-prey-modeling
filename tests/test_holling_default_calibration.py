import unittest

import numpy as np

from data.calibrate_holling_defaults import _series_diagnostics, empirical_target
from data.series import PredatorPreySeries


def _series(name: str, t: np.ndarray, prey: np.ndarray, predator: np.ndarray):
    return PredatorPreySeries(name=name, t=t, prey=prey, predator=predator)


class HollingDefaultCalibrationTests(unittest.TestCase):
    def test_joint_periodic_series_is_eligible(self):
        t = np.linspace(0.0, 40.0, 401)
        prey = 2.0 + np.sin(2.0 * np.pi * t / 5.0)
        predator = 3.0 + np.sin(2.0 * np.pi * t / 5.0 + 0.5)
        diagnostics = _series_diagnostics(_series("periodic", t, prey, predator))
        self.assertEqual(diagnostics["classification"], "periodic")
        self.assertTrue(diagnostics["included_in_target"])

    def test_one_sided_periodicity_is_excluded(self):
        t = np.linspace(0.0, 40.0, 401)
        prey = 2.0 + np.sin(2.0 * np.pi * t / 5.0)
        predator = np.linspace(2.0, 3.0, t.size)
        diagnostics = _series_diagnostics(_series("one_sided", t, prey, predator))
        self.assertFalse(diagnostics["included_in_target"])

    def test_non_increasing_time_is_excluded(self):
        t = np.array([0.0, 1.0, 1.0, 2.0])
        diagnostics = _series_diagnostics(
            _series("bad_time", t, np.ones(4), np.ones(4))
        )
        self.assertEqual(diagnostics["classification"], "invalid_time")

    def test_no_eligible_series_returns_explicit_stop_status(self):
        t = np.linspace(0.0, 10.0, 101)
        trend = np.linspace(1.0, 5.0, t.size)
        summary, rows = empirical_target([_series("trend", t, trend, trend)], K=100.0)
        self.assertEqual(summary["status"], "insufficient_eligible_series")
        self.assertIsNone(summary["target_x"])
        self.assertIsNone(summary["target_y"])
        self.assertEqual(rows[0]["classification"], "nonstationary_trend")


if __name__ == "__main__":
    unittest.main()
