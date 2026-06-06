from types import SimpleNamespace
import unittest

import numpy as np

from src.simulate import extinction_diagnostics, is_extinct


def _solution(prey: np.ndarray, predator: np.ndarray):
    return SimpleNamespace(
        t=np.linspace(0.0, 10.0, prey.size),
        y=np.vstack([prey, predator]),
    )


class ExtinctionClassificationTests(unittest.TestCase):
    def test_stable_coexistence(self):
        sol = _solution(np.ones(100), np.ones(100))
        self.assertEqual(is_extinct(sol, scales=(100.0, 100.0)), "coexist")

    def test_oscillatory_low_trough_is_not_extinct(self):
        prey = np.ones(100)
        prey[85:90] = 0.01
        sol = _solution(prey, np.ones(100))
        self.assertEqual(is_extinct(sol, scales=(100.0, 100.0)), "coexist")

    def test_persistent_species_extinction(self):
        sol = _solution(np.full(100, 0.01), np.ones(100))
        diagnostics = extinction_diagnostics(sol, scales=(100.0, 100.0))
        self.assertEqual(diagnostics.status, "prey_extinct")
        self.assertGreaterEqual(diagnostics.below_fractions[0], 0.8)

    def test_recent_recovery_is_not_extinct(self):
        prey = np.full(100, 0.01)
        prey[97] = 0.2
        sol = _solution(prey, np.ones(100))
        self.assertEqual(is_extinct(sol, scales=(100.0, 100.0)), "coexist")

    def test_scale_invariance(self):
        normalized_prey = np.full(100, 0.0005)
        normalized_predator = np.ones(100)
        small = _solution(normalized_prey, normalized_predator)
        large = _solution(normalized_prey * 100.0, normalized_predator * 100.0)
        self.assertEqual(
            is_extinct(small, scales=(1.0, 1.0)),
            is_extinct(large, scales=(100.0, 100.0)),
        )

    def test_legacy_absolute_threshold_remains_supported(self):
        sol = _solution(np.full(100, 0.01), np.ones(100))
        self.assertEqual(is_extinct(sol, threshold=0.1), "prey_extinct")
        self.assertEqual(is_extinct(sol), "coexist")


if __name__ == "__main__":
    unittest.main()
