from types import SimpleNamespace
import unittest

import numpy as np

from src.simulate import diagnose_convergence, integrate_until_converged


def _solution(t: np.ndarray, prey: np.ndarray, predator: np.ndarray):
    return SimpleNamespace(t=t, y=np.vstack([prey, predator]))


class ConvergenceDiagnosticsTests(unittest.TestCase):
    def test_steady_state_converges(self):
        t = np.linspace(0.0, 10.0, 100)
        sol = _solution(t, np.ones(100), np.full(100, 2.0))
        self.assertEqual(
            diagnose_convergence(sol, scales=(1.0, 2.0)).status,
            "converged",
        )

    def test_stationary_periodic_trajectory_converges(self):
        t = np.linspace(0.0, 10.0, 101)
        prey = 2.0 + np.sin(2.0 * np.pi * t)
        predator = 3.0 + np.cos(2.0 * np.pi * t)
        sol = _solution(t, prey, predator)
        self.assertEqual(
            diagnose_convergence(sol, scales=(2.0, 3.0)).status,
            "converged",
        )

    def test_decaying_transient_does_not_converge(self):
        t = np.linspace(0.0, 10.0, 100)
        prey = 1.0 + np.exp(-0.1 * t)
        sol = _solution(t, prey, np.ones(100))
        self.assertEqual(
            diagnose_convergence(sol, scales=(1.0, 1.0)).status,
            "not_converged",
        )

    def test_slow_monotonic_drift_does_not_converge(self):
        t = np.linspace(0.0, 100.0, 1000)
        prey = 1000.0 + 0.1 * t
        sol = _solution(t, prey, np.ones(1000))
        self.assertEqual(
            diagnose_convergence(sol, scales=(1000.0, 1.0)).status,
            "not_converged",
        )

    def test_extension_sequence_doubles_horizon_and_density(self):
        calls = []

        def integrator(t_end: float, n_points: int):
            calls.append((t_end, n_points))
            t = np.linspace(0.0, t_end, n_points)
            prey = 1.0 + t / t_end
            return _solution(t, prey, np.ones(n_points))

        result = integrate_until_converged(
            integrator,
            t_end=10.0,
            scales=(1.0, 1.0),
            n_points=100,
            max_extensions=3,
        )
        self.assertEqual(calls, [(10.0, 100), (20.0, 200), (40.0, 400), (80.0, 800)])
        self.assertEqual(result.convergence.status, "not_converged")
        self.assertEqual(result.extensions, 3)

    def test_near_zero_amplitude_is_treated_as_zero(self):
        t = np.linspace(0.0, 10.0, 100)
        noise = np.linspace(0.0, 1e-8, 100)
        sol = _solution(t, 1.0 + noise, 2.0 - noise)
        diagnostics = diagnose_convergence(sol, scales=(1.0, 2.0))
        self.assertEqual(diagnostics.amplitude_relative_changes, (0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
