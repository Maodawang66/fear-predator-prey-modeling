import unittest

import numpy as np

from data.series import PredatorPreySeries
from src.fit import (
    _multiseed_bounded_minimize,
    conditional_fear_memory_m0_scan,
    fit_baseline_to_series,
    fit_bda_fear_to_series,
    fit_fear_memory_to_series,
    profile_fear_memory_delta,
    profile_fear_memory_m0,
)
from src.model import baseline_rhs, fear_memory_rhs
from src.parameters import BaselineParams, FearMemoryParams
from src.simulate import integrate_rhs


def _baseline_series() -> PredatorPreySeries:
    params = BaselineParams(r=0.8, K=30.0, a=0.08, theta=0.03, e=0.7, mu=0.25)
    t = np.linspace(0.0, 8.0, 17)
    sol = integrate_rhs(
        lambda time, state: baseline_rhs(time, state, params),
        np.array([12.0, 3.0]),
        t_span=(float(t[0]), float(t[-1])),
        n_points=801,
    )
    return PredatorPreySeries(
        name="synthetic_baseline",
        t=t,
        prey=np.interp(t, sol.t, sol.y[0]),
        predator=np.interp(t, sol.t, sol.y[1]),
    )


def _fear_series(delta: float, m0: float = 3.0) -> PredatorPreySeries:
    params = FearMemoryParams(
        r=0.8,
        K=30.0,
        a=0.08,
        theta=0.03,
        e=0.7,
        mu=0.25,
        phi=0.015,
        delta=delta,
    )
    t = np.linspace(0.0, 8.0, 17)
    sol = integrate_rhs(
        lambda time, state: fear_memory_rhs(time, state, params),
        np.array([12.0, 3.0, m0]),
        t_span=(float(t[0]), float(t[-1])),
        n_points=801,
    )
    return PredatorPreySeries(
        name="synthetic_fear",
        t=t,
        prey=np.interp(t, sol.t, sol.y[0]),
        predator=np.interp(t, sol.t, sol.y[1]),
    )


class FullParameterFitTests(unittest.TestCase):
    def test_main_fitters_reject_absorbing_zero_initial_state(self):
        series = PredatorPreySeries(
            name="zero_initial",
            t=np.arange(5, dtype=float),
            prey=np.array([0.0, 1.0, 1.2, 1.1, 1.0]),
            predator=np.array([0.0, 0.2, 0.3, 0.25, 0.2]),
        )
        for fitter in (
            fit_baseline_to_series,
            fit_fear_memory_to_series,
            fit_bda_fear_to_series,
        ):
            with self.subTest(fitter=fitter.__name__):
                with self.assertRaisesRegex(ValueError, "first joint positive observation"):
                    fitter(series)

    def test_baseline_fits_e_mu_by_default(self):
        result = fit_baseline_to_series(
            _baseline_series(),
            optimizer="local",
            max_nfev=300,
        )
        self.assertEqual(result.n_parameters, 6)
        self.assertNotAlmostEqual(result.params["e"], BaselineParams().e)
        self.assertNotAlmostEqual(result.params["mu"], BaselineParams().mu)

    def test_baseline_fixed_e_mu_mode_remains_available(self):
        fixed = BaselineParams(e=0.33, mu=0.44)
        result = fit_baseline_to_series(
            _baseline_series(),
            fixed=fixed,
            fit_e_mu=False,
            optimizer="local",
            max_nfev=300,
        )
        self.assertEqual(result.n_parameters, 4)
        self.assertAlmostEqual(result.params["e"], fixed.e)
        self.assertAlmostEqual(result.params["mu"], fixed.mu)

    def test_fear_memory_fits_e_mu_phi_and_keeps_delta_fixed(self):
        delta = 0.6
        result = fit_fear_memory_to_series(
            _fear_series(delta),
            fixed=FearMemoryParams(delta=delta),
            optimizer="local",
            max_nfev=400,
        )
        self.assertEqual(result.n_parameters, 7)
        self.assertAlmostEqual(result.params["delta"], delta)
        self.assertIn("e", result.params)
        self.assertIn("mu", result.params)
        self.assertIn("phi", result.params)

    def test_fear_memory_accepts_fixed_initial_memory(self):
        result = fit_fear_memory_to_series(
            _fear_series(0.6, m0=1.5),
            fixed=FearMemoryParams(delta=0.6),
            initial_memory=1.5,
            optimizer="local",
            max_nfev=200,
        )
        self.assertAlmostEqual(result.params["m0"], 1.5)
        self.assertEqual(result.meta["initial_memory_source"], "fixed_input")
        with self.assertRaises(ValueError):
            fit_fear_memory_to_series(_fear_series(0.6), initial_memory=-1.0)

    def test_m0_sensitivity_and_profile_report_holdout_metrics(self):
        series = _fear_series(0.6, m0=1.5)
        base = {
            "r": 0.8, "K": 30.0, "a": 0.08, "theta": 0.03,
            "e": 0.7, "mu": 0.25, "phi": 0.015, "delta": 0.6,
        }
        conditional = conditional_fear_memory_m0_scan(
            series,
            base,
            m0_ratio_grid=np.array([0.5, 1.0]),
        )
        profile = profile_fear_memory_m0(
            series,
            base,
            m0_ratio_grid=np.array([0.5, 1.0]),
            max_nfev=100,
        )
        self.assertEqual(len(conditional), 2)
        self.assertEqual(len(profile), 2)
        self.assertAlmostEqual(profile[0]["m0"], 1.5)
        self.assertIn("validation_rmse_normalized_total", profile[0])
        self.assertIn("profile_likelihood_ratio", profile[0])

    def test_delta_profile_reports_memory_timescale_and_holdout_metrics(self):
        series = _fear_series(0.6)
        base = {
            "r": 0.8, "K": 30.0, "a": 0.08, "theta": 0.03,
            "e": 0.7, "mu": 0.25, "phi": 0.015, "delta": 0.6, "m0": 3.0,
        }
        profile = profile_fear_memory_delta(
            series,
            base,
            delta_grid=np.array([0.3, 0.6]),
            max_nfev=100,
        )
        self.assertEqual(len(profile), 2)
        self.assertAlmostEqual(profile[0]["memory_timescale"], 1.0 / 0.3)
        self.assertIn("validation_rmse_normalized_total", profile[0])
        self.assertIn("inside_confidence_interval", profile[0])

    def test_multiseed_is_reproducible_and_records_selection(self):
        def residual(values: np.ndarray) -> np.ndarray:
            return values - np.array([0.2, -0.3, 0.1, 0.4, -0.2])

        args = (
            residual,
            np.zeros(5),
            np.full(5, -1.0),
            np.full(5, 1.0),
        )
        first, first_meta = _multiseed_bounded_minimize(
            *args,
            max_nfev=40,
            seeds=(0, 1),
            param_names=["a", "b", "c", "d", "e"],
        )
        second, second_meta = _multiseed_bounded_minimize(
            *args,
            max_nfev=40,
            seeds=(0, 1),
            param_names=["a", "b", "c", "d", "e"],
        )
        np.testing.assert_allclose(first.x, second.x)
        self.assertAlmostEqual(first.cost, second.cost)
        self.assertEqual(first_meta["selected_seed"], second_meta["selected_seed"])
        self.assertEqual(len(first_meta["optimizer_runs"]), 2)
        self.assertIn("parameter_bound_hits", first_meta["optimizer_runs"][0])
        self.assertIn("parameter_bound_hits", first_meta["local_refinement"])
        self.assertLessEqual(
            first.cost,
            min(run["objective_value"] for run in first_meta["optimizer_runs"]),
        )


if __name__ == "__main__":
    unittest.main()
