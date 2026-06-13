import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from data.series import PredatorPreySeries
from data.deep_data_analysis import summarize_memory_profiles
from src.fit import (
    _OptResult,
    adaptive_fear_memory_phi_bound_diagnostic,
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

    def test_fear_memory_selects_nested_baseline_candidate_and_keeps_seven_parameter_aicc(self):
        series = _baseline_series()
        baseline = fit_baseline_to_series(series, optimizer="local", max_nfev=300)
        baseline_params = BaselineParams(**{
            name: baseline.params[name] for name in ("r", "K", "a", "theta", "e", "mu")
        })
        deliberately_poor = _OptResult(
            x=np.log([1.0, 15.0, 0.5, 0.5, 0.5, 0.5, 0.1]),
            success=False,
            status="failed",
            message="synthetic failed optimizer result",
            nfev=1,
            cost=0.0,
            bound_hit_indices=[],
        )
        with patch("src.fit._multiseed_bounded_minimize", return_value=(deliberately_poor, {})):
            result = fit_fear_memory_to_series(
                series,
                baseline_params=baseline_params,
                baseline_candidate_status="usable_limit",
                baseline_candidate_message="baseline reached iteration limit",
                optimizer="local",
                max_nfev=1,
            )
        self.assertTrue(result.meta["nested_baseline_candidate_selected"])
        self.assertEqual(result.params["phi"], 0.0)
        self.assertEqual(result.n_parameters, 7)
        self.assertEqual(result.optimization_status, "usable_limit")
        self.assertLessEqual(
            result.meta["objective_value"],
            result.meta["nested_baseline_candidate_objective"] + 1e-10,
        )

    def test_fear_memory_rejects_nonfinite_optimized_objective(self):
        series = _baseline_series()
        baseline = fit_baseline_to_series(series, optimizer="local", max_nfev=300)
        baseline_params = BaselineParams(**{
            name: baseline.params[name] for name in ("r", "K", "a", "theta", "e", "mu")
        })
        nonfinite = _OptResult(
            x=np.log([1.0, 15.0, 0.5, 0.5, 0.5, 0.5, 0.1]),
            success=True,
            status="success",
            message="synthetic nonfinite optimizer result",
            nfev=1,
            cost=float("nan"),
            bound_hit_indices=[],
        )
        with patch("src.fit._multiseed_bounded_minimize", return_value=(nonfinite, {})):
            result = fit_fear_memory_to_series(
                series,
                baseline_params=baseline_params,
                optimizer="local",
                max_nfev=1,
            )
        self.assertTrue(result.meta["nested_baseline_candidate_selected"])
        self.assertTrue(np.isfinite(result.meta["objective_value"]))

    def test_adaptive_phi_bound_diagnostic_stops_when_upper_hit_resolves(self):
        seen: list[float] = []

        def fake_fit(series, **kwargs):
            upper = float(kwargs["phi_upper"])
            seen.append(upper)
            hit = upper < 5.0
            phi = upper if hit else 2.0
            return SimpleNamespace(
                params={
                    "phi": phi, "r": 1.0, "K": 2.0, "a": 0.1, "theta": 0.2,
                    "e": 0.5, "mu": 0.3, "delta": 1.0, "m0": 1.0,
                },
                meta={
                    "parameter_bound_hits": ["phi"] if hit else [],
                    "objective_value": 1.0,
                    "optimized_fear_objective": 1.0,
                    "nested_baseline_candidate_objective": 2.0,
                    "nested_baseline_candidate_selected": False,
                },
                rmse_normalized_total=0.1,
                validation_rmse_normalized_total=0.2,
                aicc=3.0,
                optimization_status="success",
            )

        rows = adaptive_fear_memory_phi_bound_diagnostic(
            _baseline_series(),
            BaselineParams(),
            fit_function=fake_fit,
        )
        self.assertEqual(seen, [0.2, 1.0, 5.0])
        self.assertTrue(rows[-1]["boundary_resolved"])
        self.assertFalse(rows[-1]["boundary_unresolved_at_maximum"])

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

    def test_memory_profile_summary_reports_both_grid_edges(self):
        rows = [
            {
                "series": "s",
                "m0_over_y0": value,
                "profile_rss": rss,
                "validation_rmse_normalized_total": validation,
                "optimization_status": "success",
                "inside_confidence_interval": inside,
            }
            for value, rss, validation, inside in (
                (0.0, 2.0, 0.3, True),
                (1.0, 1.0, 0.2, True),
                (5.0, 1.5, 0.1, True),
            )
        ]
        summary = summarize_memory_profiles(rows, [])
        self.assertEqual(len(summary), 1)
        self.assertTrue(summary[0]["confidence_set_touches_lower"])
        self.assertTrue(summary[0]["confidence_set_touches_upper"])
        self.assertEqual(summary[0]["profile_train_best_value"], 1.0)
        self.assertEqual(summary[0]["profile_holdout_best_value"], 5.0)
        self.assertEqual(
            summary[0]["holdout_selection_role"],
            "exploratory_not_unbiased_validation",
        )

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
