import unittest

import numpy as np

from data.series import PredatorPreySeries
from src.fear_pathway_fit import (
    HOLLING_FEAR_PATHWAYS,
    fit_holling_baseline_to_series,
    fit_holling_fear_pathway_to_series,
)
from src.model import (
    baseline_rhs,
    fear_foraging_rhs,
    fear_handling_rhs,
    fear_instant_rhs,
    fear_memory_rhs,
    fear_saturating_rhs,
)
from src.parameters import (
    BaselineParams,
    FearForagingParams,
    FearHandlingParams,
    FearMemoryParams,
    FearSaturatingParams,
)
from src.simulate import integrate_rhs


class FearPathwayComparisonTests(unittest.TestCase):
    def test_all_fear_pathways_nest_to_baseline_at_zero_strength(self):
        baseline = BaselineParams()
        state = np.array([20.0, 4.0])
        expected = baseline_rhs(0.0, state, baseline)
        candidates = [
            fear_instant_rhs(0.0, state, FearMemoryParams(phi=0.0)),
            fear_memory_rhs(0.0, np.array([20.0, 4.0, 7.0]), FearMemoryParams(phi=0.0))[:2],
            fear_saturating_rhs(0.0, state, FearSaturatingParams(phi=0.0)),
            fear_foraging_rhs(0.0, state, FearForagingParams(psi=0.0)),
            fear_handling_rhs(0.0, state, FearHandlingParams(psi=0.0)),
        ]
        for candidate in candidates:
            np.testing.assert_allclose(candidate, expected, rtol=1e-12, atol=1e-12)

    def test_each_fitted_pathway_has_one_more_parameter_than_baseline(self):
        params = FearMemoryParams(r=0.8, K=30.0, a=0.08, theta=0.03, e=0.7, mu=0.25, phi=0.03)
        t = np.linspace(0.0, 5.0, 12)
        sol = integrate_rhs(
            lambda time, state: fear_instant_rhs(time, state, params),
            np.array([12.0, 3.0]),
            t_span=(0.0, 5.0),
            n_points=501,
        )
        series = PredatorPreySeries(
            "synthetic",
            t,
            np.interp(t, sol.t, sol.y[0]),
            np.interp(t, sol.t, sol.y[1]),
        )
        baseline = fit_holling_baseline_to_series(series, optimizer="local", max_nfev=20)
        self.assertEqual(baseline.n_parameters, 6)
        for pathway in HOLLING_FEAR_PATHWAYS:
            result = fit_holling_fear_pathway_to_series(
                series, pathway, optimizer="local", max_nfev=20
            )
            self.assertEqual(result.n_parameters, baseline.n_parameters + 1)
            self.assertEqual(result.meta["validation_mode"], "ordered_holdout_continuous_multistep")
            self.assertIn(result.meta["fear_parameter"], ("phi", "psi"))


if __name__ == "__main__":
    unittest.main()
