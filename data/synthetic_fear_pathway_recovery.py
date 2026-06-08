"""Noise-free synthetic recovery check for the five Holling-II fear pathways."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.series import PredatorPreySeries  # noqa: E402
from src.fear_pathway_fit import HOLLING_FEAR_PATHWAYS, fit_holling_fear_pathway_to_series  # noqa: E402
from src.fit import _simulate_at_times  # noqa: E402
from src.model import (  # noqa: E402
    fear_foraging_rhs,
    fear_handling_rhs,
    fear_instant_rhs,
    fear_memory_rhs,
    fear_saturating_rhs,
)
from src.parameters import (  # noqa: E402
    FearForagingParams,
    FearHandlingParams,
    FearMemoryParams,
    FearSaturatingParams,
)
from src.simulate import integrate_rhs  # noqa: E402

OUT = ROOT / "results" / "fear_pathway_comparison" / "synthetic_recovery.csv"
CORE = dict(r=0.8, K=30.0, a=0.08, theta=0.03, e=0.7, mu=0.25)


def _synthetic_series(pathway: str) -> PredatorPreySeries:
    rhs_and_params = {
        "fear_instant": (fear_instant_rhs, FearMemoryParams(**CORE, phi=0.08, delta=1.0)),
        "fear_memory": (fear_memory_rhs, FearMemoryParams(**CORE, phi=0.08, delta=0.5)),
        "fear_saturating": (fear_saturating_rhs, FearSaturatingParams(**CORE, phi=0.7, h=3.0)),
        "fear_foraging": (fear_foraging_rhs, FearForagingParams(**CORE, psi=0.35)),
        "fear_handling": (fear_handling_rhs, FearHandlingParams(**CORE, psi=0.35)),
    }
    rhs, params = rhs_and_params[pathway]
    initial = np.array([12.0, 3.0, 6.0]) if pathway == "fear_memory" else np.array([12.0, 3.0])
    t = np.linspace(0.0, 10.0, 31)
    solution = integrate_rhs(
        lambda time, state: rhs(time, state, params),
        initial,
        t_span=(0.0, 10.0),
        n_points=1001,
    )
    return PredatorPreySeries(
        name=f"synthetic_{pathway}",
        t=t,
        prey=np.interp(t, solution.t, solution.y[0]),
        predator=np.interp(t, solution.t, solution.y[1]),
    )


def _fixed_core_score(series: PredatorPreySeries, pathway: str) -> tuple[float, float]:
    rhs_by_pathway = {
        "fear_instant": fear_instant_rhs,
        "fear_memory": fear_memory_rhs,
        "fear_saturating": fear_saturating_rhs,
        "fear_foraging": fear_foraging_rhs,
        "fear_handling": fear_handling_rhs,
    }

    def objective(strength: float) -> float:
        if pathway in ("fear_instant", "fear_memory"):
            params = FearMemoryParams(**CORE, phi=strength, delta=0.5)
        elif pathway == "fear_saturating":
            params = FearSaturatingParams(**CORE, phi=strength, h=3.0)
        elif pathway == "fear_foraging":
            params = FearForagingParams(**CORE, psi=strength)
        else:
            params = FearHandlingParams(**CORE, psi=strength)
        initial = np.array([12.0, 3.0, 6.0]) if pathway == "fear_memory" else np.array([12.0, 3.0])
        prediction = _simulate_at_times(
            lambda time, state: rhs_by_pathway[pathway](time, state, params),
            initial,
            series.t,
        )
        residual = np.concatenate([
            (prediction[0] - series.prey) / max(float(np.max(series.prey)), 1.0),
            (prediction[1] - series.predator) / max(float(np.max(series.predator)), 1.0),
        ])
        return float(np.dot(residual, residual))

    upper = 0.99 if pathway == "fear_saturating" else 4.0
    result = minimize_scalar(objective, bounds=(0.0, upper), method="bounded")
    return float(result.fun), float(result.x)


def run_recovery(max_nfev: int = 400) -> list[dict]:
    rows: list[dict] = []
    for generating_pathway in HOLLING_FEAR_PATHWAYS:
        series = _synthetic_series(generating_pathway)
        fixed_core_scores = {
            candidate: _fixed_core_score(series, candidate)
            for candidate in HOLLING_FEAR_PATHWAYS
        }
        fixed_core_winner = min(fixed_core_scores, key=lambda candidate: fixed_core_scores[candidate][0])
        for candidate, (rss, strength) in fixed_core_scores.items():
            rows.append({
                "experiment": "fixed_core_single_parameter",
                "generating_pathway": generating_pathway,
                "candidate_pathway": candidate,
                "training_rmse": float(np.sqrt(rss / (2 * series.n_points))),
                "validation_rmse": "",
                "aicc": "",
                "optimization_status": "success",
                "fitted_fear_strength": strength,
                "winner": fixed_core_winner,
                "correctly_recovered": fixed_core_winner == generating_pathway,
            })

        candidates = []
        for candidate_pathway in HOLLING_FEAR_PATHWAYS:
            result = fit_holling_fear_pathway_to_series(
                series,
                candidate_pathway,
                optimizer="local",
                max_nfev=max_nfev,
                memory_delta=0.5 if candidate_pathway == "fear_memory" else 1.0,
                initial_memory=6.0 if candidate_pathway == "fear_memory" else None,
                saturating_half_response=3.0 if candidate_pathway == "fear_saturating" else None,
            )
            candidates.append(result)
        usable = [result for result in candidates if result.usable_for_comparison]
        winner = min(usable, key=lambda result: result.aicc).model if usable else ""
        for result in candidates:
            rows.append({
                "experiment": "joint_core_and_fear_refit",
                "generating_pathway": generating_pathway,
                "candidate_pathway": result.model,
                "training_rmse": result.rmse_normalized_total,
                "validation_rmse": result.validation_rmse_normalized_total,
                "aicc": result.aicc,
                "optimization_status": result.optimization_status,
                "fitted_fear_strength": result.params[result.meta["fear_parameter"]],
                "winner": winner,
                "correctly_recovered": winner == generating_pathway,
            })
    return rows


def main() -> None:
    rows = run_recovery()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for experiment in ("fixed_core_single_parameter", "joint_core_and_fear_refit"):
        recovered = sorted({
            row["generating_pathway"] for row in rows
            if row["experiment"] == experiment and row["correctly_recovered"]
        })
        print(f"{experiment}: correctly recovered {len(recovered)}/{len(HOLLING_FEAR_PATHWAYS)}: {recovered}")


if __name__ == "__main__":
    main()
