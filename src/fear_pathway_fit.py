"""Equally parameterized Holling-II fear-pathway fitting."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2

from .fit import (
    FitResult,
    _OptResult,
    _fit_evaluation_metrics,
    _holling_core_setup,
    _optimization_meta,
    _pack_baseline,
    _simulate_at_times,
    _train_end_index,
)
from .model import (
    baseline_rhs,
    fear_foraging_rhs,
    fear_handling_rhs,
    fear_instant_rhs,
    fear_memory_rhs,
    fear_saturating_rhs,
)
from .parameters import (
    BaselineParams,
    FearForagingParams,
    FearHandlingParams,
    FearMemoryParams,
    FearSaturatingParams,
)

DEFAULT_OPTIMIZER_SEEDS = (0, 1, 2)
HOLLING_FEAR_PATHWAYS = (
    "fear_instant",
    "fear_memory",
    "fear_saturating",
    "fear_foraging",
    "fear_handling",
)


def _bounded_minimize_seeded(
    residual: Callable[[np.ndarray], np.ndarray],
    p0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_nfev: int,
    method: str,
    seed: int = 0,
) -> _OptResult:
    nfev = 0

    def objective(log_p: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        values = residual(log_p)
        return float(np.dot(values, values))

    bounds = list(zip(lb.tolist(), ub.tolist()))
    if method == "global":
        result = differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=max(20, max_nfev // 10),
            polish=False,
            tol=1e-3,
            atol=1e-3,
        )
    elif method == "local":
        result = minimize(
            objective,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": max_nfev},
        )
    else:
        raise ValueError("method must be 'global' or 'local'")

    final_residual = residual(result.x)
    cost = float(np.dot(final_residual, final_residual))
    message = str(result.message)
    reached_limit = any(
        token in message.lower()
        for token in ("maximum number", "exceeds limit", "maxfun", "maxiter", "iteration limit")
    )
    usable = np.isfinite(cost) and np.all(np.isfinite(final_residual)) and np.max(np.abs(final_residual)) < 1e5
    status = "success" if result.success and usable else "usable_limit" if reached_limit and usable else "failed"
    bound_tol = np.maximum(1e-6, 1e-3 * (ub - lb))
    bound_hits = np.flatnonzero(
        (np.abs(result.x - lb) <= bound_tol) | (np.abs(result.x - ub) <= bound_tol)
    ).astype(int).tolist()
    return _OptResult(
        x=result.x,
        success=bool(result.success),
        status=status,
        message=message,
        nfev=nfev,
        cost=cost,
        bound_hit_indices=bound_hits,
    )


def _multiseed_bounded_minimize(
    residual: Callable[[np.ndarray], np.ndarray],
    p0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_nfev: int,
    method: str,
    seeds: tuple[int, ...],
    param_names: list[str],
) -> tuple[_OptResult, dict]:
    if method not in ("auto", "global", "local"):
        raise ValueError("method must be 'auto', 'global', or 'local'")
    use_global = method == "global" or (method == "auto" and len(p0) >= 5)
    if not use_global:
        result = _bounded_minimize_seeded(residual, p0, lb, ub, max_nfev, "local")
        return result, {
            "optimizer_seeds": [],
            "optimizer_runs": [],
            "selected_seed": None,
            "local_refinement_used": False,
        }
    if not seeds:
        raise ValueError("at least one optimizer seed is required")
    runs = [
        (int(seed), _bounded_minimize_seeded(residual, p0, lb, ub, max_nfev, "global", int(seed)))
        for seed in seeds
    ]
    usable = [item for item in runs if item[1].status in ("success", "usable_limit")]
    selected_seed, selected_global = min(
        usable or runs,
        key=lambda item: item[1].cost if np.isfinite(item[1].cost) else float("inf"),
    )
    refined = _bounded_minimize_seeded(residual, selected_global.x, lb, ub, max_nfev, "local")
    selected = (
        refined
        if refined.status in ("success", "usable_limit") and refined.cost <= selected_global.cost
        else selected_global
    )

    def names(result: _OptResult) -> list[str]:
        return [param_names[index] for index in result.bound_hit_indices]

    return selected, {
        "optimizer_seeds": [seed for seed, _ in runs],
        "optimizer_runs": [
            {
                "seed": seed,
                "status": result.status,
                "success": result.success,
                "objective_value": result.cost,
                "nfev": result.nfev,
                "termination_reason": result.message,
                "parameter_bound_hits": names(result),
            }
            for seed, result in runs
        ],
        "selected_seed": selected_seed,
        "local_refinement_used": selected is refined,
    }


def _finalize_profile_rows(
    rows: list[dict[str, float | str | bool]],
    n_residuals: int,
    confidence_level: float,
) -> list[dict[str, float | str | bool]]:
    usable = [
        row for row in rows
        if row["optimization_status"] in ("success", "usable_limit")
        and np.isfinite(float(row["profile_rss"]))
    ]
    if not usable:
        return rows
    rss_min = min(float(row["profile_rss"]) for row in usable)
    threshold = float(chi2.ppf(confidence_level, df=1))
    for row in rows:
        if row["optimization_status"] not in ("success", "usable_limit"):
            row["profile_likelihood_ratio"] = float("nan")
            row["inside_confidence_interval"] = False
            continue
        likelihood_ratio = n_residuals * np.log(float(row["profile_rss"]) / rss_min)
        row["profile_likelihood_ratio"] = float(likelihood_ratio)
        row["inside_confidence_interval"] = bool(likelihood_ratio <= threshold)
        row["confidence_level"] = confidence_level
        row["likelihood_ratio_threshold"] = threshold
    return rows


def fit_holling_baseline_to_series(
    series,
    fixed: BaselineParams | None = None,
    validation_fraction: float = 0.20,
    optimizer: str = "auto",
    optimizer_seeds: tuple[int, ...] = DEFAULT_OPTIMIZER_SEEDS,
    max_nfev: int = 500,
) -> FitResult:
    """Fit the six shared Holling-II core parameters used by every fear candidate."""
    if fixed is None:
        fixed = BaselineParams()
    train_end = _train_end_index(series.n_points, validation_fraction)
    prey_scale = max(float(np.max(series.prey[:train_end])), 1.0)
    predator_scale = max(float(np.max(series.predator[:train_end])), 1.0)
    initial = np.array([float(series.prey[0]), float(series.predator[0])])

    def residual(log_p: np.ndarray) -> np.ndarray:
        params = _pack_baseline(log_p, fixed, fit_e_mu=True)
        try:
            prediction = _simulate_at_times(
                lambda t, state: baseline_rhs(t, state, params),
                initial,
                series.t[:train_end],
            )
        except RuntimeError:
            return np.full(train_end * 2, 1e6)
        if np.any(prediction < 0) or np.any(~np.isfinite(prediction)):
            return np.full(train_end * 2, 1e6)
        return np.concatenate([
            (prediction[0] - series.prey[:train_end]) / prey_scale,
            (prediction[1] - series.predator[:train_end]) / predator_scale,
        ])

    p0_core, lb_core, ub_core = _holling_core_setup(prey_scale, fixed)
    e_mu_lb = np.array([0.01, 0.01])
    e_mu_ub = np.array([2.0, 3.0])
    e_mu_p0 = np.clip(np.array([fixed.e, fixed.mu]), e_mu_lb * 1.01, e_mu_ub / 1.01)
    p0 = np.concatenate([p0_core, np.log(e_mu_p0)])
    lb = np.concatenate([lb_core, np.log(e_mu_lb)])
    ub = np.concatenate([ub_core, np.log(e_mu_ub)])
    names = ["r", "K", "a", "theta", "e", "mu"]
    result, optimizer_meta = _multiseed_bounded_minimize(
        residual, p0, lb, ub, max_nfev, optimizer, optimizer_seeds, names
    )
    params = _pack_baseline(result.x, fixed, fit_e_mu=True)
    prediction = _simulate_at_times(
        lambda t, state: baseline_rhs(t, state, params),
        initial,
        series.t,
    )
    metrics = _fit_evaluation_metrics(
        series.prey,
        series.predator,
        prediction[0],
        prediction[1],
        prey_scale,
        predator_scale,
        train_end,
        len(names),
    )
    return FitResult(
        model="baseline",
        series_name=series.name,
        params={
            "r": params.r, "K": params.K, "a": params.a, "theta": params.theta,
            "e": params.e, "mu": params.mu, "x0": initial[0], "y0": initial[1],
        },
        **metrics,
        success=result.success,
        optimization_status=result.status,
        message=result.message,
        t_obs=series.t,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=prediction[0],
        predator_pred=prediction[1],
        meta={
            **_optimization_meta(result, names),
            **optimizer_meta,
            "validation_fraction": validation_fraction,
            "validation_mode": "ordered_holdout_continuous_multistep",
            "train_end_time": float(series.t[train_end - 1]),
            "validation_start_time": float(series.t[train_end]) if train_end < series.n_points else None,
        },
    )


def fit_holling_fear_pathway_to_series(
    series,
    pathway: str,
    fixed: BaselineParams | None = None,
    baseline_result: FitResult | None = None,
    baseline_params: BaselineParams | None = None,
    baseline_parameter_bound_hits: tuple[str, ...] = (),
    baseline_candidate_status: str = "success",
    baseline_candidate_message: str = "nested baseline candidate",
    validation_fraction: float = 0.20,
    optimizer: str = "auto",
    optimizer_seeds: tuple[int, ...] = DEFAULT_OPTIMIZER_SEEDS,
    max_nfev: int = 500,
    fear_strength: float | None = None,
    fear_strength_upper: float | None = None,
    memory_delta: float = 1.0,
    initial_memory: float | None = None,
    saturating_half_response: float | None = None,
) -> FitResult:
    """Fit one pathway with six shared core parameters and one fear parameter."""
    if pathway not in HOLLING_FEAR_PATHWAYS:
        raise ValueError(f"unknown Holling fear pathway: {pathway}")
    if baseline_result is not None and baseline_params is not None:
        raise ValueError("provide baseline_result or baseline_params, not both")
    if fixed is None:
        fixed = BaselineParams()
    if memory_delta <= 0.0 or not np.isfinite(memory_delta):
        raise ValueError("memory_delta must be finite and positive")

    train_end = _train_end_index(series.n_points, validation_fraction)
    prey_scale = max(float(np.max(series.prey[:train_end])), 1.0)
    predator_scale = max(float(np.max(series.predator[:train_end])), 1.0)
    x0 = float(series.prey[0])
    y0 = float(series.predator[0])
    m0 = y0 if initial_memory is None else float(initial_memory)
    if not np.isfinite(m0) or m0 < 0.0:
        raise ValueError("initial_memory must be a finite non-negative value")
    h = (
        max(float(np.median(series.predator[:train_end])), np.finfo(float).eps)
        if saturating_half_response is None
        else float(saturating_half_response)
    )
    if h <= 0.0 or not np.isfinite(h):
        raise ValueError("saturating_half_response must be finite and positive")

    strength_name = "phi" if pathway in ("fear_instant", "fear_memory", "fear_saturating") else "psi"
    strength_lower = 1e-8
    default_upper = 0.99 if pathway == "fear_saturating" else max(0.2, 10.0 / predator_scale)
    strength_upper = default_upper if fear_strength_upper is None else float(fear_strength_upper)
    if not np.isfinite(strength_upper) or strength_upper <= strength_lower:
        raise ValueError(f"fear_strength_upper must be finite and greater than {strength_lower}")
    default_strength = 0.2 if pathway == "fear_saturating" else min(0.02, strength_upper / 2.0)
    fit_strength = fear_strength is None
    fixed_strength = default_strength if fit_strength else float(fear_strength)
    if fixed_strength < 0.0 or not np.isfinite(fixed_strength):
        raise ValueError("fear_strength must be finite and non-negative")

    def pack(log_p: np.ndarray):
        core = _pack_baseline(log_p[:6], fixed, fit_e_mu=True)
        strength = float(np.exp(log_p[6])) if fit_strength else fixed_strength
        common = dict(r=core.r, K=core.K, a=core.a, theta=core.theta, e=core.e, mu=core.mu)
        if pathway in ("fear_instant", "fear_memory"):
            params = FearMemoryParams(**common, phi=strength, delta=memory_delta)
        elif pathway == "fear_saturating":
            params = FearSaturatingParams(**common, phi=strength, h=h)
        elif pathway == "fear_foraging":
            params = FearForagingParams(**common, psi=strength)
        else:
            params = FearHandlingParams(**common, psi=strength)
        return params, strength

    rhs_by_pathway: dict[str, Callable] = {
        "fear_instant": fear_instant_rhs,
        "fear_memory": fear_memory_rhs,
        "fear_saturating": fear_saturating_rhs,
        "fear_foraging": fear_foraging_rhs,
        "fear_handling": fear_handling_rhs,
    }
    rhs = rhs_by_pathway[pathway]
    y0_fit = np.array([x0, y0, m0]) if pathway == "fear_memory" else np.array([x0, y0])
    t_obs = series.t
    t_train = t_obs[:train_end]

    def residual(log_p: np.ndarray) -> np.ndarray:
        params, _ = pack(log_p)
        try:
            prediction = _simulate_at_times(
                lambda t, state: rhs(t, state, params),
                y0_fit,
                t_train,
            )
        except RuntimeError:
            return np.full(t_train.size * 2, 1e6)
        if np.any(prediction[:2] < 0) or np.any(~np.isfinite(prediction[:2])):
            return np.full(t_train.size * 2, 1e6)
        return np.concatenate([
            (prediction[0] - series.prey[:train_end]) / prey_scale,
            (prediction[1] - series.predator[:train_end]) / predator_scale,
        ])

    p0_core, lb_core, ub_core = _holling_core_setup(prey_scale, fixed)
    e_mu_lb = np.array([0.01, 0.01])
    e_mu_ub = np.array([2.0, 3.0])
    e_mu_p0 = np.clip(np.array([fixed.e, fixed.mu]), e_mu_lb * 1.01, e_mu_ub / 1.01)
    p0 = np.concatenate([p0_core, np.log(e_mu_p0)])
    lb = np.concatenate([lb_core, np.log(e_mu_lb)])
    ub = np.concatenate([ub_core, np.log(e_mu_ub)])
    param_names = ["r", "K", "a", "theta", "e", "mu"]
    if fit_strength:
        initial_strength = np.clip(default_strength, strength_lower * 1.01, strength_upper / 1.01)
        p0 = np.concatenate([p0, np.log([initial_strength])])
        lb = np.concatenate([lb, np.log([strength_lower])])
        ub = np.concatenate([ub, np.log([strength_upper])])
        param_names.append(strength_name)

    result, optimizer_meta = _multiseed_bounded_minimize(
        residual,
        p0,
        lb,
        ub,
        max_nfev=max_nfev,
        method=optimizer,
        seeds=optimizer_seeds,
        param_names=param_names,
    )
    optimized_params, optimized_strength = pack(result.x)
    try:
        optimized_prediction = _simulate_at_times(
            lambda t, state: rhs(t, state, optimized_params), y0_fit, t_obs
        )
        optimized_prediction_usable = bool(
            np.all(np.isfinite(optimized_prediction[:2]))
            and np.all(optimized_prediction[:2] >= 0.0)
        )
    except RuntimeError:
        optimized_prediction = None
        optimized_prediction_usable = False

    nested_selected = False
    nested_cost = float("nan")
    params = optimized_params
    strength = optimized_strength
    prediction = optimized_prediction
    final_cost = float(result.cost)
    if fit_strength:
        if baseline_result is None and baseline_params is None:
            baseline_result = fit_holling_baseline_to_series(
                series,
                fixed=fixed,
                validation_fraction=validation_fraction,
                optimizer=optimizer,
                optimizer_seeds=optimizer_seeds,
                max_nfev=max_nfev,
            )
        if baseline_result is not None:
            baseline_parameter_bound_hits = tuple(
                baseline_result.meta.get("parameter_bound_hits", [])
            )
            baseline_candidate_status = baseline_result.optimization_status
            baseline_candidate_message = baseline_result.message
        nested_core = baseline_params or BaselineParams(**{
            name: float(baseline_result.params[name])
            for name in ("r", "K", "a", "theta", "e", "mu")
        })
        common = {
            "r": nested_core.r,
            "K": nested_core.K,
            "a": nested_core.a,
            "theta": nested_core.theta,
            "e": nested_core.e,
            "mu": nested_core.mu,
        }
        if pathway in ("fear_instant", "fear_memory"):
            nested_params = FearMemoryParams(**common, phi=0.0, delta=memory_delta)
        elif pathway == "fear_saturating":
            nested_params = FearSaturatingParams(**common, phi=0.0, h=h)
        elif pathway == "fear_foraging":
            nested_params = FearForagingParams(**common, psi=0.0)
        else:
            nested_params = FearHandlingParams(**common, psi=0.0)
        nested_prediction = _simulate_at_times(
            lambda t, state: rhs(t, state, nested_params), y0_fit, t_obs
        )
        nested_residual = np.concatenate([
            (nested_prediction[0, :train_end] - series.prey[:train_end]) / prey_scale,
            (nested_prediction[1, :train_end] - series.predator[:train_end]) / predator_scale,
        ])
        nested_cost = float(np.dot(nested_residual, nested_residual))
        nested_selected = (
            result.status not in ("success", "usable_limit")
            or not np.isfinite(result.cost)
            or not optimized_prediction_usable
            or nested_cost < result.cost
        )
        if nested_selected:
            params = nested_params
            strength = 0.0
            prediction = nested_prediction
            final_cost = nested_cost
        tolerance = max(1e-10, 1e-8 * max(1.0, nested_cost))
        if final_cost > nested_cost + tolerance:
            raise RuntimeError(f"{pathway} nested-baseline invariant violated")

    metrics = _fit_evaluation_metrics(
        series.prey,
        series.predator,
        prediction[0],
        prediction[1],
        prey_scale,
        predator_scale,
        train_end,
        len(param_names),
    )
    fitted_params = {
        "r": params.r, "K": params.K, "a": params.a, "theta": params.theta,
        "e": params.e, "mu": params.mu, strength_name: strength, "x0": x0, "y0": y0,
    }
    if pathway == "fear_memory":
        fitted_params.update({"delta": memory_delta, "m0": m0})
    if pathway == "fear_saturating":
        fitted_params["h"] = h

    final_optimization_meta = _optimization_meta(result, param_names)
    final_optimization_meta.update({
        "objective_value": final_cost,
        "parameter_bound_hits": (
            list(baseline_parameter_bound_hits)
            if nested_selected else final_optimization_meta["parameter_bound_hits"]
        ),
    })
    return FitResult(
        model=pathway,
        series_name=series.name,
        params=fitted_params,
        **metrics,
        success=(baseline_candidate_status == "success") if nested_selected else bool(result.success),
        optimization_status=baseline_candidate_status if nested_selected else result.status,
        message=(
            f"nested baseline candidate selected: {baseline_candidate_message}"
            if nested_selected else str(result.message)
        ),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=prediction[0],
        predator_pred=prediction[1],
        meta={
            **final_optimization_meta,
            **optimizer_meta,
            "optimized_fear_objective": float(result.cost),
            "nested_baseline_candidate_objective": nested_cost,
            "nested_baseline_candidate_selected": nested_selected,
            "fear_parameter": strength_name,
            "fear_parameter_fitted": fit_strength,
            "fear_parameter_bounds": [strength_lower, strength_upper],
            "memory_delta_fixed": memory_delta if pathway == "fear_memory" else None,
            "initial_memory_source": (
                "predator_initial" if pathway == "fear_memory" and initial_memory is None
                else "fixed_input" if pathway == "fear_memory"
                else None
            ),
            "saturating_half_response_fixed": h if pathway == "fear_saturating" else None,
            "validation_fraction": validation_fraction,
            "validation_mode": "ordered_holdout_continuous_multistep",
            "train_end_time": float(t_obs[train_end - 1]),
            "validation_start_time": float(t_obs[train_end]) if train_end < t_obs.size else None,
        },
    )


def profile_holling_fear_strength(
    series,
    pathway: str,
    base_result: FitResult,
    strength_grid: np.ndarray | None = None,
    validation_fraction: float = 0.20,
    confidence_level: float = 0.95,
    max_nfev: int = 250,
) -> list[dict[str, float | str | bool]]:
    """Profile the single fear parameter while reoptimizing shared core parameters."""
    if pathway not in HOLLING_FEAR_PATHWAYS:
        raise ValueError(f"unknown Holling fear pathway: {pathway}")
    strength_name = "phi" if pathway in ("fear_instant", "fear_memory", "fear_saturating") else "psi"
    fitted_strength = float(base_result.params[strength_name])
    upper = float(base_result.meta["fear_parameter_bounds"][1])
    if strength_grid is None:
        positive = max(fitted_strength, 1e-8)
        strength_grid = np.unique(np.clip(
            np.array([0.0, positive / 10.0, positive / 3.0, positive, positive * 3.0, positive * 10.0]),
            0.0,
            upper,
        ))
    fixed = BaselineParams(**{name: float(base_result.params[name]) for name in (
        "r", "K", "a", "theta", "e", "mu"
    )})
    train_end = _train_end_index(series.n_points, validation_fraction)
    rows: list[dict[str, float | str | bool]] = []
    for strength in strength_grid:
        result = fit_holling_fear_pathway_to_series(
            series,
            pathway,
            fixed=fixed,
            validation_fraction=validation_fraction,
            optimizer="local",
            max_nfev=max_nfev,
            fear_strength=float(strength),
            memory_delta=float(base_result.params.get("delta", 1.0)),
            initial_memory=base_result.params.get("m0"),
            saturating_half_response=base_result.params.get("h"),
        )
        rows.append({
            "pathway": pathway,
            "fear_parameter": strength_name,
            "fear_strength": float(strength),
            "profile_rss": float(2 * train_end * result.rmse_normalized_total**2),
            "rmse_normalized_total": result.rmse_normalized_total,
            "validation_rmse_normalized_total": result.validation_rmse_normalized_total,
            "optimization_status": result.optimization_status,
        })
    return _finalize_profile_rows(rows, 2 * train_end, confidence_level)
