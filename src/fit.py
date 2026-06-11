"""将观测时间序列拟合到 ODE 模型（基线 / 恐惧记忆 / B-D+恐惧）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize
from scipy.stats import chi2

from .model import baseline_rhs, bd_fear_rhs, fear_memory_rhs
from .parameters import BDAFearParams, BaselineParams, FearMemoryParams

DEFAULT_OPTIMIZER_SEEDS = (0, 1, 2)


@dataclass
class _OptResult:
    x: np.ndarray
    success: bool
    status: str
    message: str
    nfev: int
    cost: float
    bound_hit_indices: list[int]


def _bounded_minimize(
    residual: Callable[[np.ndarray], np.ndarray],
    p0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_nfev: int = 250,
    method: str = "auto",
    seed: int = 0,
) -> _OptResult:
    """有界非线性最小二乘；参数较多时用差分进化，避免 Windows LAPACK 崩溃。"""
    nfev = 0

    def objective(log_p: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        r = residual(log_p)
        return float(np.dot(r, r))

    bounds = list(zip(lb.tolist(), ub.tolist()))
    if method not in ("auto", "global", "local"):
        raise ValueError("method must be 'auto', 'global', or 'local'")
    use_global = method == "global" or (method == "auto" and len(p0) >= 5)
    if use_global:
        res = differential_evolution(
            objective,
            bounds,
            seed=seed,
            maxiter=max(20, max_nfev // 10),
            polish=False,
            tol=1e-3,
            atol=1e-3,
        )
        x = res.x
        success = bool(res.success)
        message = str(res.message)
    else:
        res = minimize(
            objective,
            p0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": max_nfev},
        )
        x = res.x
        success = bool(res.success)
        message = str(res.message)

    final_r = residual(x)
    cost = float(np.dot(final_r, final_r))
    message_lower = message.lower()
    reached_limit = any(
        token in message_lower
        for token in ("maximum number", "exceeds limit", "maxfun", "maxiter", "iteration limit")
    )
    usable_objective = np.isfinite(cost) and np.all(np.isfinite(final_r)) and np.max(np.abs(final_r)) < 1e5
    if success and usable_objective:
        status = "success"
    elif reached_limit and usable_objective:
        status = "usable_limit"
    else:
        status = "failed"
    bound_tol = np.maximum(1e-6, 1e-3 * (ub - lb))
    bound_hit_indices = np.flatnonzero(
        (np.abs(x - lb) <= bound_tol) | (np.abs(x - ub) <= bound_tol)
    ).astype(int).tolist()
    return _OptResult(
        x=x,
        success=success,
        status=status,
        message=message,
        nfev=nfev,
        cost=cost,
        bound_hit_indices=bound_hit_indices,
    )


def _multiseed_bounded_minimize(
    residual: Callable[[np.ndarray], np.ndarray],
    p0: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    max_nfev: int,
    method: str = "auto",
    seeds: tuple[int, ...] = DEFAULT_OPTIMIZER_SEEDS,
    param_names: list[str] | None = None,
) -> tuple[_OptResult, dict]:
    """Run local optimization, or reproducible global searches followed by refinement."""
    if method not in ("auto", "global", "local"):
        raise ValueError("method must be 'auto', 'global', or 'local'")
    use_global = method == "global" or (method == "auto" and len(p0) >= 5)
    if not use_global:
        result = _bounded_minimize(
            residual,
            p0,
            lb,
            ub,
            max_nfev=max_nfev,
            method="local",
        )
        return result, {
            "optimizer_seeds": [],
            "optimizer_runs": [],
            "selected_seed": None,
            "local_refinement_used": False,
        }

    if not seeds:
        raise ValueError("at least one optimizer seed is required for global optimization")

    global_runs: list[tuple[int, _OptResult]] = []
    for seed in seeds:
        result = _bounded_minimize(
            residual,
            p0,
            lb,
            ub,
            max_nfev=max_nfev,
            method="global",
            seed=int(seed),
        )
        global_runs.append((int(seed), result))

    usable = [
        item for item in global_runs
        if item[1].status in ("success", "usable_limit") and np.isfinite(item[1].cost)
    ]
    selected_seed, selected_global = min(
        usable or global_runs,
        key=lambda item: item[1].cost if np.isfinite(item[1].cost) else float("inf"),
    )
    refined = _bounded_minimize(
        residual,
        selected_global.x,
        lb,
        ub,
        max_nfev=max_nfev,
        method="local",
    )
    refined_usable = refined.status in ("success", "usable_limit") and np.isfinite(refined.cost)
    global_usable = selected_global.status in ("success", "usable_limit") and np.isfinite(selected_global.cost)
    selected = refined if refined_usable and (not global_usable or refined.cost <= selected_global.cost) else selected_global

    def bound_hit_names(result: _OptResult) -> list[str | int]:
        return (
            [param_names[index] for index in result.bound_hit_indices]
            if param_names is not None
            else result.bound_hit_indices
        )

    optimizer_runs = [
        {
            "seed": seed,
            "status": result.status,
            "success": bool(result.success),
            "objective_value": float(result.cost),
            "nfev": int(result.nfev),
            "termination_reason": str(result.message),
            "parameter_bound_hits": bound_hit_names(result),
        }
        for seed, result in global_runs
    ]
    return selected, {
        "optimizer_seeds": [int(seed) for seed in seeds],
        "optimizer_runs": optimizer_runs,
        "selected_seed": int(selected_seed),
        "local_refinement_used": selected is refined,
        "local_refinement": {
            "status": refined.status,
            "success": bool(refined.success),
            "objective_value": float(refined.cost),
            "nfev": int(refined.nfev),
            "termination_reason": str(refined.message),
            "parameter_bound_hits": bound_hit_names(refined),
        },
    }


@dataclass
class FitResult:
    model: str
    series_name: str
    params: dict[str, float]
    rmse_normalized_prey: float
    rmse_normalized_predator: float
    rmse_normalized_total: float
    rmse_raw_prey: float
    rmse_raw_predator: float
    rmse_raw_total: float
    validation_rmse_normalized_prey: float
    validation_rmse_normalized_predator: float
    validation_rmse_normalized_total: float
    validation_rmse_raw_prey: float
    validation_rmse_raw_predator: float
    validation_rmse_raw_total: float
    aic: float
    aicc: float
    bic: float
    n_parameters: int
    n_train_points: int
    n_validation_points: int
    success: bool
    optimization_status: str
    message: str
    t_obs: np.ndarray
    prey_obs: np.ndarray
    predator_obs: np.ndarray
    prey_pred: np.ndarray
    predator_pred: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def rmse_prey(self) -> float:
        """兼容旧调用；跨模型比较统一返回归一化 RMSE。"""
        return self.rmse_normalized_prey

    @property
    def rmse_predator(self) -> float:
        """兼容旧调用；跨模型比较统一返回归一化 RMSE。"""
        return self.rmse_normalized_predator

    @property
    def rmse_total(self) -> float:
        """兼容旧调用；跨模型比较统一返回归一化 RMSE。"""
        return self.rmse_normalized_total

    @property
    def usable_for_comparison(self) -> bool:
        return self.optimization_status in ("success", "usable_limit")


def _simulate_at_times(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_obs: np.ndarray,
    rtol: float = 1e-7,
    atol: float = 1e-9,
) -> np.ndarray:
    """在观测时刻输出状态（Windows 下避免 t_eval：积分后插值）。"""
    t0, t1 = float(t_obs[0]), float(t_obs[-1])
    sol = solve_ivp(rhs, (t0, t1), y0, method="RK45", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return np.vstack([np.interp(t_obs, sol.t, sol.y[i]) for i in range(sol.y.shape[0])])


def _rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((obs - pred) ** 2)))


def _fit_rmse_metrics(
    prey_obs: np.ndarray,
    predator_obs: np.ndarray,
    prey_pred: np.ndarray,
    predator_pred: np.ndarray,
    prey_scale: float,
    predator_scale: float,
) -> dict[str, float]:
    prey_obs_norm = prey_obs / prey_scale
    predator_obs_norm = predator_obs / predator_scale
    prey_pred_norm = prey_pred / prey_scale
    predator_pred_norm = predator_pred / predator_scale
    return {
        "rmse_normalized_prey": _rmse(prey_obs_norm, prey_pred_norm),
        "rmse_normalized_predator": _rmse(predator_obs_norm, predator_pred_norm),
        "rmse_normalized_total": _rmse(
            np.concatenate([prey_obs_norm, predator_obs_norm]),
            np.concatenate([prey_pred_norm, predator_pred_norm]),
        ),
        "rmse_raw_prey": _rmse(prey_obs, prey_pred),
        "rmse_raw_predator": _rmse(predator_obs, predator_pred),
        "rmse_raw_total": _rmse(
            np.concatenate([prey_obs, predator_obs]),
            np.concatenate([prey_pred, predator_pred]),
        ),
    }


def _train_end_index(n_points: int, validation_fraction: float) -> int:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    if validation_fraction == 0.0:
        return n_points
    n_validation = max(1, int(np.ceil(n_points * validation_fraction)))
    n_train = n_points - n_validation
    if n_train < 4:
        raise ValueError(
            f"holdout leaves only {n_train} training points; at least 4 are required"
        )
    return n_train


def _positive_initial_state(series) -> tuple[float, float]:
    """Return the observed initial state, rejecting absorbing zero-population starts."""
    x0 = float(series.prey[0])
    y0 = float(series.predator[0])
    if not np.isfinite(x0) or not np.isfinite(y0) or x0 <= 0.0 or y0 <= 0.0:
        raise ValueError(
            f"{series.name}: ODE fitting requires positive finite initial prey and predator; "
            "start the series at the first joint positive observation"
        )
    return x0, y0


def _information_criteria(
    normalized_residuals: np.ndarray,
    n_parameters: int,
) -> dict[str, float]:
    """Gaussian AIC/AICc/BIC using the shared normalized residual scale."""
    n = int(normalized_residuals.size)
    rss = max(float(np.dot(normalized_residuals, normalized_residuals)), np.finfo(float).tiny)
    base = n * np.log(rss / n)
    aic = float(base + 2.0 * n_parameters)
    aicc = (
        float(aic + 2.0 * n_parameters * (n_parameters + 1) / (n - n_parameters - 1))
        if n > n_parameters + 1
        else float("inf")
    )
    bic = float(base + n_parameters * np.log(n))
    return {"aic": aic, "aicc": aicc, "bic": bic}


def _fit_evaluation_metrics(
    prey_obs: np.ndarray,
    predator_obs: np.ndarray,
    prey_pred: np.ndarray,
    predator_pred: np.ndarray,
    prey_scale: float,
    predator_scale: float,
    train_end: int,
    n_parameters: int,
) -> dict[str, float | int]:
    train = slice(0, train_end)
    metrics: dict[str, float | int] = _fit_rmse_metrics(
        prey_obs[train],
        predator_obs[train],
        prey_pred[train],
        predator_pred[train],
        prey_scale,
        predator_scale,
    )
    normalized_residuals = np.concatenate([
        (prey_pred[train] - prey_obs[train]) / prey_scale,
        (predator_pred[train] - predator_obs[train]) / predator_scale,
    ])
    metrics.update(_information_criteria(normalized_residuals, n_parameters))

    n_validation = int(prey_obs.size - train_end)
    if n_validation:
        validation = _fit_rmse_metrics(
            prey_obs[train_end:],
            predator_obs[train_end:],
            prey_pred[train_end:],
            predator_pred[train_end:],
            prey_scale,
            predator_scale,
        )
        metrics.update({f"validation_{key}": value for key, value in validation.items()})
    else:
        for key in (
            "rmse_normalized_prey",
            "rmse_normalized_predator",
            "rmse_normalized_total",
            "rmse_raw_prey",
            "rmse_raw_predator",
            "rmse_raw_total",
        ):
            metrics[f"validation_{key}"] = float("nan")
    metrics.update({
        "n_parameters": int(n_parameters),
        "n_train_points": int(train_end),
        "n_validation_points": n_validation,
    })
    return metrics


def _optimization_meta(res: _OptResult, param_names: list[str]) -> dict:
    return {
        "nfev": int(res.nfev),
        "objective_value": float(res.cost),
        "termination_reason": str(res.message),
        "parameter_bound_hits": [param_names[i] for i in res.bound_hit_indices],
    }


def _pack_baseline(
    log_params: np.ndarray,
    fixed: BaselineParams,
    fit_e_mu: bool = False,
) -> BaselineParams:
    r, K, a, theta = np.exp(log_params[:4])
    e, mu = np.exp(log_params[4:6]) if fit_e_mu else (fixed.e, fixed.mu)
    return BaselineParams(
        r=float(r),
        K=float(K),
        a=float(a),
        theta=float(theta),
        e=float(e),
        mu=float(mu),
    )


def _pack_fear_memory(
    log_params: np.ndarray,
    fixed: FearMemoryParams,
    fit_e_mu: bool = True,
) -> FearMemoryParams:
    exp4 = np.exp(log_params[:4])
    if fit_e_mu:
        e, mu, phi = np.exp(log_params[4:7])
    else:
        e, mu = fixed.e, fixed.mu
        phi = float(np.exp(log_params[4]))
    return FearMemoryParams(
        r=float(exp4[0]),
        K=float(exp4[1]),
        a=float(exp4[2]),
        theta=float(exp4[3]),
        e=float(e),
        mu=float(mu),
        phi=float(phi),
        delta=fixed.delta,
    )


def _holling_core_setup(
    prey_max: float,
    fixed: BaselineParams,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """按当前 Holling II 参数尺度生成 r,K,a,theta 的初值与边界。"""
    theta_upper = max(1.0, 10.0 / max(prey_max, 1.0))
    lower = np.array([0.01, prey_max * 0.5, 1e-5, 1e-10])
    upper = np.array([5.0, prey_max * 50.0, 2.0, theta_upper])
    initial = np.array([fixed.r, fixed.K, fixed.a, fixed.theta], dtype=float)
    initial = np.clip(initial, lower * 1.01, upper / 1.01)
    return np.log(initial), np.log(lower), np.log(upper)


def fit_baseline_to_series(
    series,
    fixed: BaselineParams | None = None,
    fit_e_mu: bool = True,
    validation_fraction: float = 0.20,
    optimizer: str = "auto",
    optimizer_seeds: tuple[int, ...] = DEFAULT_OPTIMIZER_SEEDS,
    max_nfev: int = 400,
) -> FitResult:
    """
    非线性最小二乘拟合基线 ODE 到 (t, prey, predator)。
    默认拟合 log(r), log(K), log(a), log(theta), log(e), log(mu)。
    fit_e_mu=False 可复现固定 e、mu 的旧拟合模式。
    """
    if fixed is None:
        fixed = BaselineParams()

    train_end = _train_end_index(series.n_points, validation_fraction)
    prey_max = max(float(np.max(series.prey[:train_end])), 1.0)
    pred_max = max(float(np.max(series.predator[:train_end])), 1.0)
    x0, y0 = _positive_initial_state(series)
    t_obs = series.t
    t_train = t_obs[:train_end]

    def residual(log_p: np.ndarray) -> np.ndarray:
        p = _pack_baseline(log_p, fixed, fit_e_mu=fit_e_mu)
        try:
            y = _simulate_at_times(
                lambda t, s: baseline_rhs(t, s, p),
                np.array([x0, y0]),
                t_train,
            )
        except RuntimeError:
            return np.full(t_train.size * 2, 1e6)
        if np.any(y < 0) or np.any(~np.isfinite(y)):
            return np.full(t_train.size * 2, 1e6)
        # 相对误差，平衡猎物/捕食者量级
        rp = (y[0] - series.prey[:train_end]) / prey_max
        rq = (y[1] - series.predator[:train_end]) / pred_max
        return np.concatenate([rp, rq])

    p0, lb, ub = _holling_core_setup(prey_max, fixed)
    param_names = ["r", "K", "a", "theta"]
    if fit_e_mu:
        e_mu_lb = np.array([0.01, 0.01])
        e_mu_ub = np.array([2.0, 3.0])
        e_mu_p0 = np.clip(np.array([fixed.e, fixed.mu]), e_mu_lb * 1.01, e_mu_ub / 1.01)
        p0 = np.concatenate([p0, np.log(e_mu_p0)])
        lb = np.concatenate([lb, np.log(e_mu_lb)])
        ub = np.concatenate([ub, np.log(e_mu_ub)])
        param_names.extend(["e", "mu"])

    res, optimizer_meta = _multiseed_bounded_minimize(
        residual,
        p0,
        lb,
        ub,
        max_nfev=max_nfev,
        method=optimizer,
        seeds=optimizer_seeds,
        param_names=param_names,
    )
    p_fit = _pack_baseline(res.x, fixed, fit_e_mu=fit_e_mu)
    y_pred = _simulate_at_times(
        lambda t, s: baseline_rhs(t, s, p_fit),
        np.array([x0, y0]),
        t_obs,
    )
    n_parameters = len(param_names)
    fit_metrics = _fit_evaluation_metrics(
        series.prey,
        series.predator,
        y_pred[0],
        y_pred[1],
        prey_max,
        pred_max,
        train_end,
        n_parameters,
    )

    return FitResult(
        model="baseline",
        series_name=series.name,
        params={
            "r": p_fit.r,
            "K": p_fit.K,
            "a": p_fit.a,
            "theta": p_fit.theta,
            "e": p_fit.e,
            "mu": p_fit.mu,
            "x0": x0,
            "y0": y0,
        },
        **fit_metrics,
        success=bool(res.success),
        optimization_status=res.status,
        message=str(res.message),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=y_pred[0],
        predator_pred=y_pred[1],
        meta={
            **_optimization_meta(res, param_names),
            **optimizer_meta,
            "validation_fraction": validation_fraction,
            "validation_mode": "ordered_holdout_continuous_multistep",
            "train_end_time": float(t_obs[train_end - 1]),
            "validation_start_time": float(t_obs[train_end]) if train_end < t_obs.size else None,
        },
    )


def fit_fear_memory_to_series(
    series,
    fixed: FearMemoryParams | None = None,
    baseline_params: BaselineParams | None = None,
    fit_e_mu: bool = True,
    validation_fraction: float = 0.20,
    optimizer: str = "auto",
    optimizer_seeds: tuple[int, ...] = DEFAULT_OPTIMIZER_SEEDS,
    max_nfev: int = 500,
    initial_memory: float | None = None,
) -> FitResult:
    """拟合恐惧记忆模型的动力学参数；delta 与 M(0) 保持固定。"""
    if fixed is None:
        fixed = FearMemoryParams(phi=0.02, delta=1.0)

    if baseline_params is not None:
        fixed = FearMemoryParams(
            r=baseline_params.r,
            K=baseline_params.K,
            a=baseline_params.a,
            theta=baseline_params.theta,
            e=baseline_params.e,
            mu=baseline_params.mu,
            phi=fixed.phi,
            delta=fixed.delta,
        )

    train_end = _train_end_index(series.n_points, validation_fraction)
    prey_max = max(float(np.max(series.prey[:train_end])), 1.0)
    pred_max = max(float(np.max(series.predator[:train_end])), 1.0)
    x0, y0 = _positive_initial_state(series)
    m0 = y0 if initial_memory is None else float(initial_memory)
    if not np.isfinite(m0) or m0 < 0.0:
        raise ValueError("initial_memory must be a finite non-negative value")
    t_obs = series.t
    t_train = t_obs[:train_end]

    def residual(log_p: np.ndarray) -> np.ndarray:
        p = _pack_fear_memory(log_p, fixed, fit_e_mu=fit_e_mu)
        try:
            y = _simulate_at_times(
                lambda t, s: fear_memory_rhs(t, s, p),
                np.array([x0, y0, m0]),
                t_train,
            )
        except RuntimeError:
            return np.full(t_train.size * 2, 1e6)
        if np.any(y[:2] < 0) or np.any(~np.isfinite(y[:2])):
            return np.full(t_train.size * 2, 1e6)
        rp = (y[0] - series.prey[:train_end]) / prey_max
        rq = (y[1] - series.predator[:train_end]) / pred_max
        return np.concatenate([rp, rq])

    p0_core, lb_core, ub_core = _holling_core_setup(prey_max, fixed)
    phi_initial = float(np.clip(fixed.phi, 1.01e-5, 0.2 / 1.01))
    param_names = ["r", "K", "a", "theta"]
    if fit_e_mu:
        e_mu_lb = np.array([0.01, 0.01])
        e_mu_ub = np.array([2.0, 3.0])
        e_mu_p0 = np.clip(np.array([fixed.e, fixed.mu]), e_mu_lb * 1.01, e_mu_ub / 1.01)
        p0 = np.concatenate([p0_core, np.log(e_mu_p0), np.log([phi_initial])])
        lb = np.concatenate([lb_core, np.log(e_mu_lb), np.log([1e-5])])
        ub = np.concatenate([ub_core, np.log(e_mu_ub), np.log([0.2])])
        param_names.extend(["e", "mu"])
    else:
        p0 = np.concatenate([p0_core, np.log([phi_initial])])
        lb = np.concatenate([lb_core, np.log([1e-5])])
        ub = np.concatenate([ub_core, np.log([0.2])])
    param_names.append("phi")

    res, optimizer_meta = _multiseed_bounded_minimize(
        residual,
        p0,
        lb,
        ub,
        max_nfev=max_nfev,
        method=optimizer,
        seeds=optimizer_seeds,
        param_names=param_names,
    )
    p_fit = _pack_fear_memory(res.x, fixed, fit_e_mu=fit_e_mu)
    y_pred = _simulate_at_times(
        lambda t, s: fear_memory_rhs(t, s, p_fit),
        np.array([x0, y0, m0]),
        t_obs,
    )
    fit_metrics = _fit_evaluation_metrics(
        series.prey,
        series.predator,
        y_pred[0],
        y_pred[1],
        prey_max,
        pred_max,
        train_end,
        len(param_names),
    )

    return FitResult(
        model="fear_memory",
        series_name=series.name,
        params={
            "r": p_fit.r,
            "K": p_fit.K,
            "a": p_fit.a,
            "theta": p_fit.theta,
            "e": p_fit.e,
            "mu": p_fit.mu,
            "phi": p_fit.phi,
            "delta": p_fit.delta,
            "x0": x0,
            "y0": y0,
            "m0": m0,
        },
        **fit_metrics,
        success=bool(res.success),
        optimization_status=res.status,
        message=str(res.message),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=y_pred[0],
        predator_pred=y_pred[1],
        meta={
            **_optimization_meta(res, param_names),
            **optimizer_meta,
            "initial_memory_source": "predator_initial" if initial_memory is None else "fixed_input",
            "validation_fraction": validation_fraction,
            "validation_mode": "ordered_holdout_continuous_multistep",
            "train_end_time": float(t_obs[train_end - 1]),
            "validation_start_time": float(t_obs[train_end]) if train_end < t_obs.size else None,
        },
    )


def fit_bda_fear_to_series(
    series,
    fixed: BDAFearParams | None = None,
    fit_k: bool = True,
    validation_fraction: float = 0.20,
    initial: BDAFearParams | None = None,
    optimizer: str = "auto",
    optimizer_seeds: tuple[int, ...] = DEFAULT_OPTIMIZER_SEEDS,
    max_nfev: int = 600,
) -> FitResult:
    """
    拟合 Myint B-D + 恐惧 ODE（无量纲 u,v）。
    数据先按最大值归一化到 (0,1) 量级再拟合 p,q,r,a,d,c,m,(k)。
    """
    if fixed is None:
        fixed = BDAFearParams()

    train_end = _train_end_index(series.n_points, validation_fraction)
    u_scale = max(float(np.max(series.prey[:train_end])), 1.0)
    v_scale = max(float(np.max(series.predator[:train_end])), 1.0)
    prey = series.prey / u_scale
    predator = series.predator / v_scale
    raw_x0, raw_y0 = _positive_initial_state(series)
    u0 = raw_x0 / u_scale
    v0 = raw_y0 / v_scale
    t_obs = series.t
    t_train = t_obs[:train_end]

    n_param = 8 if fit_k else 7
    names = ["r", "d", "a", "p", "q", "c", "m"] + (["k"] if fit_k else [])

    def pack(log_p: np.ndarray) -> BDAFearParams:
        vals = np.exp(log_p[:n_param])
        kw = dict(
            r=float(vals[0]),
            d=float(vals[1]),
            a=float(vals[2]),
            p=float(vals[3]),
            q=float(vals[4]),
            c=float(vals[5]),
            m=float(vals[6]),
            k=fixed.k if not fit_k else float(vals[7]),
        )
        return BDAFearParams(**kw)

    def residual(log_p: np.ndarray) -> np.ndarray:
        p = pack(log_p)
        try:
            y = _simulate_at_times(
                lambda t, s: bd_fear_rhs(t, s, p),
                np.array([u0, v0]),
                t_train,
            )
        except RuntimeError:
            return np.full(t_train.size * 2, 1e6)
        if np.any(y < 0) or np.any(~np.isfinite(y)):
            return np.full(t_train.size * 2, 1e6)
        return np.concatenate([y[0] - prey[:train_end], y[1] - predator[:train_end]])

    if initial is None:
        initial = BDAFearParams()
    p0_list = [
        initial.r,
        initial.d,
        initial.a,
        initial.p,
        initial.q,
        initial.c,
        initial.m,
    ]
    lb_list = [0.2, 0.01, 1e-4, 0.01, 1e-3, 0.1, 0.05]
    ub_list = [10.0, 3.0, 2.0, 5.0, 2.0, 3.0, 2.0]
    if fit_k:
        p0_list.append(initial.k)
        lb_list.append(1e-4)
        ub_list.append(0.5)

    p0 = np.clip(np.array(p0_list), np.array(lb_list) * 1.0001, np.array(ub_list) / 1.0001)
    res, optimizer_meta = _multiseed_bounded_minimize(
        residual,
        np.log(p0),
        np.log(lb_list),
        np.log(ub_list),
        max_nfev=max_nfev,
        method=optimizer,
        seeds=optimizer_seeds,
        param_names=names,
    )
    p_fit = pack(res.x)
    y_pred = _simulate_at_times(
        lambda t, s: bd_fear_rhs(t, s, p_fit),
        np.array([u0, v0]),
        t_obs,
    )

    params = {n: getattr(p_fit, n) for n in names}
    params.update({
        "u0": u0,
        "v0": v0,
        "u_scale": u_scale,
        "v_scale": v_scale,
        "v_observed_median": float(np.median(predator)),
        "v_observed_min": float(np.min(predator)),
        "v_observed_max": float(np.max(predator)),
        "predator_observed_median_raw": float(np.median(series.predator)),
    })
    prey_pred_raw = y_pred[0] * u_scale
    predator_pred_raw = y_pred[1] * v_scale
    fit_metrics = _fit_evaluation_metrics(
        series.prey,
        series.predator,
        prey_pred_raw,
        predator_pred_raw,
        u_scale,
        v_scale,
        train_end,
        len(names),
    )

    return FitResult(
        model="bda_fear",
        series_name=series.name,
        params=params,
        **fit_metrics,
        success=bool(res.success),
        optimization_status=res.status,
        message=str(res.message),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=prey_pred_raw,
        predator_pred=predator_pred_raw,
        meta={
            **_optimization_meta(res, names),
            **optimizer_meta,
            "normalized_fit": True,
            "validation_fraction": validation_fraction,
            "validation_mode": "ordered_holdout_continuous_multistep",
            "train_end_time": float(t_obs[train_end - 1]),
            "validation_start_time": float(t_obs[train_end]) if train_end < t_obs.size else None,
        },
    )


def bda_fear_rmse_at_params(
    series,
    params: BDAFearParams,
    validation_fraction: float = 0.20,
) -> tuple[float, float, float]:
    """给定 B-D 参数，在训练段归一化尺度上计算 RMSE。"""
    train_end = _train_end_index(series.n_points, validation_fraction)
    u_scale = max(float(np.max(series.prey[:train_end])), 1.0)
    v_scale = max(float(np.max(series.predator[:train_end])), 1.0)
    prey = series.prey / u_scale
    predator = series.predator / v_scale
    u0 = float(prey[0])
    v0 = float(predator[0])
    t_obs = series.t[:train_end]

    try:
        y_pred = _simulate_at_times(
            lambda t, s: bd_fear_rhs(t, s, params),
            np.array([u0, v0]),
            t_obs,
        )
    except RuntimeError:
        return float("inf"), float("inf"), float("inf")

    if np.any(y_pred < 0) or np.any(~np.isfinite(y_pred)):
        return float("inf"), float("inf"), float("inf")

    rmse_prey = _rmse(prey[:train_end], y_pred[0])
    rmse_predator = _rmse(predator[:train_end], y_pred[1])
    rmse_total = _rmse(
        np.concatenate([prey[:train_end], predator[:train_end]]),
        np.concatenate([y_pred[0], y_pred[1]]),
    )
    return rmse_prey, rmse_predator, rmse_total


def bda_params_from_dict(d: dict[str, float]) -> BDAFearParams:
    """从 fit_summary / params.json 字典构造 BDAFearParams。"""
    keys = ("r", "d", "a", "k", "p", "q", "c", "m")
    kw = {k: float(d[k]) for k in keys if k in d and d[k] not in ("", None)}
    base = BDAFearParams()
    return BDAFearParams(
        r=kw.get("r", base.r),
        d=kw.get("d", base.d),
        a=kw.get("a", base.a),
        k=kw.get("k", base.k),
        p=kw.get("p", base.p),
        q=kw.get("q", base.q),
        c=kw.get("c", base.c),
        m=kw.get("m", base.m),
    )


def fear_memory_params_from_dict(d: dict[str, float]) -> FearMemoryParams:
    """从 fit_summary / params.json 字典构造 FearMemoryParams。"""
    base = FearMemoryParams()

    def value(name: str) -> float:
        raw = d.get(name)
        return float(raw) if raw not in ("", None) else float(getattr(base, name))

    return FearMemoryParams(
        r=value("r"),
        K=value("K"),
        a=value("a"),
        theta=value("theta"),
        e=value("e"),
        mu=value("mu"),
        phi=value("phi"),
        delta=value("delta"),
    )


def fear_memory_metrics_at_params(
    series,
    params: FearMemoryParams,
    initial_memory: float,
    validation_fraction: float = 0.20,
) -> dict[str, float | int]:
    """Evaluate fixed fear-memory parameters and M(0) on train and holdout segments."""
    m0 = float(initial_memory)
    if not np.isfinite(m0) or m0 < 0.0:
        raise ValueError("initial_memory must be a finite non-negative value")
    x0, y0 = _positive_initial_state(series)
    train_end = _train_end_index(series.n_points, validation_fraction)
    prey_scale = max(float(np.max(series.prey[:train_end])), 1.0)
    predator_scale = max(float(np.max(series.predator[:train_end])), 1.0)
    try:
        prediction = _simulate_at_times(
            lambda t, s: fear_memory_rhs(t, s, params),
            np.array([x0, y0, m0]),
            series.t,
        )
    except RuntimeError:
        return {
            "rmse_normalized_prey": float("inf"),
            "rmse_normalized_predator": float("inf"),
            "rmse_normalized_total": float("inf"),
            "validation_rmse_normalized_prey": float("inf"),
            "validation_rmse_normalized_predator": float("inf"),
            "validation_rmse_normalized_total": float("inf"),
        }
    if np.any(prediction[:2] < 0) or np.any(~np.isfinite(prediction[:2])):
        return {
            "rmse_normalized_prey": float("inf"),
            "rmse_normalized_predator": float("inf"),
            "rmse_normalized_total": float("inf"),
            "validation_rmse_normalized_prey": float("inf"),
            "validation_rmse_normalized_predator": float("inf"),
            "validation_rmse_normalized_total": float("inf"),
        }
    metrics = _fit_evaluation_metrics(
        series.prey,
        series.predator,
        prediction[0],
        prediction[1],
        prey_scale,
        predator_scale,
        train_end,
        n_parameters=0,
    )
    for key in ("aic", "aicc", "bic", "n_parameters"):
        metrics.pop(key)
    return metrics


def _finalize_profile_rows(
    rows: list[dict[str, float | str | bool]],
    n_residuals: int,
    confidence_level: float,
) -> list[dict[str, float | str | bool]]:
    usable_rows = [
        row for row in rows
        if row["optimization_status"] in ("success", "usable_limit")
        and np.isfinite(float(row["profile_rss"]))
    ]
    if not usable_rows:
        return rows

    rss_min = min(float(row["profile_rss"]) for row in usable_rows)
    threshold = float(chi2.ppf(confidence_level, df=1))
    for row in rows:
        if row["optimization_status"] not in ("success", "usable_limit"):
            row["profile_likelihood_ratio"] = float("nan")
            row["inside_confidence_interval"] = False
            continue
        lr = n_residuals * np.log(float(row["profile_rss"]) / rss_min)
        row["profile_likelihood_ratio"] = float(lr)
        row["inside_confidence_interval"] = bool(lr <= threshold)
        row["confidence_level"] = confidence_level
        row["likelihood_ratio_threshold"] = threshold
    return rows


def conditional_fear_memory_m0_scan(
    series,
    base_params: dict[str, float],
    m0_ratio_grid: np.ndarray | None = None,
    validation_fraction: float = 0.20,
) -> list[dict[str, float]]:
    """Scan M(0)/y(0) while holding all fear-memory parameters fixed."""
    if m0_ratio_grid is None:
        m0_ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])
    base = fear_memory_params_from_dict(base_params)
    _, y0 = _positive_initial_state(series)
    rows: list[dict[str, float]] = []
    for ratio in m0_ratio_grid:
        m0 = float(ratio) * y0
        metrics = fear_memory_metrics_at_params(
            series,
            base,
            initial_memory=m0,
            validation_fraction=validation_fraction,
        )
        rows.append({
            "m0": m0,
            "m0_over_y0": float(ratio),
            "rmse_normalized_prey": float(metrics["rmse_normalized_prey"]),
            "rmse_normalized_predator": float(metrics["rmse_normalized_predator"]),
            "rmse_normalized_total": float(metrics["rmse_normalized_total"]),
            "validation_rmse_normalized_prey": float(metrics["validation_rmse_normalized_prey"]),
            "validation_rmse_normalized_predator": float(metrics["validation_rmse_normalized_predator"]),
            "validation_rmse_normalized_total": float(metrics["validation_rmse_normalized_total"]),
        })
    return rows


def profile_fear_memory_m0(
    series,
    base_params: dict[str, float],
    m0_ratio_grid: np.ndarray | None = None,
    validation_fraction: float = 0.20,
    confidence_level: float = 0.95,
    max_nfev: int = 500,
) -> list[dict[str, float | str | bool]]:
    """Profile M(0) by reoptimizing all fear-memory dynamics parameters."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if m0_ratio_grid is None:
        m0_ratio_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0])

    base = fear_memory_params_from_dict(base_params)
    _, y0 = _positive_initial_state(series)
    train_end = _train_end_index(series.n_points, validation_fraction)
    n_residuals = 2 * train_end
    rows: list[dict[str, float | str | bool]] = []
    for ratio in m0_ratio_grid:
        m0 = float(ratio) * y0
        result = fit_fear_memory_to_series(
            series,
            fixed=base,
            initial_memory=m0,
            validation_fraction=validation_fraction,
            optimizer="local",
            max_nfev=max_nfev,
        )
        row: dict[str, float | str | bool] = {
            "m0": m0,
            "m0_over_y0": float(ratio),
            "rmse_normalized_prey": result.rmse_normalized_prey,
            "rmse_normalized_predator": result.rmse_normalized_predator,
            "rmse_normalized_total": result.rmse_normalized_total,
            "validation_rmse_normalized_prey": result.validation_rmse_normalized_prey,
            "validation_rmse_normalized_predator": result.validation_rmse_normalized_predator,
            "validation_rmse_normalized_total": result.validation_rmse_normalized_total,
            "profile_rss": float(n_residuals * result.rmse_normalized_total**2),
            "optimization_status": result.optimization_status,
        }
        row.update({name: float(result.params[name]) for name in (
            "r", "K", "a", "theta", "e", "mu", "phi", "delta"
        )})
        rows.append(row)
    return _finalize_profile_rows(rows, n_residuals, confidence_level)


def profile_fear_memory_delta(
    series,
    base_params: dict[str, float],
    delta_grid: np.ndarray | None = None,
    validation_fraction: float = 0.20,
    confidence_level: float = 0.95,
    max_nfev: int = 500,
) -> list[dict[str, float | str | bool]]:
    """Profile delta and report whether alternative memory timescales improve holdout RMSE."""
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    base = fear_memory_params_from_dict(base_params)
    if delta_grid is None:
        delta_grid = np.unique(np.concatenate([np.logspace(-2, 1, 17), [base.delta]]))

    initial_memory = float(base_params.get("m0", series.predator[0]))
    train_end = _train_end_index(series.n_points, validation_fraction)
    n_residuals = 2 * train_end
    rows: list[dict[str, float | str | bool]] = []
    for delta in delta_grid:
        fixed = FearMemoryParams(
            r=base.r,
            K=base.K,
            a=base.a,
            theta=base.theta,
            e=base.e,
            mu=base.mu,
            phi=base.phi,
            delta=float(delta),
        )
        result = fit_fear_memory_to_series(
            series,
            fixed=fixed,
            initial_memory=initial_memory,
            validation_fraction=validation_fraction,
            optimizer="local",
            max_nfev=max_nfev,
        )
        row: dict[str, float | str | bool] = {
            "delta": float(delta),
            "memory_timescale": float(1.0 / delta),
            "m0": initial_memory,
            "rmse_normalized_prey": result.rmse_normalized_prey,
            "rmse_normalized_predator": result.rmse_normalized_predator,
            "rmse_normalized_total": result.rmse_normalized_total,
            "validation_rmse_normalized_prey": result.validation_rmse_normalized_prey,
            "validation_rmse_normalized_predator": result.validation_rmse_normalized_predator,
            "validation_rmse_normalized_total": result.validation_rmse_normalized_total,
            "profile_rss": float(n_residuals * result.rmse_normalized_total**2),
            "optimization_status": result.optimization_status,
        }
        row.update({name: float(result.params[name]) for name in (
            "r", "K", "a", "theta", "e", "mu", "phi"
        )})
        rows.append(row)
    return _finalize_profile_rows(rows, n_residuals, confidence_level)


def conditional_bda_k_scan(
    series,
    base_params: dict[str, float],
    k_grid: np.ndarray | None = None,
    validation_fraction: float = 0.20,
) -> list[dict[str, float]]:
    """
    固定 r,d,a,p,q,c,m，扫描恐惧参数 k，返回每条 k 的训练 RMSE。

    这是条件敏感性扫描，不是 profile likelihood。
    base_params 通常来自 bda_fear 拟合结果。
    """
    if k_grid is None:
        k_grid = np.unique(np.concatenate([
            np.logspace(-4, -1, 25),
            np.linspace(0.12, 0.5, 13),
        ]))

    p_base = bda_params_from_dict(base_params)
    rows: list[dict[str, float]] = []
    for kv in k_grid:
        p = BDAFearParams(
            r=p_base.r,
            d=p_base.d,
            a=p_base.a,
            k=float(kv),
            p=p_base.p,
            q=p_base.q,
            c=p_base.c,
            m=p_base.m,
        )
        rmse_prey, rmse_pred, rmse_total = bda_fear_rmse_at_params(
            series,
            p,
            validation_fraction=validation_fraction,
        )
        rows.append({
            "k": float(kv),
            "rmse_normalized_prey": rmse_prey,
            "rmse_normalized_predator": rmse_pred,
            "rmse_normalized_total": rmse_total,
        })
    return rows


def profile_bda_k(
    series,
    base_params: dict[str, float],
    k_grid: np.ndarray | None = None,
    validation_fraction: float = 0.20,
    confidence_level: float = 0.95,
    max_nfev: int = 500,
) -> list[dict[str, float | str | bool]]:
    """
    对每个固定 k 重新优化其余 B-D 参数，并计算 profile likelihood。

    使用完整拟合结果作为每个固定 k 的局部优化初值。似然比采用未知方差
    Gaussian 残差近似：n*log(RSS(k)/RSS_min)，置信区间按 chi-square(1) 阈值。
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if k_grid is None:
        k_grid = np.unique(np.concatenate([
            np.logspace(-4, -1, 25),
            np.linspace(0.12, 0.5, 13),
        ]))

    base = bda_params_from_dict(base_params)
    train_end = _train_end_index(series.n_points, validation_fraction)
    n_residuals = 2 * train_end
    rows: list[dict[str, float | str | bool]] = []

    for kv in k_grid:
        fixed = BDAFearParams(
            r=base.r,
            d=base.d,
            a=base.a,
            k=float(kv),
            p=base.p,
            q=base.q,
            c=base.c,
            m=base.m,
        )
        result = fit_bda_fear_to_series(
            series,
            fixed=fixed,
            fit_k=False,
            validation_fraction=validation_fraction,
            initial=base,
            optimizer="local",
            max_nfev=max_nfev,
        )
        rss = n_residuals * result.rmse_normalized_total**2
        row: dict[str, float | str | bool] = {
            "k": float(kv),
            "rmse_normalized_prey": result.rmse_normalized_prey,
            "rmse_normalized_predator": result.rmse_normalized_predator,
            "rmse_normalized_total": result.rmse_normalized_total,
            "profile_rss": float(rss),
            "optimization_status": result.optimization_status,
        }
        for name in ("r", "d", "a", "p", "q", "c", "m"):
            row[name] = float(result.params[name])
        rows.append(row)

    return _finalize_profile_rows(rows, n_residuals, confidence_level)
