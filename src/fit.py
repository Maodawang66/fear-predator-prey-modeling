"""将观测时间序列拟合到 ODE 模型（基线 / 恐惧记忆 / B-D+恐惧）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

from .model import baseline_rhs, bd_fear_rhs, fear_memory_rhs
from .parameters import BDAFearParams, BaselineParams, FearMemoryParams


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
) -> _OptResult:
    """有界非线性最小二乘；参数较多时用差分进化，避免 Windows LAPACK 崩溃。"""
    nfev = 0

    def objective(log_p: np.ndarray) -> float:
        nonlocal nfev
        nfev += 1
        r = residual(log_p)
        return float(np.dot(r, r))

    bounds = list(zip(lb.tolist(), ub.tolist()))
    if len(p0) >= 5:
        res = differential_evolution(
            objective,
            bounds,
            seed=0,
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


def _optimization_meta(res: _OptResult, param_names: list[str]) -> dict:
    return {
        "nfev": int(res.nfev),
        "objective_value": float(res.cost),
        "termination_reason": str(res.message),
        "parameter_bound_hits": [param_names[i] for i in res.bound_hit_indices],
    }


def _pack_baseline(log_params: np.ndarray, fixed: BaselineParams) -> BaselineParams:
    r, K, a, theta = np.exp(log_params[:4])
    return BaselineParams(
        r=float(r),
        K=float(K),
        a=float(a),
        theta=float(theta),
        e=fixed.e,
        mu=fixed.mu,
    )


def _pack_fear_memory(log_params: np.ndarray, fixed: FearMemoryParams) -> FearMemoryParams:
    exp4 = np.exp(log_params[:4])
    phi = float(np.exp(log_params[4]))
    return FearMemoryParams(
        r=float(exp4[0]),
        K=float(exp4[1]),
        a=float(exp4[2]),
        theta=float(exp4[3]),
        e=fixed.e,
        mu=fixed.mu,
        phi=phi,
        delta=fixed.delta,
    )


def fit_baseline_to_series(
    series,
    fixed: BaselineParams | None = None,
    fit_e_mu: bool = False,
) -> FitResult:
    """
    非线性最小二乘拟合基线 ODE 到 (t, prey, predator)。
    默认拟合 log(r), log(K), log(a), log(theta)；e, mu 固定。
    """
    if fixed is None:
        fixed = BaselineParams()

    prey_max = max(float(np.max(series.prey)), 1.0)
    pred_max = max(float(np.max(series.predator)), 1.0)
    x0 = float(series.prey[0])
    y0 = float(series.predator[0])
    t_obs = series.t

    def residual(log_p: np.ndarray) -> np.ndarray:
        p = _pack_baseline(log_p, fixed)
        try:
            y = _simulate_at_times(
                lambda t, s: baseline_rhs(t, s, p),
                np.array([x0, y0]),
                t_obs,
            )
        except RuntimeError:
            return np.full(t_obs.size * 2, 1e6)
        if np.any(y < 0) or np.any(~np.isfinite(y)):
            return np.full(t_obs.size * 2, 1e6)
        # 相对误差，平衡猎物/捕食者量级
        rp = (y[0] - series.prey) / prey_max
        rq = (y[1] - series.predator) / pred_max
        return np.concatenate([rp, rq])

    p0 = np.log([1.0, prey_max * 2.0, 0.04, 20.0])
    lb = np.log([0.05, prey_max * 0.5, 1e-5, 0.1])
    ub = np.log([5.0, prey_max * 50.0, 1.0, prey_max * 10.0])

    res = _bounded_minimize(residual, p0, lb, ub, max_nfev=200)
    p_fit = _pack_baseline(res.x, fixed)
    y_pred = _simulate_at_times(
        lambda t, s: baseline_rhs(t, s, p_fit),
        np.array([x0, y0]),
        t_obs,
    )
    rmse_metrics = _fit_rmse_metrics(
        series.prey, series.predator, y_pred[0], y_pred[1], prey_max, pred_max,
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
        **rmse_metrics,
        success=bool(res.success),
        optimization_status=res.status,
        message=str(res.message),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=y_pred[0],
        predator_pred=y_pred[1],
        meta=_optimization_meta(res, ["r", "K", "a", "theta"]),
    )


def fit_fear_memory_to_series(
    series,
    fixed: FearMemoryParams | None = None,
    baseline_params: BaselineParams | None = None,
) -> FitResult:
    """在基线参数初值上额外拟合 phi（恐惧强度）。"""
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

    prey_max = max(float(np.max(series.prey)), 1.0)
    pred_max = max(float(np.max(series.predator)), 1.0)
    x0 = float(series.prey[0])
    y0 = float(series.predator[0])
    m0 = y0
    t_obs = series.t

    def residual(log_p: np.ndarray) -> np.ndarray:
        p = _pack_fear_memory(log_p, fixed)
        try:
            y = _simulate_at_times(
                lambda t, s: fear_memory_rhs(t, s, p),
                np.array([x0, y0, m0]),
                t_obs,
            )
        except RuntimeError:
            return np.full(t_obs.size * 2, 1e6)
        if np.any(y[:2] < 0) or np.any(~np.isfinite(y[:2])):
            return np.full(t_obs.size * 2, 1e6)
        rp = (y[0] - series.prey) / prey_max
        rq = (y[1] - series.predator) / pred_max
        return np.concatenate([rp, rq])

    p0 = np.log([1.0, prey_max * 2.0, 0.04, min(20.0, prey_max), max(fixed.phi, 1e-4)])
    lb = np.log([0.05, prey_max * 0.5, 1e-5, 0.1, 1e-5])
    ub = np.log([5.0, prey_max * 50.0, 1.0, prey_max * 10.0, 0.2])

    res = _bounded_minimize(residual, p0, lb, ub, max_nfev=250)
    p_fit = _pack_fear_memory(res.x, fixed)
    y_pred = _simulate_at_times(
        lambda t, s: fear_memory_rhs(t, s, p_fit),
        np.array([x0, y0, m0]),
        t_obs,
    )
    rmse_metrics = _fit_rmse_metrics(
        series.prey, series.predator, y_pred[0], y_pred[1], prey_max, pred_max,
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
        **rmse_metrics,
        success=bool(res.success),
        optimization_status=res.status,
        message=str(res.message),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=y_pred[0],
        predator_pred=y_pred[1],
        meta=_optimization_meta(res, ["r", "K", "a", "theta", "phi"]),
    )


def fit_bda_fear_to_series(
    series,
    fixed: BDAFearParams | None = None,
    fit_k: bool = True,
) -> FitResult:
    """
    拟合 Myint B-D + 恐惧 ODE（无量纲 u,v）。
    数据先按最大值归一化到 (0,1) 量级再拟合 p,q,r,a,d,c,m,(k)。
    """
    if fixed is None:
        fixed = BDAFearParams()

    u_scale = max(float(np.max(series.prey)), 1.0)
    v_scale = max(float(np.max(series.predator)), 1.0)
    prey = series.prey / u_scale
    predator = series.predator / v_scale
    u0 = float(prey[0])
    v0 = float(predator[0])
    t_obs = series.t

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
                t_obs,
            )
        except RuntimeError:
            return np.full(t_obs.size * 2, 1e6)
        if np.any(y < 0) or np.any(~np.isfinite(y)):
            return np.full(t_obs.size * 2, 1e6)
        return np.concatenate([y[0] - prey, y[1] - predator])

    p0_list = [2.5, 0.5, 0.1, 1.0, 0.1, 0.8, 0.6]
    lb_list = [0.2, 0.01, 1e-4, 0.01, 1e-3, 0.1, 0.05]
    ub_list = [10.0, 3.0, 2.0, 5.0, 2.0, 3.0, 2.0]
    if fit_k:
        p0_list.append(max(fixed.k, 0.01))
        lb_list.append(1e-4)
        ub_list.append(0.5)

    res = _bounded_minimize(
        residual,
        np.log(np.array(p0_list)),
        np.log(lb_list),
        np.log(ub_list),
        max_nfev=300,
    )
    p_fit = pack(res.x)
    y_pred = _simulate_at_times(
        lambda t, s: bd_fear_rhs(t, s, p_fit),
        np.array([u0, v0]),
        t_obs,
    )

    params = {n: getattr(p_fit, n) for n in names}
    params.update({"u0": u0, "v0": v0, "u_scale": u_scale, "v_scale": v_scale})
    prey_pred_raw = y_pred[0] * u_scale
    predator_pred_raw = y_pred[1] * v_scale
    rmse_metrics = _fit_rmse_metrics(
        series.prey,
        series.predator,
        prey_pred_raw,
        predator_pred_raw,
        u_scale,
        v_scale,
    )

    return FitResult(
        model="bda_fear",
        series_name=series.name,
        params=params,
        **rmse_metrics,
        success=bool(res.success),
        optimization_status=res.status,
        message=str(res.message),
        t_obs=t_obs,
        prey_obs=series.prey,
        predator_obs=series.predator,
        prey_pred=prey_pred_raw,
        predator_pred=predator_pred_raw,
        meta={**_optimization_meta(res, names), "normalized_fit": True},
    )


def bda_fear_rmse_at_params(
    series,
    params: BDAFearParams,
) -> tuple[float, float, float]:
    """给定 B-D+恐惧 参数，在归一化尺度上计算 RMSE（与 fit_bda_fear_to_series 一致）。"""
    u_scale = max(float(np.max(series.prey)), 1.0)
    v_scale = max(float(np.max(series.predator)), 1.0)
    prey = series.prey / u_scale
    predator = series.predator / v_scale
    u0 = float(prey[0])
    v0 = float(predator[0])
    t_obs = series.t

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

    rmse_prey = _rmse(prey, y_pred[0])
    rmse_predator = _rmse(predator, y_pred[1])
    rmse_total = _rmse(
        np.concatenate([prey, predator]),
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


def profile_bda_k(
    series,
    base_params: dict[str, float],
    k_grid: np.ndarray | None = None,
) -> list[dict[str, float]]:
    """
    固定 r,d,a,p,q,c,m，扫描恐惧参数 k，返回每条 k 的 RMSE。
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
        rmse_prey, rmse_pred, rmse_total = bda_fear_rmse_at_params(series, p)
        rows.append({
            "k": float(kv),
            "rmse_normalized_prey": rmse_prey,
            "rmse_normalized_predator": rmse_pred,
            "rmse_normalized_total": rmse_total,
        })
    return rows
