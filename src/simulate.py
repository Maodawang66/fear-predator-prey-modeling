"""数值积分封装（dense_output 插值，避免部分环境下 t_eval 崩溃）。"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.integrate import solve_ivp

from .model import baseline_rhs, fear_instant_rhs, fear_memory_rhs
from .parameters import BaselineParams, FearMemoryParams


def _resample_uniform(sol, n_points: int = 800):
    """将自适应步长解线性插值到均匀网格（避免 Windows 下 t_eval/dense_output 崩溃）。"""
    t_uniform = np.linspace(sol.t[0], sol.t[-1], n_points)
    y_uniform = np.vstack(
        [np.interp(t_uniform, sol.t, sol.y[i]) for i in range(sol.y.shape[0])]
    )
    sol.t = t_uniform
    sol.y = y_uniform
    return sol


def _integrate(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float],
    n_points: int = 800,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    sol = solve_ivp(
        rhs,
        t_span,
        y0,
        method="RK45",
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"积分失败: {sol.message}")
    return _resample_uniform(sol, n_points)


def integrate_baseline(
    p: BaselineParams,
    t_span: tuple[float, float] = (0.0, 80.0),
    n_points: int = 800,
    x0: float = 40.0,
    y0: float = 8.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    y0_vec = np.array([x0, y0], dtype=float)
    return _integrate(
        lambda t, s: baseline_rhs(t, s, p),
        y0_vec,
        t_span,
        n_points,
        rtol,
        atol,
    )


def integrate_rhs(
    rhs: Callable[[float, np.ndarray], np.ndarray],
    y0: np.ndarray,
    t_span: tuple[float, float] = (0.0, 80.0),
    n_points: int = 800,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    """通用积分入口（供文献多机制模型调用）。"""
    return _integrate(rhs, y0, t_span, n_points, rtol, atol)


def integrate_fear_memory(
    p: FearMemoryParams,
    t_span: tuple[float, float] = (0.0, 80.0),
    n_points: int = 800,
    x0: float = 40.0,
    y0: float = 8.0,
    m0: float | None = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    if m0 is None:
        m0 = y0
    y0_vec = np.array([x0, y0, m0], dtype=float)
    return _integrate(
        lambda t, s: fear_memory_rhs(t, s, p),
        y0_vec,
        t_span,
        n_points,
        rtol,
        atol,
    )


def integrate_fear_instant(
    p: FearMemoryParams,
    t_span: tuple[float, float] = (0.0, 80.0),
    n_points: int = 800,
    x0: float = 40.0,
    y0: float = 8.0,
    rtol: float = 1e-8,
    atol: float = 1e-10,
):
    y0_vec = np.array([x0, y0], dtype=float)
    return _integrate(
        lambda t, s: fear_instant_rhs(t, s, p),
        y0_vec,
        t_span,
        n_points,
        rtol,
        atol,
    )


def long_term_mean(sol, burn_in_frac: float = 0.25) -> tuple[float, float]:
    """燃烧期后的时间平均密度（猎物、捕食者）。"""
    n = sol.t.size
    i0 = int(n * burn_in_frac)
    x_mean = float(np.mean(sol.y[0, i0:]))
    y_mean = float(np.mean(sol.y[1, i0:]))
    return x_mean, y_mean


def is_extinct(sol, threshold: float = 1e-3, tail_frac: float = 0.15) -> str:
    """根据末期密度判定灭绝类型。"""
    n = sol.t.size
    i0 = int(n * (1.0 - tail_frac))
    x_tail = sol.y[0, i0:]
    y_tail = sol.y[1, i0:]
    x_min, y_min = float(np.min(x_tail)), float(np.min(y_tail))
    if x_min < threshold and y_min < threshold:
        return "both_extinct"
    if x_min < threshold:
        return "prey_extinct"
    if y_min < threshold:
        return "predator_extinct"
    return "coexist"
