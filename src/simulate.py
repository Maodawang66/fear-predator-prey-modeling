"""数值积分封装（dense_output 插值，避免部分环境下 t_eval 崩溃）。"""

from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class ExtinctionDiagnostics:
    status: str
    thresholds: tuple[float, float]
    below_fractions: tuple[float, float]
    final_below: tuple[bool, bool]
    clear_recovery: tuple[bool, bool]


def extinction_diagnostics(
    sol,
    *,
    scales: tuple[float, float],
    relative_threshold: float = 1e-3,
    tail_frac: float = 0.15,
    min_below_fraction: float = 0.80,
    recovery_window_frac: float = 0.20,
) -> ExtinctionDiagnostics:
    """持续、尺度感知的尾段灭绝诊断。"""
    if any(scale <= 0.0 for scale in scales):
        raise ValueError("scales must be positive")
    if not 0.0 < tail_frac <= 1.0:
        raise ValueError("tail_frac must be in (0, 1]")
    if not 0.0 < min_below_fraction <= 1.0:
        raise ValueError("min_below_fraction must be in (0, 1]")
    if not 0.0 < recovery_window_frac <= 1.0:
        raise ValueError("recovery_window_frac must be in (0, 1]")

    n = sol.t.size
    i0 = int(n * (1.0 - tail_frac))
    tail = sol.y[:2, i0:]
    thresholds = tuple(relative_threshold * scale for scale in scales)
    below_fractions = tuple(
        float(np.mean(tail[index] < thresholds[index])) for index in range(2)
    )
    final_below = tuple(
        bool(tail[index, -1] < thresholds[index]) for index in range(2)
    )
    recovery_points = max(1, int(np.ceil(tail.shape[1] * recovery_window_frac)))
    clear_recovery = tuple(
        bool(np.any(tail[index, -recovery_points:] >= thresholds[index]))
        for index in range(2)
    )
    extinct = tuple(
        below_fractions[index] >= min_below_fraction
        and final_below[index]
        and not clear_recovery[index]
        for index in range(2)
    )
    if extinct == (True, True):
        status = "both_extinct"
    elif extinct[0]:
        status = "prey_extinct"
    elif extinct[1]:
        status = "predator_extinct"
    else:
        status = "coexist"
    return ExtinctionDiagnostics(
        status=status,
        thresholds=thresholds,
        below_fractions=below_fractions,
        final_below=final_below,
        clear_recovery=clear_recovery,
    )


def is_extinct(
    sol,
    threshold: float | None = None,
    tail_frac: float = 0.15,
    *,
    scales: tuple[float, float] | None = None,
    relative_threshold: float = 1e-3,
    min_below_fraction: float = 0.80,
    recovery_window_frac: float = 0.20,
) -> str:
    """返回持续、尺度感知的末期灭绝类型。"""
    if scales is None:
        if threshold is None:
            threshold = 1e-3
        scales = (1.0, 1.0)
        relative_threshold = threshold
    elif threshold is not None:
        raise ValueError("use either threshold or scales/relative_threshold, not both")
    return extinction_diagnostics(
        sol,
        scales=scales,
        relative_threshold=relative_threshold,
        tail_frac=tail_frac,
        min_below_fraction=min_below_fraction,
        recovery_window_frac=recovery_window_frac,
    ).status
