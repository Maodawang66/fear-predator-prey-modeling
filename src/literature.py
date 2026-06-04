"""文献机制对照实验：统一初值与时间跨度下比较多种恐惧建模。"""

from __future__ import annotations

from typing import Callable

import numpy as np
from .model import (
    baseline_rhs,
    bd_fear_rhs,
    fear_foraging_rhs,
    fear_handling_rhs,
    fear_instant_rhs,
    fear_memory_rhs,
    fear_saturating_rhs,
)
from .parameters import (
    MECHANISM_LABELS,
    MechanismId,
    baseline_default,
    bda_fear_default,
    fear_default,
    fear_foraging_default,
    fear_handling_default,
    fear_saturating_default,
)
from .simulate import integrate_baseline, integrate_fear_instant, integrate_fear_memory, integrate_rhs


def make_rhs(mid: MechanismId) -> tuple[Callable, np.ndarray]:
    """返回 (rhs, y0) 对。"""
    x0, y0 = 40.0, 8.0
    if mid == MechanismId.BASELINE:
        p = baseline_default
        return lambda t, s: baseline_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_MEMORY:
        p = fear_default
        return lambda t, s: fear_memory_rhs(t, s, p), np.array([x0, y0, y0])
    if mid == MechanismId.FEAR_INSTANT:
        p = fear_default
        return lambda t, s: fear_instant_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_SATURATING:
        p = fear_saturating_default
        return lambda t, s: fear_saturating_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_FORAGING:
        p = fear_foraging_default
        return lambda t, s: fear_foraging_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_HANDLING:
        p = fear_handling_default
        return lambda t, s: fear_handling_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.BDA_FEAR:
        p = bda_fear_default
        return lambda t, s: bd_fear_rhs(t, s, p), np.array([0.17, 3.9])
    raise ValueError(f"未知机制: {mid}")


def run_mechanism(
    mid: MechanismId,
    t_span: tuple[float, float] = (0.0, 80.0),
    n_points: int = 800,
):
    if mid == MechanismId.BASELINE:
        return integrate_baseline(baseline_default, t_span=t_span, n_points=n_points)
    if mid == MechanismId.FEAR_MEMORY:
        return integrate_fear_memory(fear_default, t_span=t_span, n_points=n_points)
    if mid == MechanismId.FEAR_INSTANT:
        return integrate_fear_instant(fear_default, t_span=t_span, n_points=n_points)
    rhs, y0 = make_rhs(mid)
    return integrate_rhs(rhs, y0, t_span=t_span, n_points=n_points)


def run_all_mechanisms(
    t_span: tuple[float, float] = (0.0, 80.0),
):
    return {mid: run_mechanism(mid, t_span=t_span) for mid in MechanismId}


def mechanism_labels() -> dict[MechanismId, str]:
    return dict(MECHANISM_LABELS)
