"""文献机制对照实验：统一初值与时间跨度下比较多种恐惧建模。"""

from __future__ import annotations

from typing import Callable, Mapping

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
    bda_no_fear_default,
    fear_default,
    fear_foraging_default,
    fear_handling_default,
    fear_saturating_default,
)
from .simulate import integrate_rhs


def make_rhs(mid: MechanismId, params=None) -> tuple[Callable, np.ndarray]:
    """返回 (rhs, y0) 对。"""
    x0, y0 = 40.0, 8.0
    if mid == MechanismId.BASELINE:
        p = params or baseline_default
        return lambda t, s: baseline_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_MEMORY:
        p = params or fear_default
        return lambda t, s: fear_memory_rhs(t, s, p), np.array([x0, y0, y0])
    if mid == MechanismId.FEAR_INSTANT:
        p = params or fear_default
        return lambda t, s: fear_instant_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_SATURATING:
        p = params or fear_saturating_default
        return lambda t, s: fear_saturating_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_FORAGING:
        p = params or fear_foraging_default
        return lambda t, s: fear_foraging_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.FEAR_HANDLING:
        p = params or fear_handling_default
        return lambda t, s: fear_handling_rhs(t, s, p), np.array([x0, y0])
    if mid == MechanismId.BDA_BASELINE:
        p = params or bda_no_fear_default
        return lambda t, s: bd_fear_rhs(t, s, p), np.array([0.17, 3.9])
    if mid == MechanismId.BDA_FEAR:
        p = params or bda_fear_default
        return lambda t, s: bd_fear_rhs(t, s, p), np.array([0.17, 3.9])
    raise ValueError(f"未知机制: {mid}")


def run_mechanism(
    mid: MechanismId,
    t_span: tuple[float, float] = (0.0, 80.0),
    n_points: int = 800,
    params=None,
):
    rhs, y0 = make_rhs(mid, params=params)
    return integrate_rhs(rhs, y0, t_span=t_span, n_points=n_points)


def run_all_mechanisms(
    t_span: tuple[float, float] = (0.0, 80.0),
    params_by_mechanism: Mapping[MechanismId, object] | None = None,
):
    params_by_mechanism = params_by_mechanism or {}
    return {
        mid: run_mechanism(mid, t_span=t_span, params=params_by_mechanism.get(mid))
        for mid in MechanismId
    }


def mechanism_labels() -> dict[MechanismId, str]:
    return dict(MECHANISM_LABELS)
