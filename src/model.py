"""常微分方程右端项：基线 + 文献中多种恐惧机制。"""

from __future__ import annotations

import numpy as np

from .parameters import (
    BDAFearParams,
    BaselineParams,
    FearForagingParams,
    FearHandlingParams,
    FearMemoryParams,
    FearSaturatingParams,
)


def _clip_state(*vals: float) -> tuple[float, ...]:
    return tuple(max(v, 0.0) for v in vals)


def _holling2(x: float, y: float, a: float, theta: float) -> tuple[float, float]:
    denom = 1.0 + theta * x
    attack = a * x * y / denom
    return attack, attack


def baseline_rhs(t: float, state: np.ndarray, p: BaselineParams) -> np.ndarray:
    """基线：逻辑斯蒂猎物 + Holling II。"""
    x, y = _clip_state(*state[:2])
    attack, assim = _holling2(x, y, p.a, p.theta)
    dx = p.r * x * (1.0 - x / p.K) - attack
    dy = p.e * assim - p.mu * y
    return np.array([dx, dy], dtype=float)


def fear_memory_rhs(t: float, state: np.ndarray, p: FearMemoryParams) -> np.ndarray:
    """
    繁殖抑制 + 指数记忆核（MacDonald 线性链）。
    dM/dt = y - delta*M；恐惧项 -r*phi*M*x
  对应 Wang et al. (2019) 类“风险降低有效增长”框架。
    """
    x, y, m = _clip_state(*state[:3])
    attack, assim = _holling2(x, y, p.a, p.theta)
    fear_term = p.r * p.phi * m * x
    dx = p.r * x * (1.0 - x / p.K) - attack - fear_term
    dy = p.e * assim - p.mu * y
    dm = y - p.delta * m
    return np.array([dx, dy, dm], dtype=float)


def fear_instant_rhs(t: float, state: np.ndarray, p: FearMemoryParams) -> np.ndarray:
    """繁殖抑制无记忆：M 等价于当前捕食者密度 y。"""
    x, y = _clip_state(*state[:2])
    attack, assim = _holling2(x, y, p.a, p.theta)
    fear_term = p.r * p.phi * y * x
    dx = p.r * x * (1.0 - x / p.K) - attack - fear_term
    dy = p.e * assim - p.mu * y
    return np.array([dx, dy], dtype=float)


def fear_saturating_rhs(t: float, state: np.ndarray, p: FearSaturatingParams) -> np.ndarray:
    """
    饱和型恐惧繁殖抑制（Zanette / Abrams 可饱和风险响应）。
    有效增长：r * (1 - phi*y/(y+h)) * x * (1 - x/K)
    """
    x, y = _clip_state(*state[:2])
    attack, assim = _holling2(x, y, p.a, p.theta)
    fear_factor = 1.0 - p.phi * y / (y + p.h)
    fear_factor = max(fear_factor, 0.0)
    dx = p.r * fear_factor * x * (1.0 - x / p.K) - attack
    dy = p.e * assim - p.mu * y
    return np.array([dx, dy], dtype=float)


def fear_foraging_rhs(t: float, state: np.ndarray, p: FearForagingParams) -> np.ndarray:
    """
    觅食抑制型（Lima 1998; Preisser et al. 2007 行为机制）。
    警觉降低 encounter：a_eff = a / (1 + psi*y)
    """
    x, y = _clip_state(*state[:2])
    a_eff = p.a / (1.0 + p.psi * y)
    attack, assim = _holling2(x, y, a_eff, p.theta)
    dx = p.r * x * (1.0 - x / p.K) - attack
    dy = p.e * assim - p.mu * y
    return np.array([dx, dy], dtype=float)


def bd_fear_rhs(t: float, state: np.ndarray, p: BDAFearParams) -> np.ndarray:
    """
    Myint et al. (2025) 局部动力学（式 2.1/2.2 去掉扩散）：
    du/dt = ru/(1+kv) - du - au^2 - puv/(1+qu+v)
    dv/dt = v(-m + cpu/(1+qu+v))
    恐惧：f(k,v)=1/(1+kv) 作用于繁殖项（Wang et al. 2016）。
    """
    u, v = _clip_state(*state[:2])
    denom = 1.0 + p.q * u + v
    du = p.r * u / (1.0 + p.k * v) - p.d * u - p.a * u * u - p.p * u * v / denom
    dv = v * (-p.m + p.c * p.p * u / denom)
    return np.array([du, dv], dtype=float)


def fear_handling_rhs(t: float, state: np.ndarray, p: FearHandlingParams) -> np.ndarray:
    """
    警觉延长处理时间（Sih NCE 框架中“降低摄食效率”）。
    theta_eff = theta * (1 + psi*y)
    """
    x, y = _clip_state(*state[:2])
    theta_eff = p.theta * (1.0 + p.psi * y)
    attack, assim = _holling2(x, y, p.a, theta_eff)
    dx = p.r * x * (1.0 - x / p.K) - attack
    dy = p.e * assim - p.mu * y
    return np.array([dx, dy], dtype=float)
