"""参数扫描、敏感性、文献机制对比分析。"""

from __future__ import annotations

import numpy as np
from .literature import run_mechanism
from .parameters import BDAFearParams, FearMemoryParams, MechanismId, baseline_default, bda_fear_default, fear_default
from .simulate import integrate_baseline, integrate_fear_memory, is_extinct, long_term_mean


def scan_phi(
    phi_values: np.ndarray | None = None,
    base: FearMemoryParams | None = None,
    t_end: float = 120.0,
) -> dict[str, np.ndarray]:
    if phi_values is None:
        phi_values = np.linspace(0.0, 0.06, 31)
    if base is None:
        base = fear_default

    x_mean = np.zeros_like(phi_values)
    y_mean = np.zeros_like(phi_values)
    status = []

    for i, phi in enumerate(phi_values):
        p = FearMemoryParams(
            r=base.r,
            K=base.K,
            a=base.a,
            theta=base.theta,
            e=base.e,
            mu=base.mu,
            phi=float(phi),
            delta=base.delta,
        )
        sol = integrate_fear_memory(p, t_span=(0.0, t_end))
        xm, ym = long_term_mean(sol, burn_in_frac=0.3)
        x_mean[i], y_mean[i] = xm, ym
        status.append(is_extinct(sol))

    return {
        "phi": phi_values,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "status": np.array(status, dtype=object),
    }


def scan_delta(
    delta_values: np.ndarray | None = None,
    phi: float = 0.03,
    base: FearMemoryParams | None = None,
    t_end: float = 120.0,
) -> dict[str, np.ndarray]:
    if delta_values is None:
        delta_values = np.linspace(0.2, 4.0, 20)
    if base is None:
        base = fear_default

    x_mean = np.zeros_like(delta_values)
    y_mean = np.zeros_like(delta_values)

    for i, delta in enumerate(delta_values):
        p = FearMemoryParams(
            r=base.r,
            K=base.K,
            a=base.a,
            theta=base.theta,
            e=base.e,
            mu=base.mu,
            phi=phi,
            delta=float(delta),
        )
        sol = integrate_fear_memory(p, t_span=(0.0, t_end))
        xm, ym = long_term_mean(sol, burn_in_frac=0.3)
        x_mean[i], y_mean[i] = xm, ym

    return {"delta": delta_values, "x_mean": x_mean, "y_mean": y_mean}


def compare_mechanisms(
    t_end: float = 100.0,
    mechanisms: tuple[MechanismId, ...] | None = None,
) -> dict[str, list]:
    """比较各文献机制的长期平均密度与动力学类型。"""
    if mechanisms is None:
        mechanisms = tuple(MechanismId)

    rows: dict[str, list] = {
        "mechanism": [],
        "label": [],
        "x_mean": [],
        "y_mean": [],
        "status": [],
        "amplitude_x": [],
    }

    from .parameters import MECHANISM_LABELS

    for mid in mechanisms:
        sol = run_mechanism(mid, t_span=(0.0, t_end))
        xm, ym = long_term_mean(sol, burn_in_frac=0.35)
        tail = slice(int(sol.t.size * 0.5), None)
        amp_x = float(np.max(sol.y[0, tail]) - np.min(sol.y[0, tail]))
        rows["mechanism"].append(mid.value)
        rows["label"].append(MECHANISM_LABELS[mid])
        rows["x_mean"].append(xm)
        rows["y_mean"].append(ym)
        rows["status"].append(is_extinct(sol))
        rows["amplitude_x"].append(amp_x)

    return rows


def nce_vs_consumptive_summary(
    t_end: float = 80.0,
) -> dict[str, float]:
    """
    对照“仅消耗”(基线) 与 “消耗+恐惧”(记忆模型) 的猎物平均密度差，
    对应 Preisser (2007) 中 TMI 与 DMI 可分离的思想（简化数值代理）。
    """
    sol_b = run_mechanism(MechanismId.BASELINE, t_span=(0.0, t_end))
    sol_f = run_mechanism(MechanismId.FEAR_MEMORY, t_span=(0.0, t_end))
    xb, _ = long_term_mean(sol_b, 0.3)
    xf, _ = long_term_mean(sol_f, 0.3)
    return {
        "prey_mean_baseline": xb,
        "prey_mean_fear": xf,
        "nce_reduction": xb - xf,
        "relative_reduction_pct": 100.0 * (xb - xf) / max(xb, 1e-9),
    }


def sensitivity_local(
    p: FearMemoryParams | None = None,
    eps: float = 1e-4,
    t_end: float = 100.0,
) -> dict[str, float]:
    if p is None:
        p = fear_default

    def run(params: FearMemoryParams) -> float:
        sol = integrate_fear_memory(params, t_span=(0.0, t_end))
        xm, _ = long_term_mean(sol, burn_in_frac=0.3)
        return xm

    x0 = run(p)
    sens: dict[str, float] = {}

    for name in ("r", "K", "a", "phi", "delta"):
        kwargs = {
            "r": p.r,
            "K": p.K,
            "a": p.a,
            "theta": p.theta,
            "e": p.e,
            "mu": p.mu,
            "phi": p.phi,
            "delta": p.delta,
        }
        v = kwargs[name]
        kwargs[name] = v * (1.0 + eps) if v != 0 else eps
        p_plus = FearMemoryParams(**kwargs)
        x_plus = run(p_plus)
        sens[name] = (x_plus - x0) / (eps * max(abs(v), 1e-12))

    return sens


def equilibrium_bda_fear(p: BDAFearParams | None = None) -> dict[str, float | None]:
    """
    Myint et al. (2025) 正平衡点：由 cp>mq 得 u=λv+λ，λ=m/(cp-mq)，再解 v 的三次方程。
    """
    if p is None:
        p = bda_fear_default
    cp_mq = p.c * p.p - p.m * p.q
    if cp_mq <= 0:
        return {"u_star": None, "v_star": None, "lambda": None, "note": "cp<=mq"}

    lam = p.m / cp_mq
    alpha1 = p.a * lam * p.k
    alpha2 = -(p.p * p.k / (1.0 + p.q * lam) + (p.a * lam * (1.0 + 2.0 * p.k) + p.k * p.d))
    alpha3 = p.r - p.p / (1.0 + p.q * lam) - (p.a * lam + p.d) * (1.0 + p.k)
    alpha4 = p.r - (p.d + p.a * lam)

    coeffs = [alpha1, -alpha2, -alpha3, -alpha4]
    roots = np.roots(coeffs)
    pos_real = [float(r.real) for r in roots if abs(r.imag) < 1e-8 and r.real > 0]
    if not pos_real:
        return {"u_star": None, "v_star": None, "lambda": lam}
    v_star = min(pos_real)
    u_star = lam * v_star + lam
    return {"u_star": u_star, "v_star": v_star, "lambda": lam}


def scan_bda_fear_k(
    k_values: np.ndarray | None = None,
    base: BDAFearParams | None = None,
    t_end: float = 120.0,
) -> dict[str, np.ndarray]:
    """扫描 Myint 模型恐惧参数 k。"""
    from .model import bd_fear_rhs
    from .simulate import integrate_rhs

    if k_values is None:
        k_values = np.linspace(0.0, 0.15, 16)
    if base is None:
        base = bda_fear_default

    u_mean = np.zeros_like(k_values)
    v_mean = np.zeros_like(k_values)
    for i, k in enumerate(k_values):
        p = BDAFearParams(
            r=base.r, d=base.d, a=base.a, k=float(k),
            p=base.p, q=base.q, c=base.c, m=base.m,
        )
        y0 = np.array([0.17, 3.9])
        sol = integrate_rhs(lambda t, s: bd_fear_rhs(t, s, p), y0, t_span=(0.0, t_end))
        u_mean[i], v_mean[i] = long_term_mean(sol, burn_in_frac=0.35)
    return {"k": k_values, "u_mean": u_mean, "v_mean": v_mean}


def equilibrium_baseline(p=None) -> dict[str, float | None]:
    if p is None:
        p = baseline_default

    from scipy.optimize import root

    def eq(z: np.ndarray) -> np.ndarray:
        x, y = z
        if x <= 0 or y <= 0:
            return np.array([1e3, 1e3])
        attack = p.a * x * y / (1 + p.theta * x)
        f1 = p.r * (1 - x / p.K) - attack / x
        f2 = p.e * attack / y - p.mu
        return np.array([f1, f2])

    for g in ((30, 5), (60, 10), (80, 3)):
        res = root(eq, np.array(g, dtype=float))
        if res.success and res.x[0] > 0 and res.x[1] > 0:
            return {"x_star": float(res.x[0]), "y_star": float(res.x[1])}
    return {"x_star": None, "y_star": None}
