"""参数扫描、敏感性、文献机制对比分析。"""

from __future__ import annotations

from dataclasses import asdict, replace

import numpy as np
from scipy.optimize import brentq

from .literature import run_mechanism
from .model import bd_fear_rhs
from .parameters import (
    BDAFearParams,
    FearMemoryParams,
    MechanismId,
    baseline_default,
    bda_fear_default,
    bda_no_fear_default,
    fear_default,
    fear_foraging_default,
    fear_handling_default,
    fear_saturating_default,
)
from .simulate import (
    integrate_fear_memory,
    integrate_rhs,
    integrate_until_converged,
    is_extinct,
)


def scan_phi(
    phi_values: np.ndarray | None = None,
    base: FearMemoryParams | None = None,
    t_end: float = 120.0,
) -> dict[str, np.ndarray]:
    if phi_values is None:
        phi_values = np.linspace(0.0, 0.06, 31)
    if base is None:
        base = fear_default

    x_mean = np.zeros_like(phi_values, dtype=float)
    y_mean = np.zeros_like(phi_values, dtype=float)
    status = []
    convergence_status = []
    t_end_used = np.zeros_like(phi_values, dtype=float)
    extensions = np.zeros_like(phi_values, dtype=int)

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
        result = integrate_until_converged(
            lambda end, points: integrate_fear_memory(
                p, t_span=(0.0, end), n_points=points
            ),
            t_end=t_end,
            scales=(base.K, base.K),
        )
        sol = result.sol
        xm, ym = result.metrics.means
        x_mean[i], y_mean[i] = xm, ym
        status.append(is_extinct(sol, scales=(base.K, base.K)))
        convergence_status.append(result.convergence.status)
        t_end_used[i] = result.t_end_used
        extensions[i] = result.extensions

    return {
        "phi": phi_values,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "status": np.array(status, dtype=object),
        "convergence_status": np.array(convergence_status, dtype=object),
        "t_end_used": t_end_used,
        "extensions": extensions,
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

    x_mean = np.zeros_like(delta_values, dtype=float)
    y_mean = np.zeros_like(delta_values, dtype=float)
    convergence_status = []
    t_end_used = np.zeros_like(delta_values, dtype=float)
    extensions = np.zeros_like(delta_values, dtype=int)

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
        result = integrate_until_converged(
            lambda end, points: integrate_fear_memory(
                p, t_span=(0.0, end), n_points=points
            ),
            t_end=t_end,
            scales=(base.K, base.K),
        )
        xm, ym = result.metrics.means
        x_mean[i], y_mean[i] = xm, ym
        convergence_status.append(result.convergence.status)
        t_end_used[i] = result.t_end_used
        extensions[i] = result.extensions

    return {
        "delta": delta_values,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "convergence_status": np.array(convergence_status, dtype=object),
        "t_end_used": t_end_used,
        "extensions": extensions,
    }


def compare_mechanisms(
    t_end: float = 100.0,
    mechanisms: tuple[MechanismId, ...] | None = None,
    comparison_mode: str = "equivalent",
) -> dict[str, list]:
    """按各模型体系自己的无恐惧基线比较长期均值与相对振幅。"""
    if mechanisms is None:
        mechanisms = tuple(MechanismId)
    if comparison_mode == "equivalent":
        params_by_mechanism = equivalent_fear_parameters()
    elif comparison_mode == "default":
        params_by_mechanism = {}
    else:
        raise ValueError("comparison_mode must be 'equivalent' or 'default'")

    rows: dict[str, list] = {
        "mechanism": [],
        "label": [],
        "model_family": [],
        "baseline_mechanism": [],
        "x_mean": [],
        "y_mean": [],
        "status": [],
        "amplitude_x": [],
        "amplitude_y": [],
        "relative_amplitude_x": [],
        "relative_amplitude_y": [],
        "x_mean_change_pct": [],
        "y_mean_change_pct": [],
        "relative_amplitude_x_change_pct": [],
        "relative_amplitude_y_change_pct": [],
        "convergence_status": [],
        "t_end_used": [],
        "extensions": [],
        "baseline_convergence_status": [],
        "comparison_valid": [],
    }

    from .parameters import MECHANISM_LABELS

    def family_and_baseline(mid: MechanismId) -> tuple[str, MechanismId]:
        if mid in (MechanismId.BDA_BASELINE, MechanismId.BDA_FEAR):
            return "B-D", MechanismId.BDA_BASELINE
        return "Holling II", MechanismId.BASELINE

    def metrics(mid: MechanismId) -> dict[str, float | str]:
        scales = (
            (1.0, 1.0)
            if mid in (MechanismId.BDA_BASELINE, MechanismId.BDA_FEAR)
            else (baseline_default.K, baseline_default.K)
        )
        result = integrate_until_converged(
            lambda end, points: run_mechanism(
                mid,
                t_span=(0.0, end),
                n_points=points,
                params=params_by_mechanism.get(mid),
            ),
            t_end=t_end,
            scales=scales,
        )
        sol = result.sol
        xm, ym = result.metrics.means
        amp_x, amp_y = result.metrics.amplitudes
        return {
            "x_mean": xm,
            "y_mean": ym,
            "status": is_extinct(sol, scales=scales),
            "amplitude_x": amp_x,
            "amplitude_y": amp_y,
            "relative_amplitude_x": amp_x / max(abs(xm), 1e-12),
            "relative_amplitude_y": amp_y / max(abs(ym), 1e-12),
            "convergence_status": result.convergence.status,
            "t_end_used": result.t_end_used,
            "extensions": result.extensions,
        }

    needed_baselines = {family_and_baseline(mid)[1] for mid in mechanisms}
    baseline_metrics = {mid: metrics(mid) for mid in needed_baselines}

    def pct_change(value: float, baseline: float) -> float:
        if abs(baseline) < 1e-12:
            return 0.0 if abs(value) < 1e-12 else float("nan")
        return 100.0 * (value / baseline - 1.0)

    for mid in mechanisms:
        family, baseline_mid = family_and_baseline(mid)
        current = metrics(mid)
        baseline = baseline_metrics[baseline_mid]
        rows["mechanism"].append(mid.value)
        rows["label"].append(MECHANISM_LABELS[mid])
        rows["model_family"].append(family)
        rows["baseline_mechanism"].append(baseline_mid.value)
        rows["baseline_convergence_status"].append(baseline["convergence_status"])
        rows["comparison_valid"].append(
            current["convergence_status"] == "converged"
            and baseline["convergence_status"] == "converged"
        )
        for key in (
            "x_mean",
            "y_mean",
            "status",
            "amplitude_x",
            "amplitude_y",
            "relative_amplitude_x",
            "relative_amplitude_y",
            "convergence_status",
            "t_end_used",
            "extensions",
        ):
            rows[key].append(current[key])
        for key in ("x_mean", "y_mean", "relative_amplitude_x", "relative_amplitude_y"):
            rows[f"{key}_change_pct"].append(
                pct_change(float(current[key]), float(baseline[key]))
            )

    return rows


def equivalent_fear_calibration(target_suppression: float = 0.20) -> dict:
    """
    在各模型体系无恐惧正平衡点处，将机制参数校准到相同抑制比例。

    繁殖抑制机制匹配猎物增长项降低比例；觅食/处理时间机制匹配
    Holling II 捕食率降低比例；B-D 匹配繁殖因子 1/(1+kv)。
    """
    if not 0.0 < target_suppression < 1.0:
        raise ValueError("target_suppression must be between 0 and 1")

    holling_eq = equilibrium_baseline(baseline_default)
    bda_eq = equilibrium_bda_fear(bda_no_fear_default)
    x_ref = float(holling_eq["x_star"])
    y_ref = float(holling_eq["y_star"])
    u_ref = float(bda_eq["u_star"])
    v_ref = float(bda_eq["v_star"])
    remaining = 1.0 - target_suppression
    memory_ref = y_ref / fear_default.delta

    phi_memory = target_suppression * (1.0 - x_ref / baseline_default.K) / memory_ref
    phi_instant = target_suppression * (1.0 - x_ref / baseline_default.K) / y_ref
    h_saturating = y_ref
    phi_saturating = target_suppression * (y_ref + h_saturating) / y_ref
    psi_foraging = target_suppression / (remaining * y_ref)
    psi_handling = (
        (1.0 + baseline_default.theta * x_ref)
        * target_suppression
        / (remaining * baseline_default.theta * x_ref * y_ref)
    )
    k_bda = target_suppression / (remaining * v_ref)

    params = {
        MechanismId.BASELINE: baseline_default,
        MechanismId.FEAR_MEMORY: replace(fear_default, phi=phi_memory),
        MechanismId.FEAR_INSTANT: replace(fear_default, phi=phi_instant),
        MechanismId.FEAR_SATURATING: replace(
            fear_saturating_default,
            phi=phi_saturating,
            h=h_saturating,
        ),
        MechanismId.FEAR_FORAGING: replace(fear_foraging_default, psi=psi_foraging),
        MechanismId.FEAR_HANDLING: replace(fear_handling_default, psi=psi_handling),
        MechanismId.BDA_BASELINE: bda_no_fear_default,
        MechanismId.BDA_FEAR: replace(bda_fear_default, k=k_bda),
    }

    attack_ratio_handling = (
        1.0 + baseline_default.theta * x_ref
    ) / (
        1.0
        + baseline_default.theta
        * (1.0 + psi_handling * y_ref)
        * x_ref
    )
    achieved = {
        MechanismId.FEAR_MEMORY: phi_memory * memory_ref / (1.0 - x_ref / baseline_default.K),
        MechanismId.FEAR_INSTANT: phi_instant * y_ref / (1.0 - x_ref / baseline_default.K),
        MechanismId.FEAR_SATURATING: phi_saturating * y_ref / (y_ref + h_saturating),
        MechanismId.FEAR_FORAGING: 1.0 - 1.0 / (1.0 + psi_foraging * y_ref),
        MechanismId.FEAR_HANDLING: 1.0 - attack_ratio_handling,
        MechanismId.BDA_FEAR: 1.0 - 1.0 / (1.0 + k_bda * v_ref),
    }
    return {
        "target_suppression": target_suppression,
        "reference_states": {
            "holling_no_fear_equilibrium": {"prey": x_ref, "predator": y_ref},
            "bda_k0_equilibrium": {"prey": u_ref, "predator": v_ref},
            "memory_steady_state": memory_ref,
        },
        "calibration_rule": {
            "fear_memory": "growth-term suppression at Holling equilibrium, M*=y*/delta",
            "fear_instant": "growth-term suppression at Holling equilibrium",
            "fear_saturating": "growth-term suppression at Holling equilibrium, h=y*",
            "fear_foraging": "predation-rate suppression at Holling equilibrium",
            "fear_handling": "predation-rate suppression at Holling equilibrium",
            "bda_fear": "B-D reproduction-factor suppression at k=0 equilibrium",
        },
        "params": params,
        "achieved_suppression": achieved,
    }


def equivalent_fear_parameters(target_suppression: float = 0.20) -> dict[MechanismId, object]:
    """返回用于公平机制对照的等效恐惧参数。"""
    return equivalent_fear_calibration(target_suppression)["params"]


def equivalent_fear_calibration_summary(target_suppression: float = 0.20) -> dict:
    """返回可 JSON 序列化的等效恐惧校准摘要。"""
    calibration = equivalent_fear_calibration(target_suppression)
    return {
        **{k: v for k, v in calibration.items() if k not in ("params", "achieved_suppression")},
        "params": {
            mid.value: asdict(params)
            for mid, params in calibration["params"].items()
        },
        "achieved_suppression": {
            mid.value: value
            for mid, value in calibration["achieved_suppression"].items()
        },
    }


def nce_vs_consumptive_summary(
    t_end: float = 80.0,
) -> dict[str, float]:
    """
    对照“仅消耗”(基线) 与 “消耗+恐惧”(记忆模型) 的猎物平均密度差，
    对应 Preisser (2007) 中 TMI 与 DMI 可分离的思想（简化数值代理）。
    """
    base_result = integrate_until_converged(
        lambda end, points: run_mechanism(
            MechanismId.BASELINE, t_span=(0.0, end), n_points=points
        ),
        t_end=t_end,
        scales=(baseline_default.K, baseline_default.K),
    )
    fear_result = integrate_until_converged(
        lambda end, points: run_mechanism(
            MechanismId.FEAR_MEMORY, t_span=(0.0, end), n_points=points
        ),
        t_end=t_end,
        scales=(baseline_default.K, baseline_default.K),
    )
    xb = base_result.metrics.means[0]
    xf = fear_result.metrics.means[0]
    if (
        base_result.convergence.status != "converged"
        or fear_result.convergence.status != "converged"
    ):
        raise RuntimeError("NCE comparison trajectories did not converge")
    return {
        "prey_mean_baseline": xb,
        "prey_mean_fear": xf,
        "nce_reduction": xb - xf,
        "relative_reduction_pct": 100.0 * (xb - xf) / max(xb, 1e-9),
        "baseline_convergence_status": base_result.convergence.status,
        "fear_convergence_status": fear_result.convergence.status,
    }


def sensitivity_local(
    p: FearMemoryParams | None = None,
    eps: float = 1e-4,
    t_end: float = 100.0,
) -> dict[str, float]:
    if p is None:
        p = fear_default

    def run(params: FearMemoryParams) -> float:
        result = integrate_until_converged(
            lambda end, points: integrate_fear_memory(
                params, t_span=(0.0, end), n_points=points
            ),
            t_end=t_end,
            scales=(params.K, params.K),
        )
        if result.convergence.status != "converged":
            raise RuntimeError("sensitivity trajectory did not converge")
        return result.metrics.means[0]

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
    Myint et al. (2025) 正平衡点。

    由 dv/dt=0 和 cp>mq 得 u=λ(1+v)，λ=m/(cp-mq)；代入
    du/(u dt)=0 后求唯一正根，并在返回前校验原始二维 RHS 残差。
    """
    if p is None:
        p = bda_fear_default
    cp_mq = p.c * p.p - p.m * p.q
    if cp_mq <= 0:
        return {"u_star": None, "v_star": None, "lambda": None, "note": "cp<=mq"}

    lam = p.m / cp_mq

    def prey_per_capita(v: float) -> float:
        u = lam * (1.0 + v)
        return p.r / (1.0 + p.k * v) - p.d - p.a * u - p.m * v / (p.c * u)

    if prey_per_capita(0.0) <= 0.0:
        return {
            "u_star": None,
            "v_star": None,
            "lambda": lam,
            "note": "no positive coexistence root",
        }

    upper = 1.0
    while prey_per_capita(upper) > 0.0 and upper < 1e12:
        upper *= 2.0
    if prey_per_capita(upper) > 0.0:
        return {
            "u_star": None,
            "v_star": None,
            "lambda": lam,
            "note": "failed to bracket coexistence root",
        }

    v_star = float(brentq(prey_per_capita, 0.0, upper, xtol=1e-12, rtol=1e-12))
    u_star = float(lam * (1.0 + v_star))
    rhs_residual = float(np.linalg.norm(bd_fear_rhs(0.0, np.array([u_star, v_star]), p), ord=np.inf))
    residual_tol = 1e-9 * max(1.0, u_star, v_star)
    if u_star <= 0.0 or v_star <= 0.0 or rhs_residual > residual_tol:
        return {
            "u_star": None,
            "v_star": None,
            "lambda": lam,
            "rhs_residual": rhs_residual,
            "note": "invalid coexistence root",
        }
    return {
        "u_star": u_star,
        "v_star": v_star,
        "lambda": lam,
        "rhs_residual": rhs_residual,
    }


def scan_bda_fear_k(
    k_values: np.ndarray | None = None,
    base: BDAFearParams | None = None,
    t_end: float = 120.0,
) -> dict[str, np.ndarray]:
    """扫描 Myint 模型恐惧参数 k。"""
    from .model import bd_fear_rhs
    if k_values is None:
        k_values = np.linspace(0.0, 0.15, 16)
    if base is None:
        base = bda_fear_default

    u_mean = np.zeros_like(k_values, dtype=float)
    v_mean = np.zeros_like(k_values, dtype=float)
    convergence_status = []
    t_end_used = np.zeros_like(k_values, dtype=float)
    extensions = np.zeros_like(k_values, dtype=int)
    for i, k in enumerate(k_values):
        p = BDAFearParams(
            r=base.r, d=base.d, a=base.a, k=float(k),
            p=base.p, q=base.q, c=base.c, m=base.m,
        )
        y0 = np.array([0.17, 3.9])
        result = integrate_until_converged(
            lambda end, points: integrate_rhs(
                lambda t, s: bd_fear_rhs(t, s, p),
                y0,
                t_span=(0.0, end),
                n_points=points,
            ),
            t_end=t_end,
            scales=(1.0, 1.0),
        )
        u_mean[i], v_mean[i] = result.metrics.means
        convergence_status.append(result.convergence.status)
        t_end_used[i] = result.t_end_used
        extensions[i] = result.extensions
    return {
        "k": k_values,
        "u_mean": u_mean,
        "v_mean": v_mean,
        "convergence_status": np.array(convergence_status, dtype=object),
        "t_end_used": t_end_used,
        "extensions": extensions,
    }


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
