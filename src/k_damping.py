"""
恐惧参数 k 削弱振荡：文献依据 + 可复现数值检验。

模型：Myint et al. (2025) B-D + 恐惧 f(k,v)=1/(1+kv)（`bd_fear_rhs`）。

文献要点
--------
- Wang et al. (2016, *J. Math. Biol.*)：恐惧因子 1/(1+kv) 降低有效繁殖，可改变平衡点稳定性。
- Wang et al. (2019, *Ecol. Lett.*)：非消耗效应（NCE）可使原本周期振荡的系统趋于稳定共存。
- Myint et al. (2025, arXiv:2506.22070)：在 B-D 框架下数值展示 k=0 极限环 vs k>0 趋稳。

本项目采用三种互补检验（由强到弱、由局部到全局）：
1. **平衡点 Jacobian 特征值扫描** — 判断 Hopf 型失稳边界（Re λ 由正变负）。
2. **长期数值振幅扫描** — 后半程 max-min 与相对振幅 A/ū。
3. **峰值衰减率** — 对衰减振荡拟合 successive-peak 比值，量化“振荡被压缩”。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .analysis import equilibrium_bda_fear
from .model import bd_fear_rhs
from .parameters import BDAFearParams, bda_fear_default
from .simulate import integrate_rhs, is_extinct, long_term_mean


StabilityClass = Literal[
    "no_equilibrium",
    "unstable_focus",
    "unstable_node",
    "stable_focus",
    "stable_node",
    "center_like",
]


@dataclass
class EigenvalueScanRow:
    k: float
    u_star: float | None
    v_star: float | None
    re_max: float | None
    im_abs: float | None
    trace: float | None
    det: float | None
    stability: StabilityClass
    omega: float | None  # 线性化角频率 |Im λ|


def bd_fear_jacobian(u: float, v: float, p: BDAFearParams) -> np.ndarray:
    """B-D+恐惧模型在 (u,v) 处的 2×2 Jacobian。"""
    h = 1.0 + p.q * u + v
    h2 = h * h
    fear_denom = 1.0 + p.k * v

    df_du = p.r / fear_denom - p.d - 2.0 * p.a * u - p.p * v * (1.0 + v) / h2
    df_dv = -p.r * p.k * u / (fear_denom * fear_denom) - p.p * u * (1.0 + p.q * u) / h2

    g_val = -p.m + p.c * p.p * u / h
    dg_du = v * p.c * p.p * (1.0 + v) / h2
    dg_dv = g_val - v * p.c * p.p * u / h2

    return np.array([[df_du, df_dv], [dg_du, dg_dv]], dtype=float)


def _classify_stability(re_max: float, im_abs: float, tol: float = 1e-6) -> StabilityClass:
    if abs(re_max) < tol and im_abs > tol:
        return "center_like"
    if re_max > tol:
        return "unstable_focus" if im_abs > tol else "unstable_node"
    return "stable_focus" if im_abs > tol else "stable_node"


def equilibrium_at_k(k: float, base: BDAFearParams | None = None) -> dict[str, float | None]:
    if base is None:
        base = bda_fear_default
    p = BDAFearParams(
        r=base.r, d=base.d, a=base.a, k=float(k),
        p=base.p, q=base.q, c=base.c, m=base.m,
    )
    return equilibrium_bda_fear(p)


def eigenvalue_at_k(k: float, base: BDAFearParams | None = None) -> EigenvalueScanRow:
    eq = equilibrium_at_k(k, base)
    u_star, v_star = eq.get("u_star"), eq.get("v_star")
    if u_star is None or v_star is None:
        return EigenvalueScanRow(
            k=float(k), u_star=None, v_star=None,
            re_max=None, im_abs=None, trace=None, det=None,
            stability="no_equilibrium", omega=None,
        )

    p = BDAFearParams(
        r=base.r if base else bda_fear_default.r,
        d=base.d if base else bda_fear_default.d,
        a=base.a if base else bda_fear_default.a,
        k=float(k),
        p=base.p if base else bda_fear_default.p,
        q=base.q if base else bda_fear_default.q,
        c=base.c if base else bda_fear_default.c,
        m=base.m if base else bda_fear_default.m,
    )
    j = bd_fear_jacobian(float(u_star), float(v_star), p)
    ev = np.linalg.eigvals(j)
    re = np.real(ev)
    im = np.imag(ev)
    idx = int(np.argmax(re))
    re_max = float(re[idx])
    im_abs = float(abs(im[idx]))
    return EigenvalueScanRow(
        k=float(k),
        u_star=float(u_star),
        v_star=float(v_star),
        re_max=re_max,
        im_abs=im_abs,
        trace=float(np.trace(j)),
        det=float(np.linalg.det(j)),
        stability=_classify_stability(re_max, im_abs),
        omega=im_abs if im_abs > 1e-8 else None,
    )


def scan_k_eigenvalues(
    k_values: np.ndarray | None = None,
    base: BDAFearParams | None = None,
) -> list[EigenvalueScanRow]:
    if k_values is None:
        k_values = np.linspace(0.0, 0.2, 41)
    return [eigenvalue_at_k(float(k), base) for k in k_values]


def estimate_hopf_k(
    rows: list[EigenvalueScanRow],
) -> float | None:
    """
    在扫描网格上估计 Re λ 由正变负的 k（Hopf 型阈值近似）。
    返回 None 表示在扫描范围内未发生穿越。
    """
    for i in range(len(rows) - 1):
        a, b = rows[i], rows[i + 1]
        if a.re_max is None or b.re_max is None:
            continue
        if a.re_max > 0.0 and b.re_max <= 0.0:
            # 线性插值
            t = a.re_max / (a.re_max - b.re_max)
            return float(a.k + t * (b.k - a.k))
    return None


def _find_peaks(y: np.ndarray, min_distance: int = 5) -> np.ndarray:
    """简单局部极大值索引。"""
    peaks: list[int] = []
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1]:
            if not peaks or i - peaks[-1] >= min_distance:
                peaks.append(i)
    return np.array(peaks, dtype=int)


def peak_damping_metrics(
    t: np.ndarray,
    u: np.ndarray,
    u_star: float | None = None,
    burn_in_frac: float = 0.25,
) -> dict[str, float | None]:
    """
    从猎物时间序列提取振荡指标：
    - amplitude: 后半程 max-min
    - rel_amplitude: amplitude / mean
    - peak_decay_ratio: 末峰/首峰（相对平衡距）
    - n_peaks: 后半程峰个数
    """
    n = t.size
    i0 = int(n * burn_in_frac)
    tail_u = u[i0:]
    tail_t = t[i0:]
    if tail_u.size < 10:
        return {
            "amplitude": None, "rel_amplitude": None,
            "peak_decay_ratio": None, "n_peaks": 0,
        }

    amp = float(np.max(tail_u) - np.min(tail_u))
    u_mean = float(np.mean(tail_u))
    rel_amp = amp / max(abs(u_mean), 1e-9)

    ref = u_star if u_star is not None else u_mean
    peaks = _find_peaks(tail_u)
    if peaks.size < 2:
        return {
            "amplitude": amp,
            "rel_amplitude": rel_amp,
            "peak_decay_ratio": None,
            "n_peaks": int(peaks.size),
        }

    dev = np.abs(tail_u[peaks] - ref)
    if dev[0] < 1e-12:
        ratio = None
    else:
        ratio = float(dev[-1] / dev[0])

    return {
        "amplitude": amp,
        "rel_amplitude": rel_amp,
        "peak_decay_ratio": ratio,
        "n_peaks": int(peaks.size),
    }


def simulate_bda_at_k(
    k: float,
    base: BDAFearParams | None = None,
    y0: np.ndarray | None = None,
    t_end: float = 120.0,
    n_points: int = 1200,
):
    if base is None:
        base = bda_fear_default
    if y0 is None:
        y0 = np.array([0.17, 3.9])
    p = BDAFearParams(
        r=base.r, d=base.d, a=base.a, k=float(k),
        p=base.p, q=base.q, c=base.c, m=base.m,
    )
    return integrate_rhs(
        lambda t, s: bd_fear_rhs(t, s, p),
        y0,
        t_span=(0.0, t_end),
        n_points=n_points,
    )


@dataclass
class KDampingScanRow:
    k: float
    u_mean: float
    v_mean: float
    amplitude_u: float
    amplitude_v: float
    rel_amplitude_u: float
    peak_decay_ratio: float | None
    n_peaks: int
    re_max: float | None
    stability: StabilityClass
    status: str


def scan_k_damping(
    k_values: np.ndarray | None = None,
    base: BDAFearParams | None = None,
    t_end: float = 120.0,
    y0: np.ndarray | None = None,
) -> list[KDampingScanRow]:
    """
    综合扫描：每个 k 上同时做
    - 局部稳定性（Jacobian 特征值）
    - 全局振荡幅度与峰值衰减
    """
    if k_values is None:
        k_values = np.linspace(0.0, 0.18, 19)
    if y0 is None:
        y0 = np.array([0.17, 3.9])

    rows: list[KDampingScanRow] = []
    for k in k_values:
        kf = float(k)
        ev = eigenvalue_at_k(kf, base)
        sol = simulate_bda_at_k(kf, base=base, y0=y0, t_end=t_end)
        u_mean, v_mean = long_term_mean(sol, burn_in_frac=0.35)
        tail = slice(int(sol.t.size * 0.5), None)
        amp_u = float(np.max(sol.y[0, tail]) - np.min(sol.y[0, tail]))
        amp_v = float(np.max(sol.y[1, tail]) - np.min(sol.y[1, tail]))
        pk = peak_damping_metrics(
            sol.t, sol.y[0], u_star=ev.u_star, burn_in_frac=0.35,
        )
        rel_u = amp_u / max(abs(u_mean), 1e-9)
        extinction_status = is_extinct(sol, scales=(1.0, 1.0))
        status = "coexist" if extinction_status == "coexist" else "extinct"
        rows.append(
            KDampingScanRow(
                k=kf,
                u_mean=u_mean,
                v_mean=v_mean,
                amplitude_u=amp_u,
                amplitude_v=amp_v,
                rel_amplitude_u=rel_u,
                peak_decay_ratio=pk["peak_decay_ratio"],
                n_peaks=int(pk["n_peaks"] or 0),
                re_max=ev.re_max,
                stability=ev.stability,
                status=status,
            )
        )
    return rows


def k_damping_summary(
    scan_rows: list[KDampingScanRow],
    eigen_rows: list[EigenvalueScanRow],
) -> dict:
    """生成文字摘要，便于写入报告或 JSON。"""
    hopf_k = estimate_hopf_k(eigen_rows)

    k0 = next((r for r in scan_rows if abs(r.k) < 1e-12), scan_rows[0])
    k_max = max(scan_rows, key=lambda r: r.k)
    most_damped = min(scan_rows, key=lambda r: r.rel_amplitude_u)

    return {
        "literature": [
            "Wang et al. (2016): fear factor 1/(1+kv) on reproduction",
            "Wang et al. (2019): NCE can stabilize cyclic predator-prey dynamics",
            "Myint et al. (2025): B-D + fear, k=0 limit cycle vs k>0 convergence",
        ],
        "methods": [
            "Jacobian eigenvalue scan at (u*,v*) vs k",
            "Numerical amplitude A=max-min in tail vs k",
            "Peak decay ratio (last/first peak deviation from u*)",
        ],
        "hopf_k_estimate": hopf_k,
        "k0": {
            "k": k0.k,
            "rel_amplitude_u": k0.rel_amplitude_u,
            "re_max": k0.re_max,
            "stability": k0.stability,
        },
        "k_max_scanned": {
            "k": k_max.k,
            "rel_amplitude_u": k_max.rel_amplitude_u,
            "re_max": k_max.re_max,
            "stability": k_max.stability,
        },
        "strongest_damping_at": {
            "k": most_damped.k,
            "rel_amplitude_u": most_damped.rel_amplitude_u,
        },
        "conclusion_hint": (
            "若随 k 增大 max Re(λ) 由正变负，则存在 Hopf 型阈值 k_H，恐惧使周期解失稳→稳定。"
            "默认 B-D 参数下平衡点对所有 k 已局部稳定，此时应解读为："
            "k 改变 Re(λ) 与瞬态振幅，并与 φ 扫描（Wang 2019 主模型）、"
            "数据拟合得到的 k 值（calibrate_bda）交叉验证。"
        ),
    }
