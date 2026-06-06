"""Myint et al. (2025) B-D + 恐惧：二维反应–扩散有限差分（Neumann 边界）。"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .k_damping import bd_fear_jacobian
from .model import bd_fear_rhs
from .parameters import BDAFearParams, bda_fear_default
from .simulate import integrate_rhs, integrate_until_converged


@dataclass(frozen=True)
class PDE2DConfig:
    """2D 斑图模拟配置。"""

    nx: int = 128
    ny: int = 128
    lx: float = 10.0 * np.pi
    ly: float = 10.0 * np.pi
    t_end: float = 200.0
    d1: float = 1.0
    d2: float = 0.1
    perturbation: float = 0.01
    u_star: float | None = None
    v_star: float | None = None
    clip_negative: bool = True
    dt: float | None = None
    store_stride: int = 1


@dataclass
class PDE2DResult:
    x: np.ndarray
    y: np.ndarray
    u: np.ndarray
    v: np.ndarray
    t_final: float
    config: PDE2DConfig
    params: BDAFearParams
    u_star: float
    v_star: float
    n_steps: int
    dt: float


def _reaction_uv(u: np.ndarray, v: np.ndarray, p: BDAFearParams) -> tuple[np.ndarray, np.ndarray]:
    """向量化局部反应项（与 bd_fear_rhs 一致）。"""
    u = np.maximum(u, 0.0)
    v = np.maximum(v, 0.0)
    denom = 1.0 + p.q * u + v
    du = p.r * u / (1.0 + p.k * v) - p.d * u - p.a * u * u - p.p * u * v / denom
    dv = v * (-p.m + p.c * p.p * u / denom)
    return du, dv


def _laplacian_neumann(field: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """五点格式 Laplacian，齐次 Neumann（零通量）边界。"""
    lap = np.zeros_like(field)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)

    # 内部
    lap[1:-1, 1:-1] = (
        (field[2:, 1:-1] - 2.0 * field[1:-1, 1:-1] + field[:-2, 1:-1]) * inv_dx2
        + (field[1:-1, 2:] - 2.0 * field[1:-1, 1:-1] + field[1:-1, :-2]) * inv_dy2
    )

    # 边界：ghost cell 对称
    # 左/右
    lap[0, 1:-1] = (
        (field[1, 1:-1] - field[0, 1:-1]) * inv_dx2 * 2.0
        + (field[0, 2:] - 2.0 * field[0, 1:-1] + field[0, :-2]) * inv_dy2
    )
    lap[-1, 1:-1] = (
        (field[-2, 1:-1] - field[-1, 1:-1]) * inv_dx2 * 2.0
        + (field[-1, 2:] - 2.0 * field[-1, 1:-1] + field[-1, :-2]) * inv_dy2
    )
    # 下/上
    lap[1:-1, 0] = (
        (field[2:, 0] - 2.0 * field[1:-1, 0] + field[:-2, 0]) * inv_dx2
        + (field[1:-1, 1] - field[1:-1, 0]) * inv_dy2 * 2.0
    )
    lap[1:-1, -1] = (
        (field[2:, -1] - 2.0 * field[1:-1, -1] + field[:-2, -1]) * inv_dx2
        + (field[1:-1, -2] - field[1:-1, -1]) * inv_dy2 * 2.0
    )
    # 角点
    lap[0, 0] = (field[1, 0] - field[0, 0]) * inv_dx2 * 2.0 + (field[0, 1] - field[0, 0]) * inv_dy2 * 2.0
    lap[0, -1] = (field[1, -1] - field[0, -1]) * inv_dx2 * 2.0 + (field[0, -2] - field[0, -1]) * inv_dy2 * 2.0
    lap[-1, 0] = (field[-2, 0] - field[-1, 0]) * inv_dx2 * 2.0 + (field[-1, 1] - field[-1, 0]) * inv_dy2 * 2.0
    lap[-1, -1] = (field[-2, -1] - field[-1, -1]) * inv_dx2 * 2.0 + (field[-1, -2] - field[-1, -1]) * inv_dy2 * 2.0

    return lap


def _suggest_dt(cfg: PDE2DConfig) -> float:
    dx = cfg.lx / cfg.nx
    dy = cfg.ly / cfg.ny
    d_max = max(cfg.d1, cfg.d2)
    # 2D 显式扩散稳定因子 ~ dx^2/(4d)
    dt_diff = 0.2 * min(dx * dx, dy * dy) / max(d_max, 1e-12)
    return min(dt_diff, 0.05)


def initial_condition_2d(
    cfg: PDE2DConfig,
    u_star: float,
    v_star: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    2D 扩展 Myint 论文初值：
    u0 = u* + eps cos(x/2) cos(y/2),  v0 = v* + eps cos(x) cos(y)
    """
    x = np.linspace(0.0, cfg.lx, cfg.nx, endpoint=False)
    y = np.linspace(0.0, cfg.ly, cfg.ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    amp = cfg.perturbation
    u0 = u_star + amp * np.cos(X / 2.0) * np.cos(Y / 2.0)
    v0 = v_star + amp * np.cos(X) * np.cos(Y)
    u0 = np.maximum(u0, 0.0)
    v0 = np.maximum(v0, 0.0)
    return x, y, u0, v0


def simulate_bda_fear_2d(
    params: BDAFearParams | None = None,
    config: PDE2DConfig | None = None,
    u0: np.ndarray | None = None,
    v0: np.ndarray | None = None,
    progress: bool = False,
) -> PDE2DResult:
    """显式 Euler 推进 2D 反应–扩散系统至 t_end。"""
    if params is None:
        params = bda_fear_default
    if config is None:
        config = PDE2DConfig()

    if config.u_star is not None and config.v_star is not None:
        u_star = config.u_star
        v_star = config.v_star
    else:
        u_num, v_num = resolve_bda_coexistence(params)
        u_star = config.u_star if config.u_star is not None else u_num
        v_star = config.v_star if config.v_star is not None else v_num

    x, y, u_init, v_init = initial_condition_2d(config, u_star, v_star)
    u = u0.copy() if u0 is not None else u_init
    v = v0.copy() if v0 is not None else v_init

    dx = config.lx / config.nx
    dy = config.ly / config.ny
    dt = config.dt if config.dt is not None else _suggest_dt(config)
    n_steps = int(np.ceil(config.t_end / dt))
    dt = config.t_end / n_steps

    for step in range(n_steps):
        lap_u = _laplacian_neumann(u, dx, dy)
        lap_v = _laplacian_neumann(v, dx, dy)
        ru, rv = _reaction_uv(u, v, params)
        u = u + dt * (config.d1 * lap_u + ru)
        v = v + dt * (config.d2 * lap_v + rv)
        if config.clip_negative:
            u = np.maximum(u, 0.0)
            v = np.maximum(v, 0.0)
        if progress and (step + 1) % max(n_steps // 10, 1) == 0:
            print(f"    step {step + 1}/{n_steps}  t={ (step + 1) * dt:.2f}")

    return PDE2DResult(
        x=x,
        y=y,
        u=u,
        v=v,
        t_final=config.t_end,
        config=config,
        params=params,
        u_star=u_star,
        v_star=v_star,
        n_steps=n_steps,
        dt=dt,
    )


def resolve_bda_coexistence(
    params: BDAFearParams | None = None,
    y0: tuple[float, float] = (0.17, 3.9),
    t_end: float = 500.0,
) -> tuple[float, float]:
    """
    数值求 B--D+恐惧 稳定共存态（ODE 长时间积分均值）。
    用于 2D PDE 初值中心，避免误用文献图注鞍点。
    """
    if params is None:
        params = bda_fear_default
    result = integrate_until_converged(
        lambda end, points: integrate_rhs(
            lambda t, s: bd_fear_rhs(t, s, params),
            np.array(y0, dtype=float),
            t_span=(0.0, end),
            n_points=points,
        ),
        t_end=t_end,
        scales=(1.0, 1.0),
    )
    if result.convergence.status != "converged":
        raise RuntimeError(
            f"B-D coexistence trajectory did not converge by t={result.t_end_used:g}"
        )
    if max(result.metrics.amplitudes) > 1e-6:
        raise RuntimeError("B-D coexistence trajectory converged to a cycle, not an equilibrium")
    u, v = result.metrics.means
    return float(u), float(v)


def turing_max_real_eigenvalue(
    J: np.ndarray,
    d1: float,
    d2: float,
    k_max: float = 20.0,
    n_k: int = 4000,
) -> tuple[float, float, float, float]:
    """
    均匀态线性稳定性：max Re λ(k) of det(λI - J + k²D)=0。
    返回 (max_Re, k_at_max, trace(J), det(J))。
    """
    tr = float(np.trace(J))
    det = float(np.linalg.det(J))
    J11, J22 = float(J[0, 0]), float(J[1, 1])
    ks = np.linspace(0.0, k_max, n_k)
    max_re = -1e9
    best_k = 0.0
    for k in ks:
        a0 = det + d1 * d2 * k**4 + k**2 * (d1 * J22 + d2 * J11)
        disc = tr**2 - 4.0 * a0
        if disc >= 0.0:
            mr = max((tr + np.sqrt(disc)) / 2.0, (tr - np.sqrt(disc)) / 2.0)
        else:
            mr = tr / 2.0
        if mr > max_re:
            max_re = mr
            best_k = k
    return max_re, best_k, tr, det


def scan_turing_d2(
    d2_values: np.ndarray,
    params: BDAFearParams | None = None,
    d1: float = 1.0,
    u_star: float | None = None,
    v_star: float | None = None,
) -> list[dict[str, float]]:
    """扫描 d2，返回各档 max Re λ(k) 及平衡点 Jacobian 信息。"""
    if params is None:
        params = bda_fear_default
    if u_star is None or v_star is None:
        u_star, v_star = resolve_bda_coexistence(params)

    rows: list[dict[str, float]] = []
    J = bd_fear_jacobian(u_star, v_star, params)
    for d2 in d2_values:
        max_re, k_best, tr, det = turing_max_real_eigenvalue(J, d1, float(d2))
        rows.append(
            {
                "d2": float(d2),
                "max_re_lambda": float(max_re),
                "k_at_max": float(k_best),
                "trace_J": tr,
                "det_J": det,
                "u_star": float(u_star),
                "v_star": float(v_star),
                "turing_window": float(max_re > 0.0 and tr < 0.0 and det > 0.0),
            }
        )
    return rows


def scan_d2_patterns(
    d2_values: np.ndarray | None = None,
    params: BDAFearParams | None = None,
    base_config: PDE2DConfig | None = None,
    progress: bool = True,
) -> list[PDE2DResult]:
    """扫描捕食者扩散系数 d2，复现 Figure 3–6 风格多档斑图。"""
    if d2_values is None:
        d2_values = np.array([0.02, 0.05, 0.08, 0.1])
    if base_config is None:
        base_config = PDE2DConfig()

    results: list[PDE2DResult] = []
    for i, d2 in enumerate(d2_values):
        if progress:
            print(f"  [d2={float(d2):.3g}] simulating...")
        cfg = replace(base_config, d2=float(d2))
        results.append(simulate_bda_fear_2d(params=params, config=cfg, progress=False))
    return results
