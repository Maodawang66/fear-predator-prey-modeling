"""
2D 有限差分 + Turing 线性稳定性诊断（Myint et al. 2025 拓展）。

用法:
    conda activate ai25
    python pde2d_turing.py              # 默认：稳定性曲线 + Fig.3/4 PDE
    python pde2d_turing.py --quick      # 64^2 网格、较短时间（调试）
    python pde2d_turing.py --fig all    # 输出 Fig.3–6 四组 PDE 图
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.parameters import BDAFearParams, bda_fear_default
from src.pde2d import (
    PDE2DConfig,
    resolve_bda_coexistence,
    scan_d2_patterns,
    scan_turing_d2,
    simulate_bda_fear_2d,
    turing_max_real_eigenvalue,
)
from src.k_damping import bd_fear_jacobian
from src.visualize import (
    plot_turing_d2_panel,
    plot_turing_snapshot,
    plot_turing_stability_d2,
    plot_turing_uv_pair,
)

# Myint 图注平衡点（仅作对照；默认参数下为鞍点，不作 PDE 中心）
PAPER_EQ_A01 = (0.16608, 3.89934)
PAPER_EQ_A055 = (0.16675, 3.91904)


def _equilibrium_for_params(params: BDAFearParams) -> tuple[float, float]:
    u, v = resolve_bda_coexistence(params)
    print(f"  numerical coexistence: u*={u:.5f}, v*={v:.5f}")
    return u, v


def _run_turing_diagnosis(
    params: BDAFearParams,
    out_dir: Path,
    d1: float = 1.0,
    tag: str = "default",
    paper_ref: tuple[float, float] | None = PAPER_EQ_A01,
) -> dict:
    u_star, v_star = _equilibrium_for_params(params)
    J = bd_fear_jacobian(u_star, v_star, params)
    tr, det = float(np.trace(J)), float(np.linalg.det(J))

    d2_dense = np.linspace(0.01, 0.25, 200)
    max_re_curve = np.array(
        [turing_max_real_eigenvalue(J, d1, d2)[0] for d2 in d2_dense]
    )
    d2_markers = np.array([0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.18, 0.2])
    rows = scan_turing_d2(d2_markers, params=params, d1=d1, u_star=u_star, v_star=v_star)

    plot_turing_stability_d2(
        rows,
        out_dir / f"turing_stability_d2_{tag}.png",
        d2_scan=d2_dense,
        max_re_curve=max_re_curve,
        u_star=u_star,
        v_star=v_star,
        paper_ref=paper_ref,
    )

    # 文献图注点对照（说明为何不宜直接用作 PDE 中心）
    if paper_ref is not None:
        J_paper = bd_fear_jacobian(paper_ref[0], paper_ref[1], params)
        tr_p, det_p = float(np.trace(J_paper)), float(np.linalg.det(J_paper))
        paper_note = {
            "u_star": paper_ref[0],
            "v_star": paper_ref[1],
            "trace_J": tr_p,
            "det_J": det_p,
            "note": "saddle if det<0",
        }
    else:
        paper_note = None

    summary = {
        "params": {
            "r": params.r,
            "d": params.d,
            "a": params.a,
            "k": params.k,
            "p": params.p,
            "q": params.q,
            "c": params.c,
            "m": params.m,
        },
        "numerical_coexistence": {"u_star": u_star, "v_star": v_star, "trace_J": tr, "det_J": det},
        "paper_figure_note_equilibrium": paper_note,
        "d1": d1,
        "scan_rows": rows,
        "d2_turing_upper_bound": float(
            d2_dense[np.where(max_re_curve > 0)[0][-1]]
        )
        if np.any(max_re_curve > 0)
        else None,
    }
    out_path = out_dir / f"turing_stability_{tag}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  -> Turing diagnosis: {out_path.name}, turing_stability_d2_{tag}.png")
    return summary


def _run_figure_set(
    name: str,
    params: BDAFearParams,
    d2_values: np.ndarray,
    t_end: float,
    base: PDE2DConfig,
    out_dir: Path,
    u_star: float,
    v_star: float,
) -> None:
    cfg = PDE2DConfig(
        nx=base.nx,
        ny=base.ny,
        lx=base.lx,
        ly=base.ly,
        d1=base.d1,
        t_end=t_end,
        perturbation=base.perturbation,
        u_star=u_star,
        v_star=v_star,
    )
    print(
        f"\n[{name}] a={params.a}, k={params.k}, "
        f"u*={u_star:.5f}, v*={v_star:.5f}, t_end={t_end}"
    )

    results = scan_d2_patterns(d2_values, params=params, base_config=cfg, progress=True)

    tag = f"{name}_t{int(t_end)}"
    plot_turing_d2_panel(
        results,
        out_dir / f"pde2d_{tag}_prey_panel.png",
        field="prey",
        suptitle=f"B-D+fear 2D PDE — prey u ({name}, t={t_end})",
    )
    plot_turing_d2_panel(
        results,
        out_dir / f"pde2d_{tag}_predator_panel.png",
        field="predator",
        suptitle=f"B-D+fear 2D PDE — predator v ({name}, t={t_end})",
    )

    mid = results[len(results) // 2]
    plot_turing_uv_pair(mid, out_dir / f"pde2d_{tag}_uv_d2_{mid.config.d2:g}.png")
    plot_turing_snapshot(
        mid,
        out_dir / f"pde2d_{tag}_prey_d2_{mid.config.d2:g}.png",
        field="prey",
    )
    print(f"  -> saved pde2d_{tag}_*.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="2D FD + Turing stability (Myint 2025 extension)")
    parser.add_argument(
        "--fig",
        choices=("3", "4", "5", "6", "34", "56", "all", "demo"),
        default="demo",
        help="PDE 图组：demo=Fig.3+4；all=四组",
    )
    parser.add_argument("--quick", action="store_true", help="64x64, t=50（快速试跑）")
    parser.add_argument("--nx", type=int, default=None)
    parser.add_argument("--skip-pde", action="store_true", help="仅输出 Turing 线性稳定性曲线")
    args = parser.parse_args()

    out_dir = ROOT / "results" / "pde2d"
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = args.nx or (64 if args.quick else 128)
    base = PDE2DConfig(nx=nx, ny=nx, d1=1.0, perturbation=0.01)

    print("=" * 60)
    print("2D 反应—扩散拓展 + Turing 线性稳定性诊断")
    print(f"grid={nx}x{nx}, domain=[0,{base.lx/np.pi:.1f}pi]^2, d1={base.d1}")
    print("=" * 60)

    print("\n[diagnosis] default B-D+fear parameters ...")
    diag_default = _run_turing_diagnosis(bda_fear_default, out_dir, tag="a01")

    sets: list[tuple[str, BDAFearParams, np.ndarray, float, tuple[float, float] | None]] = []

    if not args.skip_pde:
        if args.fig in ("demo", "3", "4", "34", "all"):
            t3 = 50.0 if args.quick else 50.0
            t4 = 50.0 if args.quick else 200.0
            d2_a = np.array([0.02, 0.05, 0.08, 0.1])
            p_a = bda_fear_default
            u_a, v_a = _equilibrium_for_params(p_a)
            if args.fig in ("demo", "34", "all"):
                sets.append(("fig3_a01", p_a, d2_a, t3, (u_a, v_a)))
                sets.append(("fig4_a01", p_a, d2_a, t4, (u_a, v_a)))
            elif args.fig == "3":
                sets.append(("fig3_a01", p_a, d2_a, t3, (u_a, v_a)))
            elif args.fig == "4":
                sets.append(("fig4_a01", p_a, d2_a, t4, (u_a, v_a)))

        if args.fig in ("5", "6", "56", "all"):
            t5 = 50.0 if args.quick else 50.0
            t6 = 50.0 if args.quick else 200.0
            d2_b = np.array([0.12, 0.15, 0.18, 0.2])
            p_b = BDAFearParams(a=0.055, k=bda_fear_default.k)
            print("\n[diagnosis] a=0.055 parameter set ...")
            _run_turing_diagnosis(p_b, out_dir, tag="a055", paper_ref=PAPER_EQ_A055)
            u_b, v_b = _equilibrium_for_params(p_b)
            if args.fig in ("56", "all"):
                sets.append(("fig5_a055", p_b, d2_b, t5, (u_b, v_b)))
                sets.append(("fig6_a055", p_b, d2_b, t6, (u_b, v_b)))
            elif args.fig == "5":
                sets.append(("fig5_a055", p_b, d2_b, t5, (u_b, v_b)))
            elif args.fig == "6":
                sets.append(("fig6_a055", p_b, d2_b, t6, (u_b, v_b)))

        for name, params, d2_vals, t_end, eq in sets:
            assert eq is not None
            _run_figure_set(name, params, d2_vals, t_end, base, out_dir, eq[0], eq[1])

        if args.fig == "demo" and not args.quick:
            u_s = diag_default["numerical_coexistence"]["u_star"]
            v_s = diag_default["numerical_coexistence"]["v_star"]
            print("\n[reference] single run d2=0.1, t=200")
            ref = simulate_bda_fear_2d(
                params=bda_fear_default,
                config=PDE2DConfig(
                    nx=nx, ny=nx, d1=1.0, d2=0.1, t_end=200.0, u_star=u_s, v_star=v_s
                ),
                progress=True,
            )
            plot_turing_uv_pair(ref, out_dir / "pde2d_reference_d2_0.1.png")

    print("\n" + "=" * 60)
    print(f"输出目录: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
