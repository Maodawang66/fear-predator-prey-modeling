"""
恐惧效应捕食者-猎物模型：一键运行数值实验并输出图表。

用法: python main.py
建议环境: conda activate ai25
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# 无 GUI 后端，便于 Windows 批处理运行
os.environ.setdefault("MPLBACKEND", "Agg")

from src.analysis import (
    compare_mechanisms,
    equilibrium_baseline,
    equilibrium_bda_fear,
    nce_vs_consumptive_summary,
    scan_bda_fear_k,
    scan_delta,
    scan_phi,
    sensitivity_local,
)
from src.model import bd_fear_rhs
from src.parameters import bda_fear_default, bda_no_fear_default
from src.simulate import integrate_rhs
from src.literature import mechanism_labels, run_all_mechanisms
from src.parameters import baseline_default, fear_default, fear_high
from src.simulate import integrate_baseline, integrate_fear_memory
from src.visualize import (
    plot_amplitude_comparison,
    plot_delta_scan,
    plot_mechanism_bars,
    plot_mechanism_comparison,
    plot_phase_plane,
    plot_phi_scan,
    plot_sensitivity,
    plot_three_scenarios,
    plot_timeseries_compare,
)


def main() -> None:
    root = Path(__file__).resolve().parent
    out_dir = root / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("恐惧效应捕食者-猎物动力学 — 数值实验")
    print("=" * 60)

    eq = equilibrium_baseline(baseline_default)
    print(f"\n[基线正平衡点] x*={eq['x_star']}, y*={eq['y_star']}")

    # --- 主模型实验 ---
    sol_base = integrate_baseline(baseline_default, t_span=(0.0, 80.0))
    sol_fear = integrate_fear_memory(fear_default, t_span=(0.0, 80.0))
    plot_timeseries_compare(
        sol_base,
        sol_fear,
        out_dir / "01_timeseries_baseline_vs_fear.png",
    )
    print("\n[1] 01_timeseries_baseline_vs_fear.png")

    plot_phase_plane(sol_base, out_dir / "02_phase_baseline.png", "Baseline model phase plane")
    plot_phase_plane(sol_fear, out_dir / "03_phase_fear.png", "Fear + memory model phase plane")
    print("[2-3] 02_phase_baseline.png, 03_phase_fear.png")

    sol_high = integrate_fear_memory(fear_high, t_span=(0.0, 80.0))
    plot_three_scenarios(
        [
            ("baseline (φ=0)", sol_base),
            ("fear φ=0.02", sol_fear),
            ("strong fear φ=0.045", sol_high),
        ],
        out_dir / "04_three_scenarios.png",
    )
    print("[4] 04_three_scenarios.png")

    phi_data = scan_phi()
    plot_phi_scan(phi_data, out_dir / "05_phi_scan.png")
    extinct = [s for s in phi_data["status"] if "extinct" in str(s)]
    print(f"[5] φ 扫描: 灭绝类情景 {len(extinct)}/{len(phi_data['phi'])} → 05_phi_scan.png")

    delta_data = scan_delta(phi=0.03)
    plot_delta_scan(delta_data, out_dir / "06_delta_scan.png")
    print("[6] 06_delta_scan.png")

    sens = sensitivity_local(fear_default)
    plot_sensitivity(sens, out_dir / "07_sensitivity.png")
    print("\n[敏感性] 平均猎物密度:")
    for k, v in sens.items():
        print(f"  {k}: {v:+.4f}")
    print("    → 07_sensitivity.png")

    # --- 文献机制对照实验 ---
    print("\n--- 文献机制对照 ---")
    all_sols = run_all_mechanisms(t_span=(0.0, 80.0))
    labels = mechanism_labels()
    plot_mechanism_comparison(
        all_sols, labels, out_dir / "08_literature_mechanisms_timeseries.png"
    )
    print("[8] 08_literature_mechanisms_timeseries.png")

    cmp = compare_mechanisms(t_end=100.0)
    plot_mechanism_bars(cmp, out_dir / "09_literature_mechanisms_bars.png")
    plot_amplitude_comparison(cmp, out_dir / "10_literature_oscillation_amplitude.png")
    print("[9-10] 09_literature_mechanisms_bars.png, 10_literature_oscillation_amplitude.png")

    nce = nce_vs_consumptive_summary()
    print(
        f"\n[NCE 代理] 基线猎物均值={nce['prey_mean_baseline']:.2f}, "
        f"恐惧模型={nce['prey_mean_fear']:.2f}, "
        f"相对下降={nce['relative_reduction_pct']:.1f}%"
    )

    print("\n机制对照摘要:")
    for lab, xm, ym, st in zip(
        cmp["label"], cmp["x_mean"], cmp["y_mean"], cmp["status"]
    ):
        print(f"  {lab}: x_mean={xm:.2f}, y_mean={ym:.2f}, {st}")

    # --- Myint et al. (2025) B-D + 恐惧模型 ---
    print("\n--- Myint et al. (2025) B-D + 恐惧 ---")
    eq_bda = equilibrium_bda_fear(bda_fear_default)
    print(f"[B-D 正平衡点] u*={eq_bda.get('u_star')}, v*={eq_bda.get('v_star')}, λ={eq_bda.get('lambda')}")
    y0 = np.array([0.17, 3.9])
    sol_bda0 = integrate_rhs(
        lambda t, s: bd_fear_rhs(t, s, bda_no_fear_default), y0, t_span=(0.0, 80.0)
    )
    sol_bda = integrate_rhs(
        lambda t, s: bd_fear_rhs(t, s, bda_fear_default), y0, t_span=(0.0, 80.0)
    )
    plot_timeseries_compare(
        sol_bda0,
        sol_bda,
        out_dir / "11_bda_fear_k0_vs_k.png",
        title="Myint (2025): k=0 vs k>0 (B-D + fear)",
    )
    k_scan = scan_bda_fear_k()
    plot_phi_scan(
        {"phi": k_scan["k"], "x_mean": k_scan["u_mean"], "y_mean": k_scan["v_mean"]},
        out_dir / "12_bda_fear_k_scan.png",
        x_key="phi",
        xlabel="fear parameter k",
        ylabel="long-run mean density",
        title="k parameter scan (B-D + fear)",
        prey_label="prey u mean",
        pred_label="predator v mean",
    )
    print("[11-12] 11_bda_fear_k0_vs_k.png, 12_bda_fear_k_scan.png")

    # --- k 削弱振荡：三种科学检验 ---
    print("\n--- k 削弱振荡（Jacobian / 振幅 / 峰值衰减）---")
    from src.k_damping import k_damping_summary, scan_k_damping, scan_k_eigenvalues, simulate_bda_at_k
    from src.visualize import (
        plot_k_amplitude_scan,
        plot_k_eigenvalue_scan,
        plot_k_peak_decay,
        plot_k_phase_shrink,
        plot_k_timeseries_panel,
    )

    k_dir = out_dir / "k_damping"
    k_dir.mkdir(exist_ok=True)
    k_grid = np.linspace(0.0, 0.18, 25)
    eigen_rows = scan_k_eigenvalues(k_grid, base=bda_fear_default)
    damp_rows = scan_k_damping(k_grid, base=bda_fear_default, t_end=120.0)
    kd_sum = k_damping_summary(damp_rows, eigen_rows)
    plot_k_eigenvalue_scan(eigen_rows, k_dir / "01_eigenvalue_vs_k.png")
    plot_k_amplitude_scan(damp_rows, k_dir / "02_amplitude_vs_k.png")
    plot_k_peak_decay(damp_rows, k_dir / "03_peak_decay_vs_k.png")
    sols_k = {f"k={kv:g}": simulate_bda_at_k(kv, t_end=120.0) for kv in (0.0, 0.04, 0.08, 0.14)}
    plot_k_timeseries_panel(sols_k, k_dir / "04_timeseries_multi_k.png")
    plot_k_phase_shrink(sols_k, k_dir / "05_phase_multi_k.png")
    print(f"  Hopf k≈{kd_sum['hopf_k_estimate']}, "
          f"k=0 相对振幅={kd_sum['k0']['rel_amplitude_u']:.3g}, "
          f"k_max 相对振幅={kd_sum['k_max_scanned']['rel_amplitude_u']:.3g}")
    print(f"  → results/k_damping/ （或运行 python k_damping_analysis.py 完整版）")

    print("\n" + "=" * 60)
    print(f"全部结果: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
