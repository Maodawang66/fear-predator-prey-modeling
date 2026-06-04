"""
恐惧参数 k 削弱振荡 — 三种科学检验一键运行。

用法（项目根目录）:
    conda activate ai25
    python k_damping_analysis.py

输出: results/k_damping/
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.k_damping import (
    k_damping_summary,
    scan_k_damping,
    scan_k_eigenvalues,
    simulate_bda_at_k,
)
from src.parameters import bda_fear_default
from src.visualize import (
    plot_k_amplitude_scan,
    plot_k_eigenvalue_scan,
    plot_k_peak_decay,
    plot_k_phase_shrink,
    plot_k_timeseries_panel,
)

OUT = ROOT / "results" / "k_damping"


def _write_scan_csv(rows, path: Path) -> None:
    fields = [
        "k", "u_mean", "v_mean", "amplitude_u", "amplitude_v",
        "rel_amplitude_u", "peak_decay_ratio", "n_peaks",
        "re_max", "stability", "status",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def _write_report_md(summary: dict, path: Path) -> None:
    lines = [
        "# k 增大削弱振荡：检验报告",
        "",
        "## 文献依据",
    ]
    for ref in summary["literature"]:
        lines.append(f"- {ref}")
    lines.extend(["", "## 本项目采用的方法"])
    for i, m in enumerate(summary["methods"], 1):
        lines.append(f"{i}. {m}")
    lines.extend([
        "",
        "## 数值摘要",
        f"- Hopf 阈值估计 k_H ≈ {summary['hopf_k_estimate']}",
        f"- k=0: Re λ_max={summary['k0']['re_max']}, "
        f"相对振幅={summary['k0']['rel_amplitude_u']:.4g}, "
        f"稳定性={summary['k0']['stability']}",
        f"- 扫描最大 k={summary['k_max_scanned']['k']}: "
        f"相对振幅={summary['k_max_scanned']['rel_amplitude_u']:.4g}, "
        f"Re λ_max={summary['k_max_scanned']['re_max']}",
        f"- 最强压缩出现在 k={summary['strongest_damping_at']['k']}, "
        f"相对振幅={summary['strongest_damping_at']['rel_amplitude_u']:.4g}",
        "",
        "## 论文写法提示",
        summary["conclusion_hint"],
        "",
        "图件见本目录 PNG；完整表格见 `k_scan.csv`。",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("k 削弱振荡：Jacobian + 振幅 + 峰值衰减")
    print("=" * 60)

    k_grid = np.linspace(0.0, 0.18, 37)
    eigen_rows = scan_k_eigenvalues(k_grid, base=bda_fear_default)
    scan_rows = scan_k_damping(k_grid, base=bda_fear_default, t_end=150.0)

    summary = k_damping_summary(scan_rows, eigen_rows)

    plot_k_eigenvalue_scan(eigen_rows, OUT / "01_eigenvalue_vs_k.png")
    plot_k_amplitude_scan(scan_rows, OUT / "02_amplitude_vs_k.png")
    plot_k_peak_decay(scan_rows, OUT / "03_peak_decay_vs_k.png")

    k_show = [0.0, 0.04, 0.08, 0.14]
    sols = {
        f"k={kv:g}": simulate_bda_at_k(kv, t_end=150.0)
        for kv in k_show
    }
    plot_k_timeseries_panel(sols, OUT / "04_timeseries_multi_k.png")
    plot_k_phase_shrink(sols, OUT / "05_phase_multi_k.png")

    _write_scan_csv(scan_rows, OUT / "k_scan.csv")
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_report_md(summary, OUT / "report.md")

    print(f"\nHopf 阈值估计 k_H ≈ {summary['hopf_k_estimate']}")
    print(f"k=0  相对振幅={summary['k0']['rel_amplitude_u']:.4g}, "
          f"Re λ={summary['k0']['re_max']}")
    print(f"k↑   相对振幅={summary['k_max_scanned']['rel_amplitude_u']:.4g}, "
          f"Re λ={summary['k_max_scanned']['re_max']}")
    print(f"\n输出: {OUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
