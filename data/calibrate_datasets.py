"""
用真实 CSV 标定 ODE 参数并输出拟合图。

用法（项目根目录）:
    conda activate ai25
    python data/calibrate_datasets.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.common import InvalidDataFileError  # noqa: E402
from data.load_killifish import load_killifish  # noqa: E402
from data.load_lter_fish import load_lter_fish_pair  # noqa: E402
from data.load_lynx_hare import load_lynx_hare  # noqa: E402
from data.load_lynx_roe import load_lynx_roe  # noqa: E402
from data.load_zooplankton import load_zooplankton  # noqa: E402
from src.fit import (  # noqa: E402
    FitResult,
    fit_baseline_to_series,
    fit_bda_fear_to_series,
    fit_fear_memory_to_series,
)
from src.visualize import plot_fit_result  # noqa: E402


OUT = ROOT / "results" / "calibration"


def _save_params(result: FitResult, path: Path) -> None:
    payload = {
        "model": result.model,
        "series": result.series_name,
        "params": result.params,
        "rmse_normalized_prey": result.rmse_normalized_prey,
        "rmse_normalized_predator": result.rmse_normalized_predator,
        "rmse_normalized_total": result.rmse_normalized_total,
        "rmse_raw_prey": result.rmse_raw_prey,
        "rmse_raw_predator": result.rmse_raw_predator,
        "rmse_raw_total": result.rmse_raw_total,
        "success": result.success,
        "optimization_status": result.optimization_status,
        "usable_for_comparison": result.usable_for_comparison,
        "message": result.message,
        "meta": result.meta,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _run_one(series, tag: str, log: list[str]) -> None:
    print(f"\n--- {series.name} ({series.n_points} 点) ---")
    base = fit_baseline_to_series(series)
    fear = fit_fear_memory_to_series(series)
    bda = fit_bda_fear_to_series(series, fit_k=True)

    for res in (base, fear, bda):
        stem = f"{tag}_{res.model}"
        plot_fit_result(res, OUT / f"{stem}.png")
        _save_params(res, OUT / f"{stem}.json")
        log.append(
            f"{res.optimization_status.upper()} {stem}: normalized_RMSE={res.rmse_total:.4g} "
            f"(prey={res.rmse_prey:.4g}, pred={res.rmse_predator:.4g})"
        )
        print(
            f"  [{res.model}] normalized_RMSE={res.rmse_total:.4g}  "
            f"status={res.optimization_status}"
        )
        if res.model == "fear_memory":
            print(f"    phi={res.params.get('phi', 0):.5f}")
        if res.model == "bda_fear":
            print(f"    k={res.params.get('k', 0):.5f}, p={res.params.get('p', 0):.3f}, q={res.params.get('q', 0):.3f}")


def _try_load(label: str, loader, log: list[str]):
    try:
        series = loader()
        return series
    except (FileNotFoundError, InvalidDataFileError, ValueError) as e:
        msg = f"SKIP {label}: {e}"
        log.append(msg)
        print(msg)
        return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    log: list[str] = []

    print("=" * 60)
    print("真实数据 → ODE 拟合")
    print("=" * 60)

    jobs = [
        ("hudson_bay", lambda: load_lynx_hare()),
        ("lynx_roe_r3", lambda: load_lynx_roe(region=3)),
        ("killifish_tp", lambda: load_killifish(site="TP")),
        ("zooplankton", load_zooplankton),
        ("lter_fish", lambda: load_lter_fish_pair()),
    ]

    ran = 0
    for tag, loader in jobs:
        series = _try_load(tag, loader, log)
        if series is None:
            continue
        _run_one(series, tag, log)
        ran += 1

    summary = OUT / "calibration_log.txt"
    summary.write_text("\n".join(log) + f"\n\ncompleted={ran}/{len(jobs)}\n", encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"完成 {ran}/{len(jobs)} 组数据集拟合")
    print(f"结果: {OUT}")
    if ran == 0:
        print(
            "\n提示: Dryad CSV 若被反爬替换为 HTML，请浏览器下载后放入 data/bundled/ 同名路径。"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
