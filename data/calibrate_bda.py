"""
自动发现 data/raw 中的捕食者—猎物 CSV → 拟合 ODE（基线 / 恐惧记忆 / B-D+恐惧）→ 出图与参数表。

无需手动指定文件或列名；识别逻辑见 data/auto_discover.py。

用法（项目根目录）:
    conda activate ai25
    python data/calibrate_bda.py
    python data/calibrate_bda.py --min-confidence 0.6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.auto_discover import discover_all_candidates, discover_and_load, discover_csv_paths  # noqa: E402
from data.common import is_valid_csv  # noqa: E402
from src.fit import FitResult, fit_baseline_to_series, fit_bda_fear_to_series, fit_fear_memory_to_series  # noqa: E402
from src.visualize import plot_fit_result  # noqa: E402

OUT = ROOT / "results" / "calibration_bda"


def _save_json(result: FitResult, path: Path) -> None:
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
        "validation_rmse_normalized_prey": result.validation_rmse_normalized_prey,
        "validation_rmse_normalized_predator": result.validation_rmse_normalized_predator,
        "validation_rmse_normalized_total": result.validation_rmse_normalized_total,
        "validation_rmse_raw_prey": result.validation_rmse_raw_prey,
        "validation_rmse_raw_predator": result.validation_rmse_raw_predator,
        "validation_rmse_raw_total": result.validation_rmse_raw_total,
        "aic": result.aic,
        "aicc": result.aicc,
        "bic": result.bic,
        "n_parameters": result.n_parameters,
        "n_train_points": result.n_train_points,
        "n_validation_points": result.n_validation_points,
        "success": result.success,
        "optimization_status": result.optimization_status,
        "usable_for_comparison": result.usable_for_comparison,
        "message": result.message,
        "meta": result.meta,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _fit_one_series(series, tag: str) -> list[FitResult]:
    print(f"\n--- {series.name} ({series.n_points} pts) ---")
    print(
        f"    source: {Path(series.source_path).name} | "
        f"prey={series.prey_label} | pred={series.predator_label}"
    )
    if series.meta.get("detection_method"):
        print(
            f"    detect: {series.meta.get('detection_method')} "
            f"(conf={series.meta.get('confidence', 0):.2f})"
        )

    results: list[FitResult] = []
    for fitter, model_name in (
        (fit_baseline_to_series, "baseline"),
        (fit_fear_memory_to_series, "fear_memory"),
        (fit_bda_fear_to_series, "bda_fear"),
    ):
        try:
            res = fitter(series)
            res.meta = {**res.meta, **series.meta, "source_file": series.source_path}
            results.append(res)
        except Exception as exc:
            print(f"  [{model_name}] FAIL: {exc}")
            continue

        stem = f"{tag}_{model_name}"
        plot_fit_result(res, OUT / "figures" / f"{stem}.png")
        _save_json(res, OUT / "params" / f"{stem}.json")
        print(
            f"  [{model_name}] train_RMSE={res.rmse_total:.4g}  "
            f"validation_RMSE={res.validation_rmse_normalized_total:.4g}  "
            f"AICc={res.aicc:.4g}  "
            f"status={res.optimization_status}"
        )
        if model_name == "fear_memory":
            print(f"    phi={res.params.get('phi', 0):.5f}")
        if model_name == "bda_fear":
            print(
                f"    k={res.params.get('k', 0):.5f}, "
                f"p={res.params.get('p', 0):.3f}, q={res.params.get('q', 0):.3f}"
            )
    return results


def _write_discovery_report(candidates, scanned: list[Path]) -> None:
    lines = [
        "# Auto-discovery report",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        f"scanned_valid_csv: {len(scanned)}",
        f"candidates: {len(candidates)}",
        "",
    ]
    for path in scanned:
        status = "OK" if is_valid_csv(path) else "SKIP"
        lines.append(f"- [{status}] {path.relative_to(ROOT)}")
    lines.append("")
    lines.append("## Detected predator-prey series")
    for c in candidates:
        m = c.mapping
        lines.append(
            f"- **{c.path.name}** ({c.signature}) "
            f"conf={m.confidence:.2f} method={m.method}"
        )
        lines.append(
            f"  - time=`{m.time_col}` prey=`{m.prey_col or c.prey_label}` "
            f"pred=`{m.predator_col or c.predator_label}` "
            f"group={c.group_key or '-'}"
        )
        if c.notes:
            lines.append(f"  - {c.notes}")
    (OUT / "discovery_report.md").write_text("\n".join(lines), encoding="utf-8")


def _write_summary_table(all_results: list[FitResult]) -> None:
    if not all_results:
        return
    path = OUT / "fit_summary.csv"
    param_keys = sorted({k for r in all_results for k in r.params})
    meta_keys = [
        "source_file",
        "detection_method",
        "confidence",
        "time_col",
        "prey_col",
        "predator_col",
        "group_key",
    ]
    fieldnames = (
        [
            "series", "model",
            "rmse_normalized_total", "rmse_normalized_prey", "rmse_normalized_predator",
            "rmse_raw_total", "rmse_raw_prey", "rmse_raw_predator",
            "validation_rmse_normalized_total",
            "validation_rmse_normalized_prey",
            "validation_rmse_normalized_predator",
            "validation_rmse_raw_total",
            "validation_rmse_raw_prey",
            "validation_rmse_raw_predator",
            "aic", "aicc", "bic",
            "n_parameters", "n_train_points", "n_validation_points",
            "optimization_status", "usable_for_comparison", "success",
            "termination_reason", "objective_value", "parameter_bound_hits",
        ]
        + meta_keys
        + param_keys
    )
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in all_results:
            row = {
                "series": r.series_name,
                "model": r.model,
                "rmse_normalized_total": r.rmse_normalized_total,
                "rmse_normalized_prey": r.rmse_normalized_prey,
                "rmse_normalized_predator": r.rmse_normalized_predator,
                "rmse_raw_total": r.rmse_raw_total,
                "rmse_raw_prey": r.rmse_raw_prey,
                "rmse_raw_predator": r.rmse_raw_predator,
                "validation_rmse_normalized_total": r.validation_rmse_normalized_total,
                "validation_rmse_normalized_prey": r.validation_rmse_normalized_prey,
                "validation_rmse_normalized_predator": r.validation_rmse_normalized_predator,
                "validation_rmse_raw_total": r.validation_rmse_raw_total,
                "validation_rmse_raw_prey": r.validation_rmse_raw_prey,
                "validation_rmse_raw_predator": r.validation_rmse_raw_predator,
                "aic": r.aic,
                "aicc": r.aicc,
                "bic": r.bic,
                "n_parameters": r.n_parameters,
                "n_train_points": r.n_train_points,
                "n_validation_points": r.n_validation_points,
                "optimization_status": r.optimization_status,
                "usable_for_comparison": r.usable_for_comparison,
                "success": r.success,
                "termination_reason": r.meta.get("termination_reason", r.message),
                "objective_value": r.meta.get("objective_value", ""),
                "parameter_bound_hits": ";".join(r.meta.get("parameter_bound_hits", [])),
            }
            for mk in meta_keys:
                row[mk] = r.meta.get(mk, "")
            row.update(r.params)
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-discover CSV and calibrate B-D+ fear ODE")
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--max-series", type=int, default=12, help="最多拟合序列数（防止区域×站点爆炸）")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "figures").mkdir(exist_ok=True)
    (OUT / "params").mkdir(exist_ok=True)

    print("=" * 60)
    print("自动标定：发现 CSV -> 识别列 -> 拟合 ODE -> 参数表")
    print("=" * 60)

    scanned = discover_csv_paths()
    print(f"\n[扫描] 有效 CSV/TXT: {len(scanned)}")
    for p in scanned:
        print(f"  - {p.relative_to(ROOT)}")

    candidates = discover_all_candidates()
    _write_discovery_report(candidates, scanned)

    series_list = discover_and_load(min_confidence=args.min_confidence)
    series_list.sort(
        key=lambda s: (s.meta.get("confidence", 0), s.n_points),
        reverse=True,
    )
    series_list = series_list[: args.max_series]

    print(f"\n[加载] 可拟合序列: {len(series_list)}")
    for s in series_list:
        print(f"  - {s.name} ({s.n_points} pts) <- {Path(s.source_path).name}")

    all_results: list[FitResult] = []
    log: list[str] = []
    for i, series in enumerate(series_list):
        tag = f"{i+1:02d}_{series.name}"[:48]
        results = _fit_one_series(series, tag)
        all_results.extend(results)
        for r in results:
            log.append(
                f"{r.optimization_status.upper()} {tag}_{r.model} "
                f"train_RMSE={r.rmse_total:.4g} "
                f"validation_RMSE={r.validation_rmse_normalized_total:.4g} "
                f"AICc={r.aicc:.4g} reason={r.message.strip()}"
            )

    _write_summary_table(all_results)
    (OUT / "calibration_log.txt").write_text(
        "\n".join(log) + f"\n\nseries={len(series_list)} fits={len(all_results)}\n",
        encoding="utf-8",
    )

    id_report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "scanned_files": [str(p.relative_to(ROOT)) for p in scanned],
        "candidates": [
            {
                "file": str(c.path.relative_to(ROOT)),
                "signature": c.signature,
                "confidence": c.mapping.confidence,
                "method": c.mapping.method,
                "time_col": c.mapping.time_col,
                "prey_col": c.mapping.prey_col or c.prey_label,
                "predator_col": c.mapping.predator_col or c.predator_label,
                "group_key": c.group_key,
                "notes": c.notes,
            }
            for c in candidates
        ],
        "loaded_series": [
            {
                "name": s.name,
                "n_points": s.n_points,
                "source": s.source_path,
                "meta": s.meta,
            }
            for s in series_list
        ],
    }
    (OUT / "identification_report.json").write_text(
        json.dumps(id_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print(f"完成: {len(series_list)} 条序列, {len(all_results)} 次拟合")
    print(f"参数表: {OUT / 'fit_summary.csv'}")
    print(f"识别报告: {OUT / 'discovery_report.md'}")
    print(f"JSON: {OUT / 'identification_report.json'}")
    print(f"图表: {OUT / 'figures'}")
    if not series_list:
        print(
            "\n提示: Dryad 文件若为 HTML，请浏览器下载后放入 data/bundled/ 同名路径后重试。"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
