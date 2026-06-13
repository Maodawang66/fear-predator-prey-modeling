"""Generate a seven-model heatmap using the formal protocol documented in report.tex."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.formal_series import LEGACY_FORMAL_IDS, load_formal_series  # noqa: E402
from src.fear_pathway_fit import fit_holling_fear_pathway_to_series  # noqa: E402
from src.parameters import BaselineParams  # noqa: E402

OUT = ROOT / "results" / "seven_model_real_fits"
FORMAL_SUMMARY = ROOT / "results" / "calibration_bda" / "fit_summary.csv"
FORMAL_MODELS = ("baseline", "fear_memory", "bda_fear")
EXTENDED_MODELS = ("fear_instant", "fear_saturating", "fear_foraging", "fear_handling")
MODEL_ORDER = (
    "baseline",
    "fear_instant",
    "fear_memory",
    "fear_saturating",
    "fear_foraging",
    "fear_handling",
    "bda_fear",
)
LABELS = {
    "baseline": "Baseline",
    "fear_instant": "Instant reproduction fear",
    "fear_memory": "Memory reproduction fear",
    "fear_saturating": "Saturating reproduction fear",
    "fear_foraging": "Foraging/attack suppression",
    "fear_handling": "Handling-time extension",
    "bda_fear": "B-D + fear",
}


def _formal_rows() -> dict[tuple[str, str], dict]:
    rows: dict[tuple[str, str], dict] = {}
    with FORMAL_SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] in FORMAL_MODELS:
                rows[(row["series"], row["model"])] = {
                    "series": row["series"],
                    "model": row["model"],
                    "training_rmse": float(row["rmse_normalized_total"]),
                    "validation_rmse": float(row["validation_rmse_normalized_total"]),
                    "aicc": float(row["aicc"]),
                    "optimization_status": row["optimization_status"],
                    "usable_for_comparison": row["usable_for_comparison"] == "True",
                    "fear_strength": row.get("phi", "") if row["model"] == "fear_memory" else "",
                    "fear_parameter": "phi" if row["model"] == "fear_memory" else "",
                    "fear_parameter_bounds": row.get("fear_parameter_bounds", ""),
                    "optimized_fear_objective": row.get("optimized_fear_objective", ""),
                    "nested_baseline_candidate_objective": row.get(
                        "nested_baseline_candidate_objective", ""
                    ),
                    "nested_baseline_candidate_selected": (
                        row.get("nested_baseline_candidate_selected", "") == "True"
                        if row.get("nested_baseline_candidate_selected", "")
                        else ""
                    ),
                    "source": "formal_report_fit_summary",
                }
    return rows


def _formal_baseline_candidates() -> dict[
    str, tuple[BaselineParams, tuple[str, ...], str, str]
]:
    candidates: dict[str, tuple[BaselineParams, tuple[str, ...], str, str]] = {}
    with FORMAL_SUMMARY.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != "baseline":
                continue
            candidates[row["series"]] = (
                BaselineParams(**{
                    name: float(row[name])
                    for name in ("r", "K", "a", "theta", "e", "mu")
                }),
                tuple(
                    hit.strip()
                    for hit in row.get("parameter_bound_hits", "").split(";")
                    if hit.strip()
                ),
                row["optimization_status"],
                row.get("termination_reason", "nested baseline candidate"),
            )
    return candidates


def _cache_path(model: str) -> Path:
    return OUT / f"report_protocol_{model}_metrics.csv"


def _load_cached_model(model: str) -> list[dict]:
    path = _cache_path(model)
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["validation_rmse"] = float(row["validation_rmse"])
        row["aicc"] = float(row["aicc"])
        row["usable_for_comparison"] = row["usable_for_comparison"] == "True"
    return [row for row in rows if row["series"] != "lynxhare"]


def _write_model_cache(model: str, rows: list[dict]) -> None:
    with _cache_path(model).open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fit_extended_model(
    series_list,
    model: str,
    refit_series: set[str] | None = None,
    refit_model_cache: bool = False,
) -> list[dict]:
    refit_series = refit_series or set()
    rows = [] if refit_model_cache else [
        row for row in _load_cached_model(model) if row["series"] not in refit_series
    ]
    completed = {row["series"] for row in rows}
    missing_legacy = sorted(set(LEGACY_FORMAL_IDS) - completed - refit_series)
    if missing_legacy and not refit_model_cache:
        raise RuntimeError(
            f"{model} legacy cache incomplete; refusing to refit: {missing_legacy}"
        )
    baseline_candidates = _formal_baseline_candidates()
    for index, series in enumerate(series_list, start=1):
        if series.name in completed:
            continue
        print(f"[{model} {index}/{len(series_list)}] {series.name}", flush=True)
        (
            baseline_params,
            baseline_bound_hits,
            baseline_status,
            baseline_message,
        ) = baseline_candidates[series.name]
        result = fit_holling_fear_pathway_to_series(
            series,
            model,
            baseline_params=baseline_params,
            baseline_parameter_bound_hits=baseline_bound_hits,
            baseline_candidate_status=baseline_status,
            baseline_candidate_message=baseline_message,
            optimizer="global",
            optimizer_seeds=(0, 1, 2),
            max_nfev=500,
        )
        rows.append({
            "series": series.name,
            "model": model,
            "training_rmse": result.rmse_normalized_total,
            "validation_rmse": result.validation_rmse_normalized_total,
            "aicc": result.aicc,
            "optimization_status": result.optimization_status,
            "usable_for_comparison": result.usable_for_comparison,
            "fear_strength": result.params[result.meta["fear_parameter"]],
            "fear_parameter": result.meta["fear_parameter"],
            "fear_parameter_bounds": result.meta["fear_parameter_bounds"],
            "optimized_fear_objective": result.meta["optimized_fear_objective"],
            "nested_baseline_candidate_objective": result.meta["nested_baseline_candidate_objective"],
            "nested_baseline_candidate_selected": result.meta["nested_baseline_candidate_selected"],
            "source": "report_protocol_extension_multiseed_global_local",
        })
        _write_model_cache(model, rows)
    return rows


def _write_rows(rows: list[dict], filename: str = "report_protocol_seven_model_metrics.csv") -> None:
    with (OUT / filename).open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(
    rows: list[dict],
    model_order: tuple[str, ...],
    filenames: tuple[str, ...],
    title: str,
) -> None:
    series_names = list(dict.fromkeys(row["series"] for row in rows))
    lookup = {(row["series"], row["model"]): row for row in rows}
    values = np.full((len(series_names), len(model_order)), np.nan)
    for i, series in enumerate(series_names):
        for j, model in enumerate(model_order):
            row = lookup[(series, model)]
            if row["usable_for_comparison"]:
                values[i, j] = float(row["validation_rmse"])
    row_min = np.nanmin(values, axis=1, keepdims=True)
    ratios = values / np.maximum(row_min, np.finfo(float).eps)
    masked = np.ma.masked_invalid(np.log10(ratios))

    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#d9d9d9")
    image = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
    for i, series in enumerate(series_names):
        for j, model in enumerate(model_order):
            row = lookup[(series, model)]
            text = f"{float(row['validation_rmse']):.3g}"
            if not row["usable_for_comparison"]:
                text = "failed"
            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=7.5,
                color="black" if not row["usable_for_comparison"] else "white",
            )
    ax.set_xticks(
        range(len(model_order)),
        [LABELS[model] for model in model_order],
        rotation=35,
        ha="right",
    )
    ax.set_yticks(range(len(series_names)), series_names)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label="log10(RMSE / best usable RMSE in series)")
    fig.tight_layout()
    for filename in filenames:
        fig.savefig(OUT / filename, dpi=190)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=EXTENDED_MODELS)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--refit-model-cache",
        action="store_true",
        help="discard the selected extended model cache and refit every formal series",
    )
    parser.add_argument(
        "--refit-series",
        action="append",
        default=[],
        help="discard cached extended-model metrics for this formal series",
    )
    args = parser.parse_args()
    refit_series = set(args.refit_series)
    OUT.mkdir(parents=True, exist_ok=True)
    series_list = load_formal_series()
    if args.model:
        _fit_extended_model(
            series_list,
            args.model,
            refit_series,
            refit_model_cache=args.refit_model_cache,
        )
        return
    if args.refit_model_cache:
        raise ValueError("--refit-model-cache requires --model")
    formal = _formal_rows()
    extended = [
        row
        for model in EXTENDED_MODELS
        for row in (
            _load_cached_model(model)
            if args.aggregate_only
            else _fit_extended_model(series_list, model, refit_series)
        )
    ]
    extended = [
        row for row in extended
        if row["series"] in {series.name for series in series_list}
    ]
    if len(extended) != len(series_list) * len(EXTENDED_MODELS):
        raise RuntimeError("extended model caches are incomplete")
    extended_lookup = {(row["series"], row["model"]): row for row in extended}
    rows = []
    for series in series_list:
        for model in MODEL_ORDER:
            rows.append(
                formal[(series.name, model)]
                if model in FORMAL_MODELS
                else extended_lookup[(series.name, model)]
            )
    _write_rows(rows)
    protocol_suffix = "Report protocol: train-only normalization; seeds 0,1,2 global search + local refinement"
    _plot(
        rows,
        MODEL_ORDER,
        ("report_protocol_seven_model_validation_heatmap.png", "validation_rmse_heatmap.png"),
        f"Seven-model 20% continuous multi-step holdout RMSE\n{protocol_suffix}",
    )
    six_model_order = tuple(model for model in MODEL_ORDER if model != "bda_fear")
    six_rows = [row for row in rows if row["model"] in six_model_order]
    _write_rows(six_rows, "report_protocol_six_model_metrics.csv")
    _plot(
        six_rows,
        six_model_order,
        (
            "report_protocol_six_model_validation_heatmap.png",
            "validation_rmse_heatmap_six_model.png",
        ),
        f"Six-model 20% continuous multi-step holdout RMSE\n{protocol_suffix}",
    )


if __name__ == "__main__":
    main()
