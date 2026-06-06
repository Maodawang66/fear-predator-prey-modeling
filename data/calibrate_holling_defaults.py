"""
Use the 12 fitted population time series to choose stable Holling II defaults.

The script prints a global empirical coexistence target and searches constrained
BaselineParams(a, theta, e, mu) values whose no-fear equilibrium is positive,
locally stable, and numerically persistent.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.auto_discover import discover_and_load  # noqa: E402
from data.series import PredatorPreySeries  # noqa: E402
from src.parameters import BaselineParams  # noqa: E402
from src.simulate import integrate_baseline, integrate_until_converged, is_extinct  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "holling_report_series_manifest.json"
OUT = ROOT / "results" / "holling_defaults"
DIAGNOSTIC_CONFIG = {
    "primary_scaling": "robust_q95",
    "tail_frac": 0.5,
    "tail_window_frac": 0.25,
    "tail_mean_change_tolerance": 0.20,
    "trend_correlation_threshold": 0.50,
    "trend_change_threshold": 0.25,
    "periodicity_correlation_threshold": 0.50,
    "minimum_period_fraction": 0.10,
    "minimum_cycles": 3.0,
    "period_compatibility_relative_tolerance": 0.25,
}
SEARCH_BOUND_SCENARIOS = {
    "current": {
        "a": (0.035, 0.05),
        "theta": (0.0, 0.006),
        "e": (0.1, 0.8),
        "mu": (0.05, 0.8),
    },
    "expanded": {
        "a": (0.015, 0.08),
        "theta": (0.0, 0.012),
        "e": (0.05, 1.0),
        "mu": (0.02, 1.2),
    },
}


def _load_report_series() -> tuple[list[PredatorPreySeries], dict]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    report_entries = manifest["report_series"]
    excluded_entries = manifest["excluded_series"]
    entries = report_entries + excluded_entries
    expected_ids = [entry["id"] for entry in entries]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("duplicate series id in Holling report manifest")

    discovered = discover_and_load(min_confidence=float(manifest["min_confidence"]))
    by_id = {series.name: series for series in discovered}
    actual_ids = set(by_id)
    expected_id_set = set(expected_ids)
    missing = sorted(expected_id_set - actual_ids)
    added = sorted(actual_ids - expected_id_set)
    if missing or added:
        raise RuntimeError(
            "Holling report series manifest mismatch: "
            f"missing={missing or 'none'}, unlisted={added or 'none'}"
        )

    mismatches: list[str] = []
    for entry in entries:
        series = by_id[entry["id"]]
        for key, actual in (
            ("signature", series.meta.get("signature")),
            ("group_key", series.meta.get("group_key")),
            ("n_points", series.n_points),
        ):
            if actual != entry[key]:
                mismatches.append(
                    f"{entry['id']}.{key}: expected {entry[key]!r}, found {actual!r}"
                )
    if mismatches:
        raise RuntimeError(
            "Holling report series metadata changed:\n  - " + "\n  - ".join(mismatches)
        )

    report_series = [by_id[entry["id"]] for entry in report_entries]
    return report_series, manifest


def _series_summary(series: PredatorPreySeries) -> dict[str, object]:
    try:
        source = Path(series.source_path).relative_to(ROOT).as_posix()
    except ValueError:
        source = series.source_path
    return {
        "id": series.name,
        "source": source,
        "signature": series.meta.get("signature"),
        "group_key": series.meta.get("group_key"),
        "n_points": series.n_points,
        "duration": series.duration,
        "prey_min": float(np.min(series.prey)),
        "prey_max": float(np.max(series.prey)),
        "predator_min": float(np.min(series.predator)),
        "predator_max": float(np.max(series.predator)),
    }


def _write_series_validation(
    series_list: list[PredatorPreySeries],
    manifest: dict,
) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    output = {
        "manifest": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "report_series_count": len(series_list),
        "excluded_series": manifest["excluded_series"],
        "report_series": [_series_summary(series) for series in series_list],
    }
    path = OUT / "series_manifest_validation.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _time_weighted_mean(t: np.ndarray, values: np.ndarray) -> float:
    if values.size == 1:
        return float(values[0])
    return float(trapezoid(values, t) / (t[-1] - t[0]))


def _tail_mean(t: np.ndarray, values: np.ndarray, tail_frac: float = 0.5) -> float:
    start = t[-1] - tail_frac * (t[-1] - t[0])
    mask = t >= start
    return _time_weighted_mean(t[mask], values[mask])


def _scale_values(values: np.ndarray, method: str) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=float)
    if method == "max":
        scale = max(float(np.max(np.abs(values))), 1e-12)
        return values / scale, {"center": 0.0, "scale": scale}
    if method == "robust_q95":
        scale = max(float(np.quantile(np.abs(values), 0.95)), 1e-12)
        return values / scale, {"center": 0.0, "scale": scale}
    if method == "zscore":
        center = float(np.mean(values))
        scale = max(float(np.std(values)), 1e-12)
        return (values - center) / scale, {"center": center, "scale": scale}
    raise ValueError(f"unknown scaling method: {method}")


def _population_diagnostics(
    t: np.ndarray,
    values: np.ndarray,
) -> dict[str, float | int | bool | None]:
    scaled, _ = _scale_values(values, "robust_q95")
    n = scaled.size
    t_normalized = (t - t[0]) / (t[-1] - t[0])
    slope, intercept = np.polyfit(t_normalized, scaled, 1)
    trend_corr = (
        float(np.corrcoef(t_normalized, scaled)[0, 1])
        if np.std(scaled) > 1e-12
        else 0.0
    )
    trend_change = float(abs(slope))
    has_trend = (
        abs(trend_corr) >= DIAGNOSTIC_CONFIG["trend_correlation_threshold"]
        and trend_change >= DIAGNOSTIC_CONFIG["trend_change_threshold"]
    )

    window_duration = DIAGNOSTIC_CONFIG["tail_window_frac"] * (t[-1] - t[0])
    previous_mask = (t >= t[-1] - 2.0 * window_duration) & (t < t[-1] - window_duration)
    final_mask = t >= t[-1] - window_duration
    previous_mean = _time_weighted_mean(t[previous_mask], scaled[previous_mask])
    final_mean = _time_weighted_mean(t[final_mask], scaled[final_mask])
    tail_mean_change = abs(final_mean - previous_mean) / max(
        abs(previous_mean),
        abs(final_mean),
        0.05,
    )
    tail_stable = tail_mean_change <= DIAGNOSTIC_CONFIG["tail_mean_change_tolerance"]

    uniform_t = np.linspace(t[0], t[-1], n)
    uniform_scaled = np.interp(uniform_t, t, scaled)
    uniform_t_normalized = (uniform_t - uniform_t[0]) / (uniform_t[-1] - uniform_t[0])
    detrended = uniform_scaled - (slope * uniform_t_normalized + intercept)
    max_lag = n // 3
    autocorrelation = np.array(
        [
            float(np.corrcoef(detrended[:-lag], detrended[lag:])[0, 1])
            if np.std(detrended[:-lag]) > 1e-12
            and np.std(detrended[lag:]) > 1e-12
            else 0.0
            for lag in range(1, max_lag + 1)
        ]
    )
    peaks, _ = find_peaks(autocorrelation)
    min_period = max(3, int(np.ceil(n * DIAGNOSTIC_CONFIG["minimum_period_fraction"])))
    valid_peaks = [
        int(index)
        for index in peaks
        if index + 1 >= min_period
        and n / (index + 1) >= DIAGNOSTIC_CONFIG["minimum_cycles"]
    ]
    if valid_peaks:
        best_index = max(valid_peaks, key=lambda index: autocorrelation[index])
        periodicity_correlation = float(autocorrelation[best_index])
        period_points: int | None = best_index + 1
        period_duration: float | None = float(
            period_points * (uniform_t[1] - uniform_t[0])
        )
    else:
        periodicity_correlation = 0.0
        period_points = None
        period_duration = None
    is_periodic = (
        periodicity_correlation
        >= DIAGNOSTIC_CONFIG["periodicity_correlation_threshold"]
    )
    return {
        "trend_correlation": trend_corr,
        "trend_change": trend_change,
        "has_trend": bool(has_trend),
        "tail_mean_change": float(tail_mean_change),
        "tail_stable": bool(tail_stable),
        "periodicity_correlation": periodicity_correlation,
        "period_points": period_points,
        "period_duration": period_duration,
        "is_periodic": bool(is_periodic),
    }


def _series_diagnostics(series: PredatorPreySeries) -> dict[str, object]:
    if (
        not np.all(np.isfinite(series.t))
        or series.duration <= 0.0
        or np.any(np.diff(series.t) <= 0.0)
    ):
        return {
            "id": series.name,
            "classification": "invalid_time",
            "included_in_target": False,
            "target_definition": "excluded",
            "period_duration": None,
            "prey": None,
            "predator": None,
        }
    prey = _population_diagnostics(series.t, series.prey)
    predator = _population_diagnostics(series.t, series.predator)
    prey_period = prey["period_duration"]
    predator_period = predator["period_duration"]
    periods_compatible = (
        prey["is_periodic"]
        and predator["is_periodic"]
        and prey_period is not None
        and predator_period is not None
        and abs(prey_period - predator_period) / max(prey_period, predator_period)
        <= DIAGNOSTIC_CONFIG["period_compatibility_relative_tolerance"]
    )
    if prey["has_trend"] or predator["has_trend"]:
        classification = "nonstationary_trend"
    elif periods_compatible:
        classification = "periodic"
    elif prey["tail_stable"] and predator["tail_stable"]:
        classification = "stable"
    else:
        classification = "nonstationary"
    period_duration = (
        float(np.mean([prey_period, predator_period]))
        if periods_compatible
        else None
    )
    return {
        "id": series.name,
        "classification": classification,
        "included_in_target": classification in ("stable", "periodic"),
        "target_definition": (
            "complete_cycles_mean"
            if classification == "periodic"
            else "tail_mean"
            if classification == "stable"
            else "excluded"
        ),
        "period_duration": period_duration,
        "prey": prey,
        "predator": predator,
    }


def _target_value(
    t: np.ndarray,
    values: np.ndarray,
    classification: str,
    period_duration: float | None,
) -> float:
    if classification == "stable":
        return _tail_mean(t, values, DIAGNOSTIC_CONFIG["tail_frac"])
    if classification == "periodic" and period_duration is not None:
        complete_cycles = int((t[-1] - t[0]) // period_duration)
        start = t[-1] - complete_cycles * period_duration
        mask = t >= start
        return _time_weighted_mean(t[mask], values[mask])
    raise ValueError(f"cannot compute target for {classification}")


def empirical_target(
    series_list: list[PredatorPreySeries],
    K: float,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    scaling_targets = {method: [] for method in ("max", "robust_q95", "zscore")}
    for series in series_list:
        diagnostics = _series_diagnostics(series)
        row = {**diagnostics, "scaling": {}}
        if diagnostics["included_in_target"]:
            for method in scaling_targets:
                prey, prey_scaling = _scale_values(series.prey, method)
                pred, pred_scaling = _scale_values(series.predator, method)
                prey_target = _target_value(
                    series.t,
                    prey,
                    str(diagnostics["classification"]),
                    diagnostics["period_duration"],
                )
                pred_target = _target_value(
                    series.t,
                    pred,
                    str(diagnostics["classification"]),
                    diagnostics["period_duration"],
                )
                row["scaling"][method] = {
                    "prey": prey_scaling,
                    "predator": pred_scaling,
                    "target_prey": prey_target,
                    "target_predator": pred_target,
                }
                scaling_targets[method].append((prey_target, pred_target))
        rows.append(row)

    included_series = [row["id"] for row in rows if row["included_in_target"]]
    excluded_series = [row["id"] for row in rows if not row["included_in_target"]]
    primary_targets = scaling_targets[DIAGNOSTIC_CONFIG["primary_scaling"]]
    scaling_sensitivity = {}
    for method, targets in scaling_targets.items():
        if not targets:
            scaling_sensitivity[method] = {
                "status": "not_available",
                "reason": "no eligible equilibrium-target series",
            }
            continue
        values = np.asarray(targets, dtype=float)
        scaling_sensitivity[method] = (
            {
                "diagnostic_only": True,
                "prey_standardized_location_median": float(np.median(values[:, 0])),
                "predator_standardized_location_median": float(np.median(values[:, 1])),
            }
            if method == "zscore"
            else {
                "target_x": float(np.median(values[:, 0]) * K),
                "target_y": float(np.median(values[:, 1]) * K),
                "prey_median": float(np.median(values[:, 0])),
                "predator_median": float(np.median(values[:, 1])),
            }
        )
    summary: dict[str, object] = {
        "status": "ok" if primary_targets else "insufficient_eligible_series",
        "reason": (
            None
            if primary_targets
            else "No report series passed the strict stable-or-joint-periodic diagnostics."
        ),
        "diagnostic_config": DIAGNOSTIC_CONFIG,
        "included_series": included_series,
        "excluded_series": excluded_series,
        "target_x": None,
        "target_y": None,
        "scaling_sensitivity": scaling_sensitivity,
    }
    if primary_targets:
        target_array = np.asarray(primary_targets, dtype=float)
        target_prey = target_array[:, 0]
        target_pred = target_array[:, 1]
        summary.update(
            {
                "target_x": float(np.median(target_prey) * K),
                "target_y": float(np.median(target_pred) * K),
                "target_prey_median": float(np.median(target_prey)),
                "target_predator_median": float(np.median(target_pred)),
                "target_prey_q25": float(np.quantile(target_prey, 0.25)),
                "target_prey_q75": float(np.quantile(target_prey, 0.75)),
                "target_predator_q25": float(np.quantile(target_pred, 0.25)),
                "target_predator_q75": float(np.quantile(target_pred, 0.75)),
            }
        )
    return summary, rows


def _write_target_diagnostics(summary: dict[str, object], rows: list[dict[str, object]]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "equilibrium_target_diagnostics.json"
    path.write_text(
        json.dumps({"summary": summary, "series": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _structural_identifiability_report(
    summary: dict[str, object],
    base: BaselineParams,
) -> dict[str, object]:
    target_available = summary["status"] == "ok"
    skipped = {
        "status": "skipped",
        "reason": "No valid equilibrium target is available from the strict diagnostics.",
    }
    return {
        "status": "structurally_non_identifiable_from_equilibrium_target",
        "target_status": summary["status"],
        "target": {"x_star": summary["target_x"], "y_star": summary["target_y"]},
        "artificial_prior": {
            "status": "removed",
            "note": "The calibration objective contains no parameter-centering prior.",
        },
        "objective": "equilibrium RMSE in model-density units; parameter prior removed",
        "hardcoded_equilibrium_bounds": {
            "status": "removed",
            "remaining_constraints": [
                "positive finite coexistence equilibrium",
                "x_star < K",
                "positive predator invasion growth",
                "local stability trace < 0",
            ],
        },
        "derivation": {
            "free_parameters": ["theta", "e"],
            "dependent_parameters": {
                "a": "r * (1 - x_star / K) * (1 + theta * x_star) / y_star",
                "mu": "e * r * x_star * (1 - x_star / K) / y_star",
            },
            "conclusion": (
                "A single coexistence equilibrium supplies only two independent "
                "relations for a, theta, e, and mu, leaving a two-dimensional ridge."
            ),
            "fixed_parameters": {"r": base.r, "K": base.K},
        },
        "search_bound_scenarios": SEARCH_BOUND_SCENARIOS,
        "numerical_search": (
            {"status": "available", "note": "Run only when a valid target exists."}
            if target_available
            else skipped
        ),
        "near_optimal_parameter_distribution": (
            {"status": "available_when_numerical_search_runs"}
            if target_available
            else skipped
        ),
        "objective_contours": (
            {"status": "available_when_numerical_search_runs"}
            if target_available
            else skipped
        ),
        "search_bound_sensitivity": (
            {"status": "available_when_numerical_search_runs"}
            if target_available
            else skipped
        ),
        "target_definition_sensitivity": (
            {"status": "available_when_multiple_valid_target_definitions_exist"}
            if target_available
            else skipped
        ),
    }


def _write_identifiability_report(report: dict[str, object]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "identifiability_sensitivity.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _equilibrium_and_stability(
    r: float,
    K: float,
    a: np.ndarray,
    theta: np.ndarray,
    e: np.ndarray,
    mu: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    denom = e * a - mu * theta
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        x_star = np.where(denom > 0.0, mu / np.maximum(denom, 1e-300), np.inf)
        y_star = r * (1.0 - x_star / K) * (1.0 + theta * x_star) / a
        trace = r * (1.0 - 2.0 * x_star / K) - a * y_star / (1.0 + theta * x_star) ** 2
        gmax = e * a * K / (1.0 + theta * K) - mu
    return x_star, y_star, trace, gmax


def search_holling_defaults(
    target_x: float,
    target_y: float,
    base: BaselineParams,
    top_n: int,
    bounds: dict[str, tuple[float, float]] | None = None,
) -> list[dict[str, float]]:
    bounds = bounds or SEARCH_BOUND_SCENARIOS["current"]
    a_grid = np.linspace(*bounds["a"], 61)
    theta_grid = np.linspace(*bounds["theta"], 61)
    e_grid = np.linspace(*bounds["e"], 71)
    mu_grid = np.linspace(*bounds["mu"], 151)
    e_mesh, mu_mesh = np.meshgrid(e_grid, mu_grid, indexing="ij")

    candidates: list[np.ndarray] = []
    for a in a_grid:
        for theta in theta_grid:
            x_star, y_star, trace, gmax = _equilibrium_and_stability(
                base.r,
                base.K,
                np.array(a),
                np.array(theta),
                e_mesh,
                mu_mesh,
            )
            ok = (
                (gmax > 0.0)
                & (x_star > 0.0)
                & (x_star < base.K)
                & (y_star > 0.0)
                & (trace < 0.0)
            )
            if not np.any(ok):
                continue

            score = np.sqrt(
                ((x_star[ok] - target_x) ** 2 + (y_star[ok] - target_y) ** 2)
                / 2.0
            )
            block = np.column_stack(
                [
                    score,
                    np.full(score.size, a),
                    np.full(score.size, theta),
                    e_mesh[ok],
                    mu_mesh[ok],
                    x_star[ok],
                    y_star[ok],
                    trace[ok],
                    gmax[ok],
                ]
            )
            candidates.append(block)

    if not candidates:
        return []
    all_candidates = np.vstack(candidates)
    order = np.argsort(all_candidates[:, 0])[:top_n]
    rows: list[dict[str, float]] = []
    for row in all_candidates[order]:
        rows.append(
            {
                "score": float(row[0]),
                "a": float(row[1]),
                "theta": float(row[2]),
                "e": float(row[3]),
                "mu": float(row[4]),
                "x_star": float(row[5]),
                "y_star": float(row[6]),
                "trace": float(row[7]),
                "predator_gmax": float(row[8]),
            }
        )
    return rows


def verify_candidates(
    candidates: list[dict[str, float]],
    base: BaselineParams,
    t_end: float = 300.0,
) -> list[dict[str, float]]:
    verified: list[dict[str, float]] = []
    for row in candidates:
        p = BaselineParams(
            r=base.r,
            K=base.K,
            a=row["a"],
            theta=row["theta"],
            e=row["e"],
            mu=row["mu"],
        )
        result = integrate_until_converged(
            lambda end, points: integrate_baseline(
                p, t_span=(0.0, end), n_points=points
            ),
            t_end=t_end,
            n_points=1500,
            scales=(base.K, base.K),
        )
        sol = result.sol
        x_mean, y_mean = result.metrics.means
        tail = sol.y[:, int(sol.t.size * 0.75) :]
        verified.append(
            {
                **row,
                "status": is_extinct(sol, scales=(base.K, base.K)),
                "convergence_status": result.convergence.status,
                "t_end_used": result.t_end_used,
                "extensions": result.extensions,
                "tail_x_min": float(np.min(tail[0])),
                "tail_y_min": float(np.min(tail[1])),
                "tail_x_max": float(np.max(tail[0])),
                "tail_y_max": float(np.max(tail[1])),
                "mean_x": x_mean,
                "mean_y": y_mean,
            }
        )
    return verified


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate stable Holling II defaults")
    parser.add_argument("--top", type=int, default=10)
    args = parser.parse_args()

    base = BaselineParams()
    series_list, manifest = _load_report_series()
    validation_path = _write_series_validation(series_list, manifest)

    summary, target_rows = empirical_target(series_list, base.K)
    diagnostics_path = _write_target_diagnostics(summary, target_rows)
    identifiability_path = _write_identifiability_report(
        _structural_identifiability_report(summary, base)
    )

    print("=" * 72)
    print("Holling II global default calibration")
    print("=" * 72)
    print("Series:")
    for series in series_list:
        print(f"  - {series.name} ({series.n_points} points)")
    print(f"  validation: {validation_path.relative_to(ROOT)}")
    print("\nEmpirical equilibrium target diagnostics:")
    for row in target_rows:
        print(f"  - {row['id']}: {row['classification']}")
    print(f"  included={len(summary['included_series'])}, excluded={len(summary['excluded_series'])}")
    print(f"  diagnostics: {diagnostics_path.relative_to(ROOT)}")
    print(f"  identifiability: {identifiability_path.relative_to(ROOT)}")
    if summary["status"] != "ok":
        print("\nCalibration stopped:")
        print(f"  status={summary['status']}")
        print(f"  reason={summary['reason']}")
        print("  No equilibrium target or representative BaselineParams was produced.")
        print("=" * 72)
        return

    print(
        "  prey median={target_prey_median:.3f} "
        "IQR=({target_prey_q25:.3f}, {target_prey_q75:.3f})".format(**summary)
    )
    print(
        "  predator median={target_predator_median:.3f} "
        "IQR=({target_predator_q25:.3f}, {target_predator_q75:.3f})".format(**summary)
    )
    print(f"  target equilibrium on K={base.K:g} scale: x*={summary['target_x']:.3f}, y*={summary['target_y']:.3f}")

    candidates = search_holling_defaults(
        target_x=float(summary["target_x"]),
        target_y=float(summary["target_y"]),
        base=base,
        top_n=args.top,
    )
    verified = verify_candidates(candidates, base)

    print("\nTop verified candidates:")
    for i, row in enumerate(verified, start=1):
        print(
            f"  {i:02d}. a={row['a']:.4f}, theta={row['theta']:.4f}, "
            f"e={row['e']:.3f}, mu={row['mu']:.3f} | "
            f"x*={row['x_star']:.3f}, y*={row['y_star']:.3f}, "
            f"trace={row['trace']:.4f}, gmax={row['predator_gmax']:.4f}, "
            f"status={row['status']}, mean=({row['mean_x']:.3f}, {row['mean_y']:.3f})"
        )

    if verified:
        best = verified[0]
        params = BaselineParams(
            r=base.r,
            K=base.K,
            a=best["a"],
            theta=best["theta"],
            e=best["e"],
            mu=best["mu"],
        )
        print("\nRepresentative best grid candidate (not uniquely identified):")
        print(f"  {asdict(params)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
