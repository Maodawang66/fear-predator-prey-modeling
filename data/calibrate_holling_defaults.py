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

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.auto_discover import discover_and_load  # noqa: E402
from data.series import PredatorPreySeries  # noqa: E402
from src.parameters import BaselineParams  # noqa: E402
from src.simulate import integrate_baseline, is_extinct, long_term_mean  # noqa: E402

MANIFEST_PATH = ROOT / "data" / "holling_report_series_manifest.json"
OUT = ROOT / "results" / "holling_defaults"


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


def _tail_mean(values: np.ndarray, tail_frac: float = 0.5) -> float:
    i0 = int(values.size * (1.0 - tail_frac))
    return float(np.mean(values[i0:]))


def empirical_target(
    series_list: list[PredatorPreySeries],
    K: float,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    for series in series_list:
        prey_scale = max(float(np.max(series.prey)), 1.0)
        pred_scale = max(float(np.max(series.predator)), 1.0)
        prey = series.prey / prey_scale
        pred = series.predator / pred_scale
        rows.append(
            {
                "tail_prey": _tail_mean(prey),
                "tail_predator": _tail_mean(pred),
                "median_prey": float(np.median(prey)),
                "median_predator": float(np.median(pred)),
                "mean_prey": float(np.mean(prey)),
                "mean_predator": float(np.mean(pred)),
            }
        )

    tail_prey = np.array([r["tail_prey"] for r in rows], dtype=float)
    tail_pred = np.array([r["tail_predator"] for r in rows], dtype=float)
    summary = {
        "target_x": float(np.median(tail_prey) * K),
        "target_y": float(np.median(tail_pred) * K),
        "tail_prey_median": float(np.median(tail_prey)),
        "tail_predator_median": float(np.median(tail_pred)),
        "tail_prey_q25": float(np.quantile(tail_prey, 0.25)),
        "tail_prey_q75": float(np.quantile(tail_prey, 0.75)),
        "tail_predator_q25": float(np.quantile(tail_pred, 0.25)),
        "tail_predator_q75": float(np.quantile(tail_pred, 0.75)),
    }
    return summary, rows


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
) -> list[dict[str, float]]:
    a_grid = np.linspace(0.035, 0.05, 61)
    theta_grid = np.linspace(0.0, 0.006, 61)
    e_grid = np.linspace(0.1, 0.8, 71)
    mu_grid = np.linspace(0.05, 0.8, 151)
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
                & (x_star >= 20.0)
                & (x_star <= 30.0)
                & (y_star >= 20.0)
                & (y_star <= 35.0)
            )
            if not np.any(ok):
                continue

            target_score = (
                ((x_star[ok] - target_x) / 10.0) ** 2
                + ((y_star[ok] - target_y) / 10.0) ** 2
            )
            # e, mu, a, and theta are weakly identifiable from an aggregate
            # equilibrium target, so this prior only breaks near-ties.
            prior_score = 0.05 * (
                ((a - 0.044) / 0.02) ** 2
                + ((theta - 0.005) / 0.005) ** 2
                + ((e_mesh[ok] - 0.5) / 0.3) ** 2
                + ((mu_mesh[ok] - 0.4) / 0.3) ** 2
            )
            score = target_score + prior_score
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
        sol = integrate_baseline(p, t_span=(0.0, t_end), n_points=1500)
        x_mean, y_mean = long_term_mean(sol, burn_in_frac=0.5)
        tail = sol.y[:, int(sol.t.size * 0.75) :]
        verified.append(
            {
                **row,
                "status": is_extinct(sol),
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

    summary, _ = empirical_target(series_list, base.K)
    candidates = search_holling_defaults(
        target_x=summary["target_x"],
        target_y=summary["target_y"],
        base=base,
        top_n=args.top,
    )
    verified = verify_candidates(candidates, base)

    print("=" * 72)
    print("Holling II global default calibration")
    print("=" * 72)
    print("Series:")
    for series in series_list:
        print(f"  - {series.name} ({series.n_points} points)")
    print(f"  validation: {validation_path.relative_to(ROOT)}")
    print("\nEmpirical normalized tail means across series:")
    print(
        "  prey median={tail_prey_median:.3f} "
        "IQR=({tail_prey_q25:.3f}, {tail_prey_q75:.3f})".format(**summary)
    )
    print(
        "  predator median={tail_predator_median:.3f} "
        "IQR=({tail_predator_q25:.3f}, {tail_predator_q75:.3f})".format(**summary)
    )
    print(f"  target equilibrium on K={base.K:g} scale: x*={summary['target_x']:.3f}, y*={summary['target_y']:.3f}")

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
        print("\nRecommended BaselineParams:")
        print(f"  {asdict(params)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
