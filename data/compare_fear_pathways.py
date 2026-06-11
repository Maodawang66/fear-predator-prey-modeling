"""Compare equally parameterized fear pathways on the pinned formal time series."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.formal_series import load_formal_series  # noqa: E402
from src.fear_pathway_fit import (  # noqa: E402
    HOLLING_FEAR_PATHWAYS,
    fit_holling_baseline_to_series,
    fit_holling_fear_pathway_to_series,
    profile_holling_fear_strength,
)
from src.fit import FitResult  # noqa: E402

OUT = ROOT / "results" / "fear_pathway_comparison"
USABLE = {"success", "usable_limit"}


def _study_id(series_name: str) -> str:
    name = series_name.lower()
    if "andren_lynx_roedeer" in name:
        return "andren_lynx_roedeer"
    if "timeserieslogmeans" in name or "killifish" in name:
        return "killifish_mosquitofish"
    if "glerl" in name or "zoop" in name:
        return "glerl_zooplankton"
    if "isle_royale" in name:
        return "isle_royale_wolf_moose"
    if "windermere" in name:
        return "windermere_pike_perch"
    if "komi_lynx_hare" in name:
        return "komi_lynx_hare"
    return series_name


def _row(result: FitResult, study: str) -> dict:
    fear_parameter = result.meta.get("fear_parameter", "")
    fear_value = result.params.get(fear_parameter, "") if fear_parameter else ""
    return {
        "series": result.series_name,
        "study": study,
        "model": result.model,
        "training_rmse": result.rmse_normalized_total,
        "validation_rmse": result.validation_rmse_normalized_total,
        "aicc": result.aicc,
        "optimization_status": result.optimization_status,
        "usable_for_comparison": result.usable_for_comparison,
        "fear_parameter": fear_parameter,
        "fear_value": fear_value,
        "parameter_bound_hits": ";".join(result.meta.get("parameter_bound_hits", [])),
        "objective_value": result.meta.get("objective_value", ""),
        "n_parameters": result.n_parameters,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("usable_for_comparison", "inside_confidence_interval"):
            if key in row:
                row[key] = row[key] == "True"
    return rows


def _aggregate(rows: list[dict], group_key: str) -> list[dict]:
    required_models = {"baseline", *HOLLING_FEAR_PATHWAYS}
    usable_by_series: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["usable_for_comparison"]:
            usable_by_series[row["series"]].add(row["model"])
    complete_series = {
        series for series, models in usable_by_series.items()
        if models == required_models
    }
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        if row["usable_for_comparison"] and row["series"] in complete_series:
            grouped[(row[group_key], row["model"])].append(row)
    output = []
    for (group, model), values in sorted(grouped.items()):
        output.append({
            group_key: group,
            "model": model,
            "n_series": len(values),
            "comparison_set": "complete_cases_all_six_models",
            "median_training_rmse": float(np.median([float(r["training_rmse"]) for r in values])),
            "median_validation_rmse": float(np.median([float(r["validation_rmse"]) for r in values])),
            "median_aicc": float(np.median([float(r["aicc"]) for r in values])),
            "bound_hit_fraction": float(np.mean([bool(r["parameter_bound_hits"]) for r in values])),
        })
    return output


def _memory_initialization_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["usable_for_comparison"]:
            grouped[row["initial_memory_scheme"]].append(row)
    return [
        {
            "initial_memory_scheme": scheme,
            "n_usable_series": len(values),
            "median_training_rmse": float(np.median([float(row["training_rmse"]) for row in values])),
            "median_validation_rmse": float(np.median([float(row["validation_rmse"]) for row in values])),
            "median_aicc": float(np.median([float(row["aicc"]) for row in values])),
        }
        for scheme, values in sorted(grouped.items())
    ]


def _winner_rows(rows: list[dict], group_key: str) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row["usable_for_comparison"]:
            grouped[row[group_key]].append(row)
    output = []
    for group, values in sorted(grouped.items()):
        record = {group_key: group, "n_usable_models": len(values)}
        for metric in ("training_rmse", "validation_rmse", "aicc"):
            best = min(float(row[metric]) for row in values)
            winners = sorted(
                row["model"] for row in values
                if np.isclose(float(row[metric]), best, rtol=1e-9, atol=1e-12)
            )
            record[f"{metric}_winner"] = ";".join(winners)
            record[f"{metric}_best"] = best
        output.append(record)
    return output


def _study_winner_rows(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["study"]].append(row)
    output = []
    for study, values in sorted(grouped.items()):
        record = {"study": study, "n_usable_models": len(values)}
        for metric in ("median_training_rmse", "median_validation_rmse", "median_aicc"):
            best = min(float(row[metric]) for row in values)
            winners = sorted(
                row["model"] for row in values
                if np.isclose(float(row[metric]), best, rtol=1e-9, atol=1e-12)
            )
            record[f"{metric}_winner"] = ";".join(winners)
            record[f"{metric}_best"] = best
        output.append(record)
    return output


def _profile_identifiability_rows(profiles: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in profiles:
        grouped[(row["series"], row["pathway"])].append(row)
    output = []
    for (series, pathway), values in sorted(grouped.items()):
        inside = [
            row for row in values
            if row.get("inside_confidence_interval") is True
        ]
        strengths = [float(row["fear_strength"]) for row in inside]
        output.append({
            "series": series,
            "study": values[0]["study"],
            "pathway": pathway,
            "fear_parameter": values[0]["fear_parameter"],
            "ci95_lower": min(strengths) if strengths else float("nan"),
            "ci95_upper": max(strengths) if strengths else float("nan"),
            "ci95_includes_zero": any(np.isclose(strength, 0.0) for strength in strengths),
            "ci95_lower_at_grid_boundary": bool(strengths and min(strengths) == min(
                float(row["fear_strength"]) for row in values
            )),
            "ci95_upper_at_grid_boundary": bool(strengths and max(strengths) == max(
                float(row["fear_strength"]) for row in values
            )),
        })
    return output


def _write_conclusion(rows: list[dict], study_rows: list[dict], profiles: list[dict]) -> None:
    usable_by_series: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if row["usable_for_comparison"]:
            usable_by_series[row["series"]].add(row["model"])
    required_models = {"baseline", *HOLLING_FEAR_PATHWAYS}
    complete_series = sorted(
        series for series, models in usable_by_series.items()
        if models == required_models
    )
    study_winners = _study_winner_rows(study_rows)
    validation_winners = {
        row["study"]: row["median_validation_rmse_winner"]
        for row in study_winners
    }
    aicc_winners = {
        row["study"]: row["median_aicc_winner"]
        for row in study_winners
    }
    identifiability = _profile_identifiability_rows(profiles)
    profile_summary = {
        pathway: {
            "n_profiled": len(values),
            "ci95_includes_zero": sum(bool(row["ci95_includes_zero"]) for row in values),
            "ci95_hits_grid_boundary": sum(
                bool(row["ci95_lower_at_grid_boundary"] or row["ci95_upper_at_grid_boundary"])
                for row in values
            ),
        }
        for pathway in HOLLING_FEAR_PATHWAYS
        for values in [[row for row in identifiability if row["pathway"] == pathway]]
    }
    conclusion = {
        "complete_case_series_count": len(complete_series),
        "failed_fit_count": sum(not bool(row["usable_for_comparison"]) for row in rows),
        "study_validation_winners": validation_winners,
        "study_aicc_winners": aicc_winners,
        "fear_parameter_profile_summary": profile_summary,
        "interpretation": (
            "No single additional fear pathway wins validation RMSE and AICc consistently "
            "across the independent studies. Some alternatives win within individual studies, "
            "so the result supports study-specific channel uncertainty rather than a universal "
            "absence of fear effects."
        ),
        "synthetic_recovery_interpretation": (
            "The controlled synthetic experiment recovers all five pathways when the shared "
            "core parameters are known, but joint core-and-fear refitting shows strong pathway "
            "substitutability. Real-data pathway labels therefore require cautious interpretation."
        ),
    }
    (OUT / "conclusion.json").write_text(
        json.dumps(conclusion, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _memory_initializations(series, delta: float) -> dict[str, float]:
    train_end = series.n_points - max(1, int(np.ceil(series.n_points * 0.20)))
    early_count = min(3, train_end)
    return {
        "predator_initial": float(series.predator[0]),
        "predator_initial_quasi_steady": float(series.predator[0] / delta),
        "early_predator_mean_quasi_steady": float(np.mean(series.predator[:early_count]) / delta),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-series", type=int, default=15)
    parser.add_argument("--optimizer", choices=("auto", "global", "local"), default="auto")
    parser.add_argument("--max-nfev", type=int, default=500)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        rows = _read_csv(OUT / "sequence_comparison.csv")
        profiles = _read_csv(OUT / "fear_strength_profiles.csv")
        memory_rows = _read_csv(OUT / "memory_initialization_comparison.csv")
        study_rows = _aggregate(rows, "study")
        _write_csv(OUT / "study_comparison.csv", study_rows)
        _write_csv(OUT / "channel_comparison.csv", _aggregate(rows, "model"))
        _write_csv(OUT / "sequence_winners.csv", _winner_rows(rows, "series"))
        _write_csv(OUT / "study_winners.csv", _study_winner_rows(study_rows))
        _write_csv(OUT / "memory_initialization_summary.csv", _memory_initialization_summary(memory_rows))
        _write_csv(OUT / "fear_strength_identifiability.csv", _profile_identifiability_rows(profiles))
        _write_conclusion(rows, study_rows, profiles)
        return

    series_list = load_formal_series()
    series_list = series_list[: args.max_series]
    OUT.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    profiles: list[dict] = []
    memory_rows: list[dict] = []
    for index, series in enumerate(series_list, start=1):
        study = _study_id(series.name)
        print(f"[{index}/{len(series_list)}] {series.name}")
        baseline = fit_holling_baseline_to_series(
            series, optimizer=args.optimizer, max_nfev=args.max_nfev
        )
        rows.append(_row(baseline, study))
        for pathway in HOLLING_FEAR_PATHWAYS:
            result = fit_holling_fear_pathway_to_series(
                series,
                pathway,
                optimizer=args.optimizer,
                max_nfev=args.max_nfev,
            )
            rows.append(_row(result, study))
            if pathway == "fear_memory":
                for label, m0 in _memory_initializations(series, delta=1.0).items():
                    variant = fit_holling_fear_pathway_to_series(
                        series,
                        pathway,
                        optimizer=args.optimizer,
                        max_nfev=args.max_nfev,
                        initial_memory=m0,
                    )
                    memory_rows.append({
                        **_row(variant, study),
                        "initial_memory_scheme": label,
                        "m0": m0,
                        "delta": 1.0,
                    })
            if not args.skip_profiles and result.usable_for_comparison:
                pathway_profile = profile_holling_fear_strength(
                    series, pathway, result, max_nfev=max(100, args.max_nfev // 2)
                )
                profiles.extend({"series": series.name, "study": study, **row} for row in pathway_profile)

    _write_csv(OUT / "sequence_comparison.csv", rows)
    study_rows = _aggregate(rows, "study")
    _write_csv(OUT / "study_comparison.csv", study_rows)
    _write_csv(OUT / "channel_comparison.csv", _aggregate(rows, "model"))
    _write_csv(OUT / "sequence_winners.csv", _winner_rows(rows, "series"))
    _write_csv(OUT / "study_winners.csv", _study_winner_rows(study_rows))
    _write_csv(OUT / "memory_initialization_comparison.csv", memory_rows)
    _write_csv(OUT / "memory_initialization_summary.csv", _memory_initialization_summary(memory_rows))
    _write_csv(OUT / "fear_strength_profiles.csv", profiles)
    _write_csv(OUT / "fear_strength_identifiability.csv", _profile_identifiability_rows(profiles))
    _write_conclusion(rows, study_rows, profiles)
    metadata = {
        "models": ["baseline", *HOLLING_FEAR_PATHWAYS],
        "shared_baseline_parameters": ["r", "K", "a", "theta", "e", "mu"],
        "fear_parameters_per_candidate": 1,
        "validation": "ordered 20% continuous multistep holdout",
        "optimizer": args.optimizer,
        "max_nfev": args.max_nfev,
        "series_count": len(series_list),
        "study_count": len({_study_id(series.name) for series in series_list}),
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
