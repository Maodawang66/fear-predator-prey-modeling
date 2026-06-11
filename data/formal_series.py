"""Pinned population-series set used by all formal report analyses."""

from __future__ import annotations

from .auto_discover import discover_and_load
from .load_gpdd_pairs import load_komi_lynx_hare, load_windermere_north, load_windermere_south
from .load_isle_royale import load_isle_royale_pre_intervention
from .series import PredatorPreySeries

LEGACY_FORMAL_IDS = (
    "glerl_m110_zoop_1994-201",
    "andren_lynx_roedeer_data_1",
    "andren_lynx_roedeer_data_2",
    "andren_lynx_roedeer_data_3",
    "andren_lynx_roedeer_data_4",
    "andren_lynx_roedeer_data_5",
    "andren_lynx_roedeer_data_6",
    "andren_lynx_roedeer_data_7",
    "timeserieslogmeans_WRHW",
    "timeserieslogmeans_TP",
    "timeserieslogmeans_WRGP",
)
NEW_FORMAL_IDS = (
    "isle_royale_wolf_moose_pre_2018",
    "windermere_north_pike_perch",
    "windermere_south_pike_perch",
    "komi_lynx_hare",
)
FORMAL_SERIES_IDS = LEGACY_FORMAL_IDS + NEW_FORMAL_IDS
EXCLUDED_FORMAL_IDS = ("lynxhare",)


def _start_at_first_joint_positive_observation(series: PredatorPreySeries) -> PredatorPreySeries:
    """Drop leading observations that would put population ODEs in an absorbing zero state."""
    joint_positive = (series.prey > 0.0) & (series.predator > 0.0)
    indices = joint_positive.nonzero()[0]
    if not indices.size:
        raise ValueError(f"{series.name}: no observation has both prey and predator positive")
    start = int(indices[0])
    if start == 0:
        return series
    return PredatorPreySeries(
        name=series.name,
        t=series.t[start:] - series.t[start],
        prey=series.prey[start:],
        predator=series.predator[start:],
        time_unit=series.time_unit,
        prey_label=series.prey_label,
        predator_label=series.predator_label,
        source_path=series.source_path,
        meta={
            **series.meta,
            "formal_start_rule": "first_joint_positive_observation",
            "leading_observations_dropped": start,
            "original_n_points": series.n_points,
            "original_start_time": float(series.t[start]),
            "n_points": series.n_points - start,
        },
    )


def load_formal_series() -> list[PredatorPreySeries]:
    discovered = {series.name: series for series in discover_and_load(min_confidence=0.5)}
    missing = [name for name in LEGACY_FORMAL_IDS if name not in discovered]
    if missing:
        raise RuntimeError(f"missing legacy formal series: {missing}")
    added = {
        series.name: series
        for series in (
            load_isle_royale_pre_intervention(),
            load_windermere_north(),
            load_windermere_south(),
            load_komi_lynx_hare(),
        )
    }
    combined = {**discovered, **added}
    series_list = [
        _start_at_first_joint_positive_observation(combined[name])
        for name in FORMAL_SERIES_IDS
    ]
    names = [series.name for series in series_list]
    if len(names) != 15 or len(set(names)) != 15 or set(names) & set(EXCLUDED_FORMAL_IDS):
        raise RuntimeError(f"invalid formal series set: {names}")
    return series_list
