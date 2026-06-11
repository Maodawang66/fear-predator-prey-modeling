"""Load the explicitly selected GPDD predator-prey pairs used in the report."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import normalize_time_years, read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries

PAIR_SPECS = {
    "windermere_north_pike_perch": {
        "prey_id": "6097",
        "predator_id": "6098",
        "prey_label": "Eurasian perch biomass",
        "predator_label": "pike biomass",
        "group_key": "Windermere North Basin",
        "notes": "Pike were removed by gill netting; both series use kg/ha.",
    },
    "windermere_south_pike_perch": {
        "prey_id": "6099",
        "predator_id": "6100",
        "prey_label": "Eurasian perch biomass",
        "predator_label": "pike biomass",
        "group_key": "Windermere South Basin",
        "notes": "Pike were removed by gill netting; both series use kg/ha.",
    },
    "komi_lynx_hare": {
        "prey_id": "9511",
        "predator_id": "9512",
        "prey_label": "mountain hare transformed count",
        "predator_label": "Eurasian lynx transformed count",
        "group_key": "Komi Republic",
        "notes": "GPDD proportion-transformed count series; reliability grade 1.",
    },
}


def load_gpdd_pair(name: str, path: str | None = None) -> PredatorPreySeries:
    if name not in PAIR_SPECS:
        raise ValueError(f"unknown formal GPDD pair: {name}")
    spec = PAIR_SPECS[name]
    csv_path = resolve_data_file(
        path or "05_gpdd/data/gpdd_population_records.csv"
    )
    wanted = {spec["prey_id"], spec["predator_id"]}
    by_id: dict[str, dict[int, float]] = defaultdict(dict)
    for row in read_csv_dicts(csv_path):
        main_id = row.get("MainID", "")
        if main_id not in wanted:
            continue
        try:
            year = int(float(row["SampleYear"]))
            value = float(row["Population"])
        except (KeyError, ValueError):
            continue
        by_id[main_id][year] = value

    years = sorted(set(by_id[spec["prey_id"]]) & set(by_id[spec["predator_id"]]))
    if len(years) < 4:
        raise ValueError(f"{name}: fewer than four overlapping GPDD years")
    year_array = np.asarray(years, dtype=float)
    prey = np.asarray([by_id[spec["prey_id"]][year] for year in years])
    predator = np.asarray([by_id[spec["predator_id"]][year] for year in years])
    return PredatorPreySeries(
        name=name,
        t=normalize_time_years(year_array),
        prey=prey,
        predator=predator,
        prey_label=spec["prey_label"],
        predator_label=spec["predator_label"],
        source_path=str(csv_path),
        meta={
            "signature": "gpdd_formal_pair",
            "detection_method": "formal_gpdd_pair",
            "confidence": 1.0,
            "time_col": "SampleYear",
            "prey_col": f"Population(MainID={spec['prey_id']})",
            "predator_col": f"Population(MainID={spec['predator_id']})",
            "group_key": spec["group_key"],
            "year_start": years[0],
            "year_end": years[-1],
            "n_points": len(years),
            "gpdd_prey_main_id": spec["prey_id"],
            "gpdd_predator_main_id": spec["predator_id"],
            "notes": spec["notes"],
        },
    )


def load_windermere_north() -> PredatorPreySeries:
    return load_gpdd_pair("windermere_north_pike_perch")


def load_windermere_south() -> PredatorPreySeries:
    return load_gpdd_pair("windermere_south_pike_perch")


def load_komi_lynx_hare() -> PredatorPreySeries:
    return load_gpdd_pair("komi_lynx_hare")
