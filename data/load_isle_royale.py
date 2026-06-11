"""Load the pre-intervention Isle Royale wolf-moose population series."""

from __future__ import annotations

import numpy as np

from .common import normalize_time_years, read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries

NPS_SOURCE_URL = "https://www.nps.gov/isro/learn/nature/wolf-moose-populations.htm"
INTERVENTION_YEAR = 2018


def load_isle_royale_pre_intervention(path: str | None = None) -> PredatorPreySeries:
    csv_path = resolve_data_file(
        path or "15_isle_royale_wolf_moose/isle_royale_wolf_moose_pre_2018.csv"
    )
    rows = read_csv_dicts(csv_path)
    years = np.array([float(row["year"]) for row in rows])
    if np.any(years >= INTERVENTION_YEAR):
        raise ValueError("Isle Royale pre-intervention series must end before 2018")
    moose = np.array([float(row["moose"]) for row in rows])
    wolves = np.array([float(row["wolves"]) for row in rows])
    return PredatorPreySeries(
        name="isle_royale_wolf_moose_pre_2018",
        t=normalize_time_years(years),
        prey=moose,
        predator=wolves,
        prey_label="moose",
        predator_label="wolves",
        source_path=str(csv_path),
        meta={
            "signature": "isle_royale_wolf_moose",
            "detection_method": "formal_loader",
            "confidence": 1.0,
            "time_col": "year",
            "prey_col": "moose",
            "predator_col": "wolves",
            "group_key": None,
            "year_start": int(years[0]),
            "year_end": int(years[-1]),
            "n_points": len(years),
            "source_url": NPS_SOURCE_URL,
            "intervention_excluded": "NPS wolf relocation beginning September 2018",
        },
    )
