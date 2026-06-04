"""从 LTER 威斯康星鱼类丰度 CSV 提取捕食者—猎物对。"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .common import normalize_time_years, read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries

DEFAULT_PAIR = ("Bluegill", "Largemouth Bass")
DEFAULT_LAKE = 804600  # WIfishAbundance WBIC，与 calibrate_bda 中 wifish 序列一致


def load_lter_fish_pair(
    path: str | None = None,
    lake_id: int = DEFAULT_LAKE,
    prey_species: str = DEFAULT_PAIR[0],
    predator_species: str = DEFAULT_PAIR[1],
    gear: str | None = "Boat Electrofishing",
) -> PredatorPreySeries:
    """
    读取 WIfishAbundance.csv，按湖×年聚合 CPUE（猎物、捕食者）。

    列: WBIC, YEAR, GEARNAME, CPUE, LOGCPUE, N, taxon_id
    """
    csv_path = resolve_data_file(
        path or "07_lter_fish/WIfishAbundance.csv",
    )
    rows = read_csv_dicts(csv_path)

    by_year: dict[int, dict[str, float]] = defaultdict(dict)
    for r in rows:
        try:
            lake = int(float(r["WBIC"]))
            year = int(float(r["YEAR"]))
        except ValueError:
            continue
        if lake != lake_id:
            continue
        if gear and r.get("GEARNAME", "") != gear:
            continue
        sp = r.get("taxon_id", "").strip()
        if sp not in (prey_species, predator_species):
            continue
        cpue_raw = r.get("CPUE", "")
        if cpue_raw in ("", "NA", "NaN", "nan", None):
            continue
        try:
            cpue = float(cpue_raw)
        except ValueError:
            continue
        prev = by_year[year].get(sp)
        by_year[year][sp] = cpue if prev is None else 0.5 * (prev + cpue)

    years = sorted(y for y, d in by_year.items() if prey_species in d and predator_species in d)
    if len(years) < 4:
        raise ValueError(
            f"湖 {lake_id} 上 {prey_species}/{predator_species} 同期记录仅 {len(years)} 年"
        )

    year_arr = np.array(years, dtype=float)
    prey = np.array([by_year[y][prey_species] for y in years])
    predator = np.array([by_year[y][predator_species] for y in years])
    t = normalize_time_years(year_arr)

    return PredatorPreySeries(
        name=f"lter_fish_{lake_id}_{prey_species}_vs_{predator_species}".replace(" ", "_"),
        t=t,
        prey=prey,
        predator=predator,
        time_unit="year",
        prey_label=prey_species,
        predator_label=predator_species,
        source_path=str(csv_path),
        meta={
            "lake_id": lake_id,
            "gear": gear,
            "year_start": years[0],
            "year_end": years[-1],
            "n_years": len(years),
        },
    )


if __name__ == "__main__":
    s = load_lter_fish_pair()
    print(s.name, s.n_points, s.meta)
