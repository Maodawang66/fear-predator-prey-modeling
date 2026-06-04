"""加载哈德逊湾猞猁–雪兔经典年度数据。"""

from __future__ import annotations

import numpy as np

from .common import normalize_time_years, read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries


def load_lynx_hare(path: str | None = None) -> PredatorPreySeries:
    """
    读取 lynxhare.csv（year, hare, lynx）。
    猎物=雪兔，捕食者=猞猁；时间为年，从 0 起算。
    """
    csv_path = resolve_data_file(
        path or "03_hudson_bay_lynx_hare/lynxhare.csv",
    )
    rows = read_csv_dicts(csv_path)
    years = np.array([float(r["year"]) for r in rows])
    hare = np.array([float(r["hare"]) for r in rows])
    lynx = np.array([float(r["lynx"]) for r in rows])
    t = normalize_time_years(years)
    return PredatorPreySeries(
        name="hudson_bay_lynx_hare",
        t=t,
        prey=hare,
        predator=lynx,
        time_unit="year",
        prey_label="hare",
        predator_label="lynx",
        source_path=str(csv_path),
        meta={"year_start": int(years[0]), "year_end": int(years[-1]), "n_years": len(years)},
    )


if __name__ == "__main__":
    s = load_lynx_hare()
    print(s.name, s.n_points, "points", s.meta)
