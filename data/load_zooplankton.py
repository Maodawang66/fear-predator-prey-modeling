"""加载密歇根湖浮游动物 GLERL 监测数据（Daphnia vs Bythotrephes）。"""

from __future__ import annotations

import numpy as np

from .common import read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries

DEFAULT_PATH = "04_lake_michigan_zooplankton/GLERL_M110_Zoop_1994-2012.txt"
PREY_COL = "D.mendotae"
PRED_COL = "Bythotrephes"
TIME_COL = "JulianDay"


def load_zooplankton(path: str | None = None) -> PredatorPreySeries:
    """
    猎物 = D.mendotae 丰度；捕食者 = Bythotrephes 丰度。
    时间 = JulianDay（1994 年起算日序，归一化后用于 ODE 拟合）。
    """
    data_path = resolve_data_file(path or DEFAULT_PATH)
    rows = read_csv_dicts(data_path)

    times: list[float] = []
    prey: list[float] = []
    pred: list[float] = []
    for r in sorted(rows, key=lambda x: float(x.get(TIME_COL, 0) or 0)):
        t = float(r[TIME_COL])
        pv = float(r[PREY_COL])
        qv = float(r[PRED_COL])
        times.append(t)
        prey.append(pv)
        pred.append(qv)

    t_arr = np.array(times, dtype=float)
    t_arr = t_arr - t_arr[0]

    return PredatorPreySeries(
        name="lake_michigan_daphnia_bythotrephes",
        t=t_arr,
        prey=np.array(prey, dtype=float),
        predator=np.array(pred, dtype=float),
        time_unit="day",
        prey_label="D.mendotae",
        predator_label="Bythotrephes",
        source_path=str(data_path),
        meta={
            "time_col": TIME_COL,
            "prey_col": PREY_COL,
            "predator_col": PRED_COL,
            "nce_flag_col": "Prior_to_1st_Daphnia_obs",
            "signature": "zooplankton",
        },
    )


if __name__ == "__main__":
    s = load_zooplankton()
    print(s.name, s.n_points, s.prey_label, "vs", s.predator_label)
