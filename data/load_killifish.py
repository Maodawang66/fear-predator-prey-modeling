"""加载 Killifish–Mosquitofish 月度对数密度数据。"""

from __future__ import annotations

import re

import numpy as np

from .common import col_to_float, pick_column, read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries

# 来自 Zenodo 10890432 FINAL.R：列 7/9 为 Heterandria / Gambusia 对数密度均值
# TimeSeriesLogMeans.csv 列名可能为 MEAN2LOGHETADS, MEAN4LOGGAMBO 等


def _site_key(row: dict[str, str]) -> str:
    loc = pick_column(row, r"^location$", r"^loc$")
    site = pick_column(row, r"^site$")
    if loc and site:
        return f"{row[loc]}_{row[site]}"
    if loc:
        return row[loc]
    if site:
        return row[site]
    raise KeyError(f"无法构造站点键，列: {list(row.keys())}")


def _time_from_row(row: dict[str, str]) -> float:
    dateseq = pick_column(row, r"^dateseq$", r"dateseq")
    if dateseq:
        return float(row[dateseq])
    year = col_to_float(row, r"^year$", r"year")
    month_key = pick_column(row, r"^month$", r"^mon$")
    if month_key:
        month = float(row[month_key])
        return year + (month - 1) / 12.0
    t_key = pick_column(row, r"^time$", r"^t$", r"date")
    if t_key:
        return float(row[t_key])
    return year


def load_killifish(
    path: str | None = None,
    site: str = "TP",
) -> PredatorPreySeries:
    """
    读取 TimeSeriesLogMeans.csv，提取单站点月度序列。

    猎物 = Least Killifish (Heterandria)，捕食者 = Mosquitofish (Gambusia)。
    观测值为 log10 密度，拟合前转为线性尺度 10**log。
    """
    csv_path = resolve_data_file(
        path or "02_killifish_mosquitofish/TimeSeriesLogMeans.csv",
    )
    rows = read_csv_dicts(csv_path)
    sample = rows[0]

    prey_log_key = pick_column(
        sample,
        r"mean.*log.*het",
        r"log.*het.*mean",
        r"mean2loghet",
        r"hetads",
    )
    pred_log_key = pick_column(
        sample,
        r"mean.*log.*gamb",
        r"log.*gamb",
        r"mean4loggambo",
        r"gambo",
    )
    if prey_log_key is None or pred_log_key is None:
        # 按 R 脚本列号 fallback（1-based → 0-based）
        keys = list(sample.keys())
        if len(keys) >= 9:
            prey_log_key = keys[6]
            pred_log_key = keys[8]
        else:
            raise KeyError(f"无法识别猎物/捕食者列，可用: {keys}")

    sub = [r for r in rows if _site_key(r) == site]
    if len(sub) < 4:
        available = sorted({_site_key(r) for r in rows})
        raise ValueError(
            f"站点 {site!r} 数据不足 ({len(sub)} 行)。可用站点: {available}"
        )

    sub.sort(key=_time_from_row)
    times = np.array([_time_from_row(r) for r in sub])
    prey = np.power(10.0, np.array([float(r[prey_log_key]) for r in sub]))
    predator = np.power(10.0, np.array([float(r[pred_log_key]) for r in sub]))
    t = times - times[0]

    return PredatorPreySeries(
        name=f"killifish_mosquitofish_{site}",
        t=t,
        prey=prey,
        predator=predator,
        time_unit="year",
        prey_label="Heterandria formosa (linear from log10)",
        predator_label="Gambusia holbrooki (linear from log10)",
        source_path=str(csv_path),
        meta={
            "site": site,
            "prey_log_column": prey_log_key,
            "predator_log_column": pred_log_key,
            "log_scale_input": True,
        },
    )


def list_killifish_sites(path: str | None = None) -> list[str]:
    csv_path = resolve_data_file(
        path or "02_killifish_mosquitofish/TimeSeriesLogMeans.csv",
    )
    rows = read_csv_dicts(csv_path)
    return sorted({_site_key(r) for r in rows})


if __name__ == "__main__":
    print("sites:", list_killifish_sites())
    s = load_killifish(site="TP")
    print(s.name, s.n_points, s.meta)
