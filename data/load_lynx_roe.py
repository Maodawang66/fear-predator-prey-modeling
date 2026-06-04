"""加载 Andrén et al. 欧亚猞猁–狍多区域数据。"""

from __future__ import annotations

import re

import numpy as np

from .common import col_to_float, normalize_time_years, read_csv_dicts, resolve_data_file
from .series import PredatorPreySeries

# Andrén & Liberg (2023) 模型：猞猁家族群数 / 总猞猁 ≈ 0.184
LYNX_FAMILY_TO_TOTAL = 0.184

REGION_YEARS = {1: 29, 2: 29, 3: 29, 4: 29, 5: 29, 6: 27, 7: 24}


def _infer_region_column(rows: list[dict[str, str]]) -> str | None:
    sample = rows[0]
    for pat in (r"^region$", r"region", r"^area_id$"):
        for k in sample:
            if re.search(pat, k, re.I):
                return k
    return None


def load_lynx_roe(
    path: str | None = None,
    region: int = 3,
    use_total_lynx: bool = True,
) -> PredatorPreySeries:
    """
    读取 Andren_lynx_roedeer_data.csv，提取单区域时间序列。

    列（Dryad 标准）：
    year, region, roe_deer_harvest_mean, lynx_family_groups, area, ...

    密度代理（per 1000 km²，与原文作图一致）：
    - 猎物 x = roe_deer_harvest_mean / area
    - 捕食者 y = lynx_family_groups / area（或 ×1/0.184 得总猞猁密度代理）
    """
    if region not in REGION_YEARS:
        raise ValueError(f"region 须为 1–7，收到 {region}")

    csv_path = resolve_data_file(
        path or "01_lynx_roe_deer/Andren_lynx_roedeer_data.csv",
    )
    rows = read_csv_dicts(csv_path)
    region_key = _infer_region_column(rows)

    if region_key is not None:
        sub = [r for r in rows if int(float(r[region_key])) == region]
    else:
        offset = sum(REGION_YEARS[i] for i in range(1, region))
        sub = rows[offset : offset + REGION_YEARS[region]]

    if len(sub) < 4:
        raise ValueError(f"区域 {region} 有效行数不足: {len(sub)}")

    years = np.array([col_to_float(r, r"^year$", r"year") for r in sub])
    roe = np.array([col_to_float(r, r"roe_deer_harvest_mean", r"roe.*mean", r"harvest_mean") for r in sub])
    lynx_fg = np.array([col_to_float(r, r"lynx_family_groups", r"lynx.*family") for r in sub])
    area = np.array([col_to_float(r, r"^area$", r"area_km") for r in sub])

    prey = roe / area
    predator = lynx_fg / area
    if use_total_lynx:
        predator = predator / LYNX_FAMILY_TO_TOTAL

    t = normalize_time_years(years)
    return PredatorPreySeries(
        name=f"lynx_roe_region_{region}",
        t=t,
        prey=prey,
        predator=predator,
        time_unit="year",
        prey_label="roe_deer (harvest/area)",
        predator_label="lynx (family_groups/area)" + ("" if not use_total_lynx else ", scaled to total"),
        source_path=str(csv_path),
        meta={
            "region": region,
            "year_start": int(years[0]),
            "year_end": int(years[-1]),
            "lynx_family_to_total": LYNX_FAMILY_TO_TOTAL if use_total_lynx else 1.0,
        },
    )


def load_all_lynx_roe_regions(path: str | None = None) -> list[PredatorPreySeries]:
    return [load_lynx_roe(path=path, region=r) for r in range(1, 8)]


if __name__ == "__main__":
    for r in (1, 3, 7):
        s = load_lynx_roe(region=r)
        print(s.name, s.n_points, s.meta)
