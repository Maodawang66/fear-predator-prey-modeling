"""数据集目录：读取 dataset_catalog.json，同步各 raw 子目录标注。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import RAW

CATALOG_PATH = Path(__file__).resolve().parent / "dataset_catalog.json"

# 不参与 ODE 自动发现的目录（避免 generic 误识别列）
NON_ODE_DIRS = frozenset(
    {
        "05_gpdd",
        "08_coral_reef_fear",
        "09_landscape_of_fear",
        "10_damselfly_predator_cues",
        "12_peacor_risk_meta",
    }
)


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def folder_id_for_path(path: Path) -> str | None:
    try:
        rel = path.relative_to(RAW)
    except ValueError:
        return None
    parts = rel.parts
    return parts[0] if parts else None


def is_ode_fit_path(path: Path) -> bool:
    """该文件是否应进入 calibrate_bda 自动 ODE 拟合扫描。"""
    fid = folder_id_for_path(path)
    if fid is None:
        return True
    if fid in NON_ODE_DIRS:
        return False
    cat = load_catalog()
    entry = cat.get(fid, {})
    if not entry.get("fit_ode", True):
        return False
    primary = entry.get("primary_file")
    if primary and path.name != primary:
        # 同目录非主文件：仅当明确列为 population 才扫描（目前无）
        return path.name == primary
    return True


def annotate_raw_folders(catalog: dict[str, Any] | None = None) -> list[Path]:
    """将 catalog 条目写入 data/raw/<id>/dataset.json。"""
    catalog = catalog or load_catalog()
    written: list[Path] = []
    for folder_id, entry in catalog.items():
        out_dir = RAW / folder_id
        if not out_dir.is_dir():
            continue
        payload = {"id": folder_id, **entry}
        dest = out_dir / "dataset.json"
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        written.append(dest)
    return written


if __name__ == "__main__":
    paths = annotate_raw_folders()
    print(f"Wrote {len(paths)} dataset.json files:")
    for p in paths:
        print(f"  {p.relative_to(RAW.parent.parent)}")
