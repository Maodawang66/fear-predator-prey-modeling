"""数据路径解析、CSV 读取与有效性检查。"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parent
RAW = DATA_ROOT / "raw"
BUNDLED = DATA_ROOT / "bundled"
PROJECT_ROOT = DATA_ROOT.parent


class InvalidDataFileError(FileNotFoundError):
    """CSV 缺失或被 Dryad 反爬页面替换。"""


def is_valid_csv(path: Path) -> bool:
    if not path.is_file():
        return False
    head = path.read_bytes()[:256].lstrip()
    if head.startswith(b"<!") or b"<html" in head[:64].lower():
        return False
    if head.startswith(b"PK"):
        return False
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return False
    return "," in lines[0] or "\t" in lines[0]


def resolve_data_file(*candidates: str | Path) -> Path:
    """
    按优先级查找数据文件：raw/ → bundled/ → 绝对路径。
    若找到但内容为 HTML/无效，抛出 InvalidDataFileError。
    """
    tried: list[str] = []
    for cand in candidates:
        p = Path(cand)
        if not p.is_absolute():
            for base in (RAW, BUNDLED, DATA_ROOT):
                candidate = base / p
                tried.append(str(candidate))
                if candidate.is_file():
                    if not is_valid_csv(candidate):
                        raise InvalidDataFileError(
                            f"文件存在但不是有效 CSV（可能被 Dryad 反爬替换为 HTML）：\n  {candidate}\n"
                            "请浏览器手动下载后放到 data/bundled/ 同名路径，再重试。"
                        )
                    return candidate
        else:
            tried.append(str(p))
            if p.is_file():
                if not is_valid_csv(p):
                    raise InvalidDataFileError(f"无效 CSV: {p}")
                return p
    raise FileNotFoundError(
        "未找到数据文件，已尝试：\n  " + "\n  ".join(tried)
    )


def read_csv_dicts(path: Path, delimiter: str | None = None) -> list[dict[str, str]]:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    if delimiter is None:
        delimiter = "\t" if text.splitlines()[0].count("\t") > text.splitlines()[0].count(",") else ","
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"无法解析表头: {path}")
        for row in reader:
            rows.append({(k or "").strip(): (v or "").strip() for k, v in row.items()})
    if not rows:
        raise ValueError(f"CSV 无数据行: {path}")
    return rows


def pick_column(row: dict[str, str], *patterns: str) -> str | None:
    keys = list(row.keys())
    for pat in patterns:
        rx = re.compile(pat, re.I)
        for k in keys:
            if rx.search(k):
                return k
    return None


def col_to_float(row: dict[str, str], *patterns: str, default: float | None = None) -> float:
    key = pick_column(row, *patterns)
    if key is None:
        if default is not None:
            return default
        raise KeyError(f"未找到列（模式 {patterns}），可用列: {list(row.keys())}")
    val = row[key].strip()
    if val in ("", "NA", "NaN", "nan", "NULL"):
        if default is not None:
            return default
        raise ValueError(f"列 {key} 为空")
    return float(val)


def normalize_time_years(years: np.ndarray, month_fraction: np.ndarray | None = None) -> np.ndarray:
    """将年份（+ 可选月内小数）转为从 0 开始的连续时间。"""
    t = np.asarray(years, dtype=float)
    if month_fraction is not None:
        t = t + np.asarray(month_fraction, dtype=float)
    t = t - t[0]
    return t
