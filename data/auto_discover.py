"""
自动扫描 data/raw 与 data/bundled，识别可拟合的捕食者—猎物时间序列。

不依赖用户指定 CSV 路径或列名；结合文件名、表头关键词与生态学先验配对。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from .common import BUNDLED, RAW, is_valid_csv, read_csv_dicts
from .dataset_registry import is_ode_fit_path
from .series import PredatorPreySeries

SKIP_NAME_PARTS = (
    "download_instructions",
    "readme",
    "source.txt",
    "manifest",
    "instructions",
    "dataset.json",
)

TIME_PATTERNS = (
    r"^year$",
    r"^yr$",
    r"^time$",
    r"^t$",
    r"^date$",
    r"^dateseq$",
    r"dateseq",
    r"^seq$",
    r"year",
    r"month",
    r"^mon$",
    r"period",
    r"week",
)

PREY_PATTERNS = (
    r"hare",
    r"prey",
    r"victim",
    r"roe",
    r"deer",
    r"het",
    r"killifish",
    r"daphnia",
    r"zoop",
    r"herbiv",
    r"bluegill",
    r"perch",
    r"minnow",
    r"prey_mean",
    r"harvest_mean",
)

PREDATOR_PATTERNS = (
    r"lynx",
    r"predator",
    r"pred_",
    r"gamb",
    r"mosquito",
    r"bass",
    r"pike",
    r"wolf",
    r"family_group",
    r"bythotrephes",
)

# 长表鱼类：物种对 + 猎物通常更常见/数量级
FISH_PREDATOR_PREY_PAIRS: tuple[tuple[str, str], ...] = (
    ("Bluegill", "Largemouth Bass"),
    ("Yellow Perch", "Walleye"),
    ("Black Crappie", "Largemouth Bass"),
    ("Pumpkinseed", "Largemouth Bass"),
    ("Bluegill", "Northern Pike"),
)

# 路径/文件名签名 → 专用解析（优先级最高）
# Andrén 猞猁–狍 stacked 表无 region 列时的固定行块长度
LYNX_ROE_REGION_YEARS = {1: 29, 2: 29, 3: 29, 4: 29, 5: 29, 6: 27, 7: 24}

PATH_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("lynxhare", "lynx_hare"),
    ("lynx_roe", "lynx_roe"),
    ("killifish", "killifish"),
    ("mosquitofish", "killifish"),
    ("wifishabundance", "lter_fish"),
    ("fishabundance", "lter_fish"),
    ("zoop", "zooplankton"),
    ("glerl", "zooplankton"),
)


@dataclass
class ColumnMapping:
    time_col: str | None = None
    month_col: str | None = None
    prey_col: str | None = None
    predator_col: str | None = None
    value_col: str | None = None
    species_col: str | None = None
    site_cols: list[str] = field(default_factory=list)
    group_col: str | None = None
    gear_col: str | None = None
    gear_filter: str | None = None
    transform_prey: str | None = None
    transform_predator: str | None = None
    confidence: float = 0.0
    method: str = ""


@dataclass
class DetectedCandidate:
    path: Path
    signature: str
    mapping: ColumnMapping
    group_key: str | None = None
    prey_label: str = "prey"
    predator_label: str = "predator"
    notes: str = ""


def _match_col(name: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, name, re.I) for p in patterns)


def _numeric_columns(rows: list[dict[str, str]]) -> list[str]:
    if not rows:
        return []
    numeric: list[str] = []
    for key in rows[0]:
        vals = [r.get(key, "") for r in rows[: min(50, len(rows))]]
        ok = 0
        for v in vals:
            v = v.strip()
            if v in ("", "NA", "NaN", "nan", "NULL", "None"):
                continue
            try:
                float(v)
                ok += 1
            except ValueError:
                break
        if ok >= max(3, len(vals) // 3):
            numeric.append(key)
    return numeric


def _find_time_col(keys: list[str]) -> str | None:
    for k in keys:
        if _match_col(k, TIME_PATTERNS):
            return k
    return None


def _find_month_col(keys: list[str]) -> str | None:
    for k in keys:
        if re.search(r"^month$|^mon$", k, re.I):
            return k
    return None


def _classify_population_cols(keys: list[str]) -> tuple[str | None, str | None]:
    prey = pred = None
    for k in keys:
        if _match_col(k, PREY_PATTERNS) and not _match_col(k, PREDATOR_PATTERNS):
            prey = k
        if _match_col(k, PREDATOR_PATTERNS):
            pred = k
    if prey and pred:
        return prey, pred
    # 恰好两列数值且列名含 classic names
    pop_cols = [k for k in keys if _match_col(k, PREY_PATTERNS + PREDATOR_PATTERNS)]
    if len(pop_cols) >= 2:
        p_prey = next((c for c in pop_cols if _match_col(c, PREY_PATTERNS) and not _match_col(c, PREDATOR_PATTERNS)), None)
        p_pred = next((c for c in pop_cols if _match_col(c, PREDATOR_PATTERNS)), None)
        if p_prey and p_pred:
            return p_prey, p_pred
    return prey, pred


def _infer_signature(path: Path) -> str:
    blob = str(path).lower()
    for needle, sig in PATH_SIGNATURES:
        if needle in blob:
            return sig
    return "generic"


def _parse_float(val: str) -> float | None:
    v = val.strip()
    if v in ("", "NA", "NaN", "nan", "NULL", "None"):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def _apply_transform(values: np.ndarray, transform: str | None) -> np.ndarray:
    if transform == "log10":
        return np.power(10.0, values)
    if transform == "exp":
        return np.exp(values)
    return values


def discover_csv_paths() -> list[Path]:
    paths: list[Path] = []
    for root in (RAW, BUNDLED):
        if not root.is_dir():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            low = p.name.lower()
            if any(x in low for x in SKIP_NAME_PARTS):
                continue
            if p.suffix.lower() not in (".csv", ".txt", ".tsv"):
                continue
            if not is_valid_csv(p):
                continue
            if not is_ode_fit_path(p):
                continue
            paths.append(p)
    return paths


def _detect_lynx_hare(path: Path, rows: list[dict[str, str]]) -> list[DetectedCandidate]:
    keys = list(rows[0].keys())
    time_col = _find_time_col(keys) or "year"
    prey_col = next((k for k in keys if re.search(r"hare|prey|roe", k, re.I)), None)
    pred_col = next((k for k in keys if re.search(r"lynx|pred", k, re.I)), None)
    if not prey_col or not pred_col:
        nums = _numeric_columns(rows)
        if len(nums) >= 2:
            prey_col, pred_col = nums[0], nums[1]
        else:
            return []
    m = ColumnMapping(
        time_col=time_col,
        prey_col=prey_col,
        predator_col=pred_col,
        confidence=0.95,
        method="lynx_hare_signature",
    )
    return [
        DetectedCandidate(
            path=path,
            signature="lynx_hare",
            mapping=m,
            prey_label=prey_col,
            predator_label=pred_col,
            notes="classic hudson bay or lynx-roedeer style",
        )
    ]


def _detect_lynx_roe(path: Path, rows: list[dict[str, str]]) -> list[DetectedCandidate]:
    keys = list(rows[0].keys())
    time_col = _find_time_col(keys)
    if not time_col:
        return []
    prey_col = next((k for k in keys if re.search(r"roe.*harvest|harvest.*mean|roe_deer", k, re.I)), None)
    pred_col = next((k for k in keys if re.search(r"lynx.*family|family.*group", k, re.I)), None)
    area_col = next((k for k in keys if re.search(r"^area$|area_km", k, re.I)), None)
    region_col = next((k for k in keys if re.search(r"^region$|area_id", k, re.I)), None)
    if not prey_col or not pred_col:
        return []

    out: list[DetectedCandidate] = []
    if region_col:
        regions = sorted({r.get(region_col, "") for r in rows if r.get(region_col, "")})
        for reg in regions:
            m = ColumnMapping(
                time_col=time_col,
                prey_col=prey_col,
                predator_col=pred_col,
                group_col=region_col,
                transform_prey="divide_area" if area_col else None,
                transform_predator="divide_area" if area_col else None,
                confidence=0.92,
                method="lynx_roe_by_region",
            )
            out.append(
                DetectedCandidate(
                    path=path,
                    signature="lynx_roe",
                    mapping=m,
                    group_key=str(reg),
                    prey_label=f"roe_deer_r{reg}",
                    predator_label=f"lynx_r{reg}",
                    notes=f"region={reg}, density via area" if area_col else f"region={reg}",
                )
            )
    else:
        for reg in range(1, 8):
            m = ColumnMapping(
                time_col=time_col,
                prey_col=prey_col,
                predator_col=pred_col,
                transform_prey="divide_area" if area_col else None,
                transform_predator="divide_area" if area_col else None,
                confidence=0.88,
                method="lynx_roe_stacked_region",
            )
            out.append(
                DetectedCandidate(
                    path=path,
                    signature="lynx_roe",
                    mapping=m,
                    group_key=str(reg),
                    prey_label=f"roe_deer_r{reg}",
                    predator_label=f"lynx_r{reg}",
                    notes=f"stacked rows region {reg}",
                )
            )
    return out


def _detect_killifish(path: Path, rows: list[dict[str, str]]) -> list[DetectedCandidate]:
    keys = list(rows[0].keys())
    time_col = _find_time_col(keys)
    month_col = _find_month_col(keys)
    if not time_col:
        return []

    prey_col = next((k for k in keys if re.search(r"het|killifish|prey", k, re.I) and re.search(r"mean|log", k, re.I)), None)
    pred_col = next((k for k in keys if re.search(r"gamb|mosquito|pred", k, re.I) and re.search(r"mean|log", k, re.I)), None)
    if not prey_col or not pred_col:
        nums = _numeric_columns(rows)
        if len(nums) >= 2:
            prey_col, pred_col = nums[-2], nums[-1]

    loc = next((k for k in keys if re.search(r"^location$|^loc$", k, re.I)), None)
    site = next((k for k in keys if re.search(r"^site$", k, re.I)), None)
    transform = "log10" if prey_col and "log" in prey_col.lower() else None

    def site_key(r: dict[str, str]) -> str:
        if loc and site:
            return f"{r.get(loc, '')}_{r.get(site, '')}"
        if loc:
            return r.get(loc, "")
        if site:
            return r.get(site, "")
        return "all"

    sites = sorted({site_key(r) for r in rows if site_key(r)})
    out: list[DetectedCandidate] = []
    for sk in sites:
        if not sk or sk == "all":
            continue
        site_cols = [c for c in (loc, site) if c]
        m = ColumnMapping(
            time_col=time_col,
            month_col=month_col,
            prey_col=prey_col,
            predator_col=pred_col,
            site_cols=site_cols,
            transform_prey=transform,
            transform_predator=transform,
            confidence=0.9,
            method="killifish_by_site",
        )
        out.append(
            DetectedCandidate(
                path=path,
                signature="killifish",
                mapping=m,
                group_key=sk,
                prey_label="Heterandria",
                predator_label="Gambusia",
                notes=f"site={sk}, log->linear" if transform else f"site={sk}",
            )
        )
    if not out and prey_col and pred_col:
        m = ColumnMapping(
            time_col=time_col,
            month_col=month_col,
            prey_col=prey_col,
            predator_col=pred_col,
            transform_prey=transform,
            transform_predator=transform,
            confidence=0.8,
            method="killifish_all",
        )
        out.append(
            DetectedCandidate(
                path=path,
                signature="killifish",
                mapping=m,
                prey_label="Heterandria",
                predator_label="Gambusia",
            )
        )
    return out


def _detect_zooplankton(path: Path, rows: list[dict[str, str]]) -> list[DetectedCandidate]:
    keys = list(rows[0].keys())
    prey_col = next((k for k in keys if re.search(r"d\.?mendotae|daphnia", k, re.I)), None)
    pred_col = next((k for k in keys if re.search(r"bythotrephes", k, re.I)), None)
    time_col = next((k for k in keys if re.search(r"^date$", k, re.I)), None)
    if not time_col:
        time_col = next((k for k in keys if re.search(r"^julianday$", k, re.I)), None)
    if not time_col:
        time_col = _find_time_col(keys)
    if not (prey_col and pred_col and time_col):
        return []

    m = ColumnMapping(
        time_col=time_col,
        prey_col=prey_col,
        predator_col=pred_col,
        confidence=0.93,
        method="zooplankton_glerl",
    )
    return [
        DetectedCandidate(
            path=path,
            signature="zooplankton",
            mapping=m,
            prey_label="D.mendotae",
            predator_label="Bythotrephes",
            notes="Marino et al. Lake Michigan CE/NCE monitoring",
        )
    ]


def _detect_lter_fish(path: Path, rows: list[dict[str, str]]) -> list[DetectedCandidate]:
    keys = list(rows[0].keys())
    if not any(re.search(r"taxon|species", k, re.I) for k in keys):
        return []
    species_col = next(k for k in keys if re.search(r"taxon|species", k, re.I))
    year_col = next((k for k in keys if re.search(r"^year$", k, re.I)), None)
    lake_col = next((k for k in keys if re.search(r"wbic|lake|site", k, re.I)), None)
    value_col = next((k for k in keys if re.search(r"^cpue$|abundance|count|density", k, re.I)), None)
    gear_col = next((k for k in keys if re.search(r"gear", k, re.I)), None)
    if not year_col or not value_col or not lake_col:
        return []

    preferred_gear = "Boat Electrofishing"
    gears_seen = {r.get(gear_col, "").strip() for r in rows if gear_col}
    use_gear = preferred_gear if preferred_gear in gears_seen else None

    # (lake, year, species) 一次扫描建索引
    presence: set[tuple[str, int, str]] = set()
    species_present: set[str] = set()
    for r in rows:
        if use_gear and r.get(gear_col, "") != use_gear:
            continue
        sp = r.get(species_col, "").strip()
        if not sp:
            continue
        species_present.add(sp)
        lake = r.get(lake_col, "").strip()
        y = _parse_float(r.get(year_col, ""))
        if not lake or y is None:
            continue
        presence.add((lake, int(y), sp))

    out: list[DetectedCandidate] = []
    for prey_sp, pred_sp in FISH_PREDATOR_PREY_PAIRS:
        if prey_sp not in species_present or pred_sp not in species_present:
            continue
        lake_years: dict[str, int] = {}
        lakes = {lake for lake, _, sp in presence if sp in (prey_sp, pred_sp)}
        for lake in lakes:
            years_both = {
                yr
                for lake2, yr, sp in presence
                if lake2 == lake
                and sp == prey_sp
                and (lake, yr, pred_sp) in presence
            }
            if len(years_both) >= 4:
                lake_years[lake] = len(years_both)

        for lake, n_years in sorted(lake_years.items(), key=lambda x: -x[1])[:2]:
            m = ColumnMapping(
                time_col=year_col,
                prey_col=prey_sp,
                predator_col=pred_sp,
                value_col=value_col,
                species_col=species_col,
                site_cols=[lake_col],
                gear_col=gear_col,
                gear_filter=use_gear,
                confidence=0.88,
                method="lter_fish_pair",
            )
            out.append(
                DetectedCandidate(
                    path=path,
                    signature="lter_fish",
                    mapping=m,
                    group_key=f"{lake_col}={lake}",
                    prey_label=prey_sp,
                    predator_label=pred_sp,
                    notes=f"{prey_sp} vs {pred_sp}, lake={lake}, n_years={n_years}",
                )
            )

    out.sort(key=lambda c: int(re.search(r"n_years=(\d+)", c.notes or "").group(1)) if re.search(r"n_years=(\d+)", c.notes or "") else 0, reverse=True)
    seen_pair: set[tuple[str, str]] = set()
    deduped: list[DetectedCandidate] = []
    for c in out:
        key = (c.prey_label, c.predator_label)
        if key in seen_pair:
            continue
        seen_pair.add(key)
        deduped.append(c)
    return deduped[:3]


def _detect_generic(path: Path, rows: list[dict[str, str]]) -> list[DetectedCandidate]:
    keys = list(rows[0].keys())
    time_col = _find_time_col(keys)
    month_col = _find_month_col(keys)
    prey_col, pred_col = _classify_population_cols(keys)
    nums = _numeric_columns(rows)

    if time_col and prey_col and pred_col:
        conf = 0.7
    elif time_col and len(nums) >= 2:
        # 两列数值：按均值大的为猎物（经典 LV 数据）
        m0, m1 = nums[0], nums[1]
        mean0 = np.nanmean([_parse_float(r.get(m0, "")) for r in rows if _parse_float(r.get(m0, "")) is not None])
        mean1 = np.nanmean([_parse_float(r.get(m1, "")) for r in rows if _parse_float(r.get(m1, "")) is not None])
        if mean0 >= mean1:
            prey_col, pred_col = m0, m1
        else:
            prey_col, pred_col = m1, m0
        conf = 0.55
    else:
        return []

    m = ColumnMapping(
        time_col=time_col,
        month_col=month_col,
        prey_col=prey_col,
        predator_col=pred_col,
        confidence=conf,
        method="generic_heuristic",
    )
    return [
        DetectedCandidate(
            path=path,
            signature="generic",
            mapping=m,
            prey_label=prey_col or "col_prey",
            predator_label=pred_col or "col_pred",
            notes="auto: keyword or mean-magnitude assignment",
        )
    ]


def detect_candidates(path: Path) -> list[DetectedCandidate]:
    if not is_ode_fit_path(path):
        return []
    try:
        rows = read_csv_dicts(path)
    except (ValueError, OSError):
        return []

    sig = _infer_signature(path)
    if sig == "lynx_hare":
        cands = _detect_lynx_hare(path, rows)
    elif sig == "lynx_roe":
        cands = _detect_lynx_roe(path, rows)
    elif sig == "killifish":
        cands = _detect_killifish(path, rows)
    elif sig == "zooplankton":
        cands = _detect_zooplankton(path, rows)
    elif sig == "lter_fish":
        cands = _detect_lter_fish(path, rows)
    else:
        cands = _detect_generic(path, rows)

    if not cands and sig != "generic":
        cands = _detect_generic(path, rows)
    return cands


def discover_all_candidates() -> list[DetectedCandidate]:
    all_cands: list[DetectedCandidate] = []
    for path in discover_csv_paths():
        all_cands.extend(detect_candidates(path))
    return all_cands


def _lynx_roe_stacked_rows(
    rows: list[dict[str, str]], region: int, time_col: str
) -> list[dict[str, str]]:
    offset = sum(LYNX_ROE_REGION_YEARS[i] for i in range(1, region))
    n = LYNX_ROE_REGION_YEARS[region]
    sub = rows[offset : offset + n]
    return sorted(sub, key=lambda x: float(_parse_float(x.get(time_col, "")) or 0))


def _lake_matches(row_val: str, target: str) -> bool:
    rv = row_val.strip()
    tv = target.strip()
    if rv == tv:
        return True
    try:
        return int(float(rv)) == int(float(tv))
    except ValueError:
        return False


def _lter_fish_synthetic_rows(
    rows: list[dict[str, str]], cand: DetectedCandidate
) -> list[dict[str, str]]:
    m = cand.mapping
    lake_col = m.site_cols[0]
    lake_val = (cand.group_key or "").split("=", 1)[-1]
    prey, pred = cand.prey_label, cand.predator_label
    by_year: dict[int, dict[str, float]] = {}
    for r in rows:
        if not _lake_matches(r.get(lake_col, ""), lake_val):
            continue
        if m.gear_col and m.gear_filter and r.get(m.gear_col, "") != m.gear_filter:
            continue
        sp = r.get(m.species_col or "", "").strip()
        if sp not in (prey, pred):
            continue
        y = _parse_float(r.get(m.time_col or "", ""))
        val = _parse_float(r.get(m.value_col or "", ""))
        if y is None or val is None:
            continue
        yi = int(y)
        prev = by_year.get(yi, {}).get(sp)
        by_year.setdefault(yi, {})[sp] = val if prev is None else 0.5 * (prev + val)

    synthetic: list[dict[str, str]] = []
    for yi in sorted(by_year):
        if prey in by_year[yi] and pred in by_year[yi]:
            synthetic.append(
                {
                    m.time_col or "year": str(yi),
                    prey: str(by_year[yi][prey]),
                    pred: str(by_year[yi][pred]),
                }
            )
    return synthetic


def _rows_for_candidate(rows: list[dict[str, str]], cand: DetectedCandidate) -> list[dict[str, str]]:
    m = cand.mapping
    out = rows
    if (
        cand.signature == "lynx_roe"
        and m.method == "lynx_roe_stacked_region"
        and cand.group_key
        and m.group_col is None
    ):
        return _lynx_roe_stacked_rows(out, int(cand.group_key), m.time_col or "year")
    if cand.signature == "lter_fish" and m.site_cols and cand.group_key:
        return _lter_fish_synthetic_rows(rows, cand)
    if cand.group_key and m.group_col:
        out = [r for r in rows if r.get(m.group_col, "") == cand.group_key]
    elif cand.group_key and m.site_cols:
        loc, site = (m.site_cols + [None, None])[:2]
        if loc and site:
            parts = cand.group_key.split("_", 1)
            if len(parts) == 2:
                out = [r for r in rows if r.get(loc, "") == parts[0] and r.get(site, "") == parts[1]]
        elif loc and not site:
            out = [r for r in rows if r.get(loc, "") == cand.group_key]
        elif site:
            out = [r for r in rows if r.get(site, "") == cand.group_key]
    return out


def load_candidate(cand: DetectedCandidate) -> PredatorPreySeries:
    rows = read_csv_dicts(cand.path)
    sub = _rows_for_candidate(rows, cand)
    m = cand.mapping
    if len(sub) < 4:
        raise ValueError(f"{cand.path.name}: 有效点不足 ({len(sub)})")

    times: list[float] = []
    prey_vals: list[float] = []
    pred_vals: list[float] = []

    area_col = next((k for k in sub[0] if re.search(r"^area$", k, re.I)), None)

    def parse_time(row: dict[str, str]) -> float | None:
        raw_time = row.get(m.time_col or "", "")
        if cand.signature == "zooplankton" and m.time_col and m.time_col.lower() == "date":
            try:
                return datetime.strptime(raw_time.strip(), "%m/%d/%y").toordinal() / 365.25
            except ValueError:
                return None
        return _parse_float(raw_time)

    for r in sorted(sub, key=lambda row: parse_time(row) or 0.0):
        t = parse_time(r)
        if t is None:
            continue
        if m.month_col:
            mo = _parse_float(r.get(m.month_col, ""))
            if mo is not None:
                t = t + (mo - 1.0) / 12.0

        if cand.signature == "lter_fish":
            pv = _parse_float(r.get(cand.prey_label, ""))
            qv = _parse_float(r.get(cand.predator_label, ""))
        else:
            pv = _parse_float(r.get(m.prey_col or "", ""))
            qv = _parse_float(r.get(m.predator_col or "", ""))

        if pv is None or qv is None:
            continue

        if m.transform_prey == "log10":
            pv = float(np.power(10.0, pv))
        if m.transform_predator == "log10":
            qv = float(np.power(10.0, qv))
        if m.transform_prey == "divide_area" and area_col:
            a = _parse_float(r.get(area_col, ""))
            if a and a > 0:
                pv /= a
        if m.transform_predator == "divide_area" and area_col:
            a = _parse_float(r.get(area_col, ""))
            if a and a > 0:
                qv /= a

        times.append(t)
        prey_vals.append(pv)
        pred_vals.append(qv)

    if len(times) < 4:
        raise ValueError(f"{cand.path.name}: 解析后点数不足")

    t_arr = np.array(times, dtype=float)
    t_arr = t_arr - t_arr[0]
    prey = np.array(prey_vals, dtype=float)
    pred = np.array(pred_vals, dtype=float)

    slug = cand.path.stem.lower().replace(" ", "_")[:24]
    g_suffix = f"_{cand.group_key}".replace("=", "").replace("/", "_")[:20] if cand.group_key else ""
    name = f"{slug}{g_suffix}"

    return PredatorPreySeries(
        name=name,
        t=t_arr,
        prey=prey,
        predator=pred,
        time_unit="year",
        prey_label=cand.prey_label,
        predator_label=cand.predator_label,
        source_path=str(cand.path),
        meta={
            "signature": cand.signature,
            "detection_method": m.method,
            "confidence": m.confidence,
            "time_col": m.time_col,
            "prey_col": m.prey_col or cand.prey_label,
            "predator_col": m.predator_col or cand.predator_label,
            "group_key": cand.group_key,
            "notes": cand.notes,
            "n_points": len(times),
        },
    )


def discover_and_load(min_confidence: float = 0.5) -> list[PredatorPreySeries]:
    """发现全部候选并加载为可拟合序列（过滤低置信度与加载失败项）。"""
    series_list: list[PredatorPreySeries] = []
    seen_names: set[str] = set()

    for cand in discover_all_candidates():
        if cand.mapping.confidence < min_confidence:
            continue
        try:
            s = load_candidate(cand)
        except (ValueError, KeyError, TypeError):
            continue
        if s.name in seen_names:
            continue
        seen_names.add(s.name)
        series_list.append(s)

    return series_list
