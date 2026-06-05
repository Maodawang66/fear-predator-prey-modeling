"""
基于现有 data/raw 与 calibration_bda 拟合结果的一、二档数据深挖分析。

用法（项目根目录）:
    conda activate ai25
    python data/deep_data_analysis.py
    python data/deep_data_analysis.py --fit-summary results/calibration_bda/fit_summary.csv

依赖: 先运行 python data/calibrate_bda.py 生成 fit_summary.csv

输出: results/deep_analysis/
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np

from data.auto_discover import discover_and_load
from data.common import RAW, read_csv_dicts
from data.load_lter_fish import load_lter_fish_pair
from data.load_peacor import load_peacor_plp
from src.fit import (
    fit_bda_fear_to_series,
    profile_bda_k,
)
from src.parameters import BDAFearParams

OUT = ROOT / "results" / "deep_analysis"

# k profile 代表序列（可在全 12 条上扩展）
PROFILE_REPRESENTATIVES = (
    "glerl_m110_zoop_1994-201",
    "lynxhare",
    "timeserieslogmeans_TP",
)

LTER_DEFAULT_LAKE = 804600

LTER_EXTRA_PAIRS = (
    ("Bluegill", "Largemouth Bass"),
    ("Pumkinseed", "Largemouth Bass"),  # LTER 原始拼写
    ("Smallmouth Bass", "Largemouth Bass"),
)


def _find_fit_summary(custom: Path | None) -> Path:
    if custom and custom.is_file():
        return custom
    for p in (
        ROOT / "results" / "calibration_bda" / "fit_summary.csv",
        ROOT / "results" / "results" / "calibration_bda" / "fit_summary.csv",
    ):
        if p.is_file():
            return p
    raise FileNotFoundError(
        "未找到 fit_summary.csv，请先运行: python data/calibrate_bda.py"
    )


def _load_fit_table(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str, default: float = float("nan")) -> float:
    v = row.get(key, "")
    if v in ("", None):
        return default
    try:
        return float(v)
    except ValueError:
        return default


def _is_usable_fit(row: dict) -> bool:
    status = row.get("optimization_status", "")
    if status:
        return status in ("success", "usable_limit")
    return str(row.get("success", "")).lower() == "true"


def _classify_group(series: str) -> str:
    s = series.lower()
    if "glerl" in s or "zoop" in s:
        return "zooplankton"
    if "timeserieslogmeans" in s or "killifish" in s:
        return "fish"
    if "andren" in s or "lynxhare" in s or "lynx" in s:
        return "mammal"
    return "other"


def _eta(k: float, v: float) -> float:
    if k <= 0 or v <= 0:
        return 0.0
    return k * v / (1.0 + k * v)


def _series_index(series_list) -> dict[str, object]:
    return {s.name: s for s in series_list}


def _write_csv(rows: list[dict], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_cross_system_table(rows: list[dict]) -> list[dict]:
    """从 fit_summary 构建跨系统 k、η 汇总表。"""
    bda = [
        r for r in rows
        if r.get("model") == "bda_fear" and _is_usable_fit(r)
    ]
    out: list[dict] = []
    for r in bda:
        name = r["series"]
        k = _f(r, "k")
        v0 = _f(r, "v0", 1.0)
        eta_v0 = _eta(k, v0)
        eta_v5 = _eta(k, 5.0)
        out.append({
            "series": name,
            "group": _classify_group(name),
            "group_key": r.get("group_key", ""),
            "k": k,
            "v0": v0,
            "eta_v0": eta_v0,
            "eta_v5": eta_v5,
            "rmse_normalized_total": _f(r, "rmse_normalized_total"),
            "rmse_normalized_prey": _f(r, "rmse_normalized_prey"),
            "rmse_normalized_predator": _f(r, "rmse_normalized_predator"),
            "p": _f(r, "p"),
            "q": _f(r, "q"),
            "r_param": _f(r, "r"),
        })
    return out


def build_rmse_improvement(rows: list[dict]) -> list[dict]:
    """baseline / fear_memory / bda_fear RMSE 与改进倍数。"""
    by_series: dict[str, dict[str, float]] = {}
    for r in rows:
        if not _is_usable_fit(r):
            continue
        s = r["series"]
        by_series.setdefault(s, {})[r["model"]] = _f(r, "rmse_normalized_total")

    out: list[dict] = []
    for s, models in sorted(by_series.items()):
        base = models.get("baseline", float("nan"))
        bda = models.get("bda_fear", float("nan"))
        mem = models.get("fear_memory", float("nan"))
        ratio = base / bda if bda > 0 and np.isfinite(bda) else float("nan")
        out.append({
            "series": s,
            "group": _classify_group(s),
            "rmse_normalized_baseline": base,
            "rmse_normalized_fear_memory": mem,
            "rmse_normalized_bda_fear": bda,
            "improvement_ratio": ratio,
        })
    return out


def _short_series_name(series: str) -> str:
    if series == "lynxhare":
        return "lynxhare"
    if series.startswith("andren_lynx_roedeer_data_"):
        return f"andren_{series.rsplit('_', 1)[-1]}"
    if series.startswith("glerl_"):
        return "glerl_zoop"
    if series.startswith("timeserieslogmeans_"):
        return series.replace("timeserieslogmeans_", "")
    return series[:32]


def plot_rmse_improvement(data: list[dict], path: Path) -> None:
    data = [d for d in data if np.isfinite(d.get("improvement_ratio", float("nan")))]
    data = sorted(data, key=lambda d: d.get("improvement_ratio", 0), reverse=True)
    names = [_short_series_name(d["series"]) for d in data]
    ratios = [d["improvement_ratio"] for d in data]
    group_colors = {"mammal": "#4C72B0", "fish": "#55A868", "zooplankton": "#C44E52"}
    bar_colors = [group_colors.get(d["group"], "#888888") for d in data]
    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(names))))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, ratios, color=bar_colors)
    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("RMSE improvement ratio (baseline / B-D+fear, log scale)")
    ax.set_title("Three-model fit: baseline vs B-D+fear RMSE improvement (12 series)")
    ax.axvline(1.0, color="gray", ls="--", lw=0.8)
    for i, (r, d) in enumerate(zip(ratios, data)):
        ax.text(r * 1.08, i, f"{r:.1e}", va="center", fontsize=7)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor=c, label=g) for g, c in group_colors.items()],
        loc="lower right",
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_eta_by_group(cross: list[dict], path: Path) -> None:
    groups = ["mammal", "fish", "zooplankton"]
    palette = {"mammal": "#4C72B0", "fish": "#55A868", "zooplankton": "#C44E52"}
    data = {g: [_eta(r["k"], 5.0) for r in cross if r["group"] == g] for g in groups}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax = axes[0]
    for i, g in enumerate(groups):
        ys = data[g]
        if not ys:
            continue
        xs = np.full(len(ys), i + 1) + np.linspace(-0.08, 0.08, len(ys))
        ax.scatter(xs, ys, c=palette[g], s=55, alpha=0.9, label=g, zorder=3)
        if len(ys) >= 2:
            bp = ax.boxplot([ys], positions=[i + 1], widths=0.45, showfliers=False, patch_artist=True)
            bp["boxes"][0].set_facecolor(palette[g])
            bp["boxes"][0].set_alpha(0.25)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_ylabel(r"$\eta(k,v=5)=5k/(1+5k)$")
    ax.set_title("Cross-system equivalent fear strength (by group, n per point)")
    ax.legend(fontsize=8)

    ax2 = axes[1]
    for g in groups:
        xs = [r["k"] for r in cross if r["group"] == g]
        ys = [r["rmse_normalized_total"] for r in cross if r["group"] == g]
        ax2.scatter(xs, ys, label=f"{g} (n={len(xs)})", c=palette[g], s=60, alpha=0.85)
    ax2.set_xscale("log")
    ax2.set_xlabel("fitted k")
    ax2.set_ylabel("RMSE total (B-D+fear)")
    ax2.set_title("k vs fit quality (identifiability)")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_killifish_sites(cross: list[dict], path: Path) -> None:
    fish = [r for r in cross if r["group"] == "fish"]
    if not fish:
        return
    fish.sort(key=lambda d: d["series"])
    labels = [r.get("group_key") or r["series"].split("_")[-1] for r in fish]
    ks = [r["k"] for r in fish]
    etas = [r["eta_v5"] for r in fish]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    ax1.bar(x - 0.2, ks, width=0.4, label="k", color="#4C72B0")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("fitted k")
    ax1.set_yscale("symlog", linthresh=1e-4)
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, etas, width=0.4, label=r"$\eta(v=5)$", color="#C44E52", alpha=0.7)
    ax2.set_ylabel(r"$\eta(v=5)$")
    ax1.set_title("Killifish 3 sites: k and eta comparison")
    for i, (kv, ev) in enumerate(zip(ks, etas)):
        ax1.text(i - 0.2, kv, f"{kv:.2e}", ha="center", va="bottom", fontsize=7, rotation=90)
        ax2.text(i + 0.2, ev, f"{ev:.3f}", ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_andren_regions(cross: list[dict], path: Path) -> None:
    reg = [r for r in cross if "andren" in r["series"].lower()]
    if not reg:
        return

    def _region(name: str) -> int:
        parts = name.rsplit("_", 1)
        try:
            return int(parts[-1])
        except ValueError:
            return 0

    reg.sort(key=lambda d: _region(d["series"]))
    regions = [_region(r["series"]) for r in reg]
    ks = [r["k"] for r in reg]
    rmses = [r["rmse_normalized_total"] for r in reg]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(regions, ks, "o-", color="#4C72B0", label="k")
    ax1.set_xlabel("Andrén region")
    ax1.set_ylabel("fitted k")
    ax2 = ax1.twinx()
    ax2.plot(regions, rmses, "s--", color="#C44E52", label="RMSE")
    ax2.set_ylabel("RMSE total")
    ax1.set_title("Lynx-roe deer 7 regions: k and RMSE heterogeneity")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _profile_identifiability(profile_rows: list[dict], fitted_k: float) -> dict:
    rmses = [
        r["rmse_normalized_total"]
        for r in profile_rows
        if np.isfinite(r["rmse_normalized_total"])
    ]
    if not rmses:
        return {"k_min_rmse": float("nan"), "rmse_min": float("nan"), "width_10pct": float("nan")}
    rmin = min(rmses)
    tol = 1.10 * rmin
    ks_ok = [r["k"] for r in profile_rows if r["rmse_normalized_total"] <= tol]
    return {
        "k_min_rmse": min(ks_ok) if ks_ok else float("nan"),
        "k_max_rmse": max(ks_ok) if ks_ok else float("nan"),
        "width_10pct": (max(ks_ok) - min(ks_ok)) if len(ks_ok) >= 2 else 0.0,
        "rmse_min": rmin,
        "fitted_k": fitted_k,
    }


def plot_k_profile(
    series_name: str,
    profile_rows: list[dict],
    fitted_k: float,
    path: Path,
) -> None:
    ks = [r["k"] for r in profile_rows]
    rmses = [r["rmse_normalized_total"] for r in profile_rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, rmses, "b.-", lw=1.2)
    ax.axvline(fitted_k, color="red", ls="--", label=f"fitted k={fitted_k:.4g}")
    rmin = min(r for r in rmses if np.isfinite(r))
    ax.axhline(1.1 * rmin, color="gray", ls=":", label="110% min RMSE")
    ax.set_xscale("log")
    ax.set_xlabel("k (fixed other B-D params)")
    ax.set_ylabel("RMSE total")
    ax.set_title(f"k profile: {series_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_k_profile_grid(all_profiles: dict[str, list[dict]], fitted_ks: dict[str, float], path: Path) -> None:
    names = sorted(all_profiles.keys())
    n = len(names)
    ncols = 4
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows))
    axes = np.atleast_2d(axes)
    for idx, name in enumerate(names):
        ax = axes[idx // ncols, idx % ncols]
        rows = all_profiles[name]
        ks = [r["k"] for r in rows]
        rmses = [r["rmse_normalized_total"] for r in rows]
        ax.plot(ks, rmses, "b.-", ms=3)
        fk = fitted_ks.get(name, float("nan"))
        if np.isfinite(fk):
            ax.axvline(fk, color="red", ls="--", lw=0.8)
        ax.set_xscale("log")
        ax.set_title(name[:18], fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")
    fig.suptitle("k profile RMSE curves (all series)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_k_profiles(
    bda_rows: list[dict],
    series_by_name: dict,
    out_dir: Path,
    representatives_only: bool = False,
) -> tuple[dict[str, list[dict]], list[dict]]:
    all_profiles: dict[str, list[dict]] = {}
    ident_rows: list[dict] = []

    for row in bda_rows:
        name = row["series"]
        if representatives_only and name not in PROFILE_REPRESENTATIVES:
            continue
        if name not in series_by_name:
            print(f"  [skip profile] series not loaded: {name}")
            continue
        series = series_by_name[name]
        profile = profile_bda_k(series, row)
        all_profiles[name] = profile
        fk = _f(row, "k")
        ident = _profile_identifiability(profile, fk)
        ident["series"] = name
        ident["group"] = _classify_group(name)
        ident_rows.append(ident)

        if name in PROFILE_REPRESENTATIVES:
            plot_k_profile(name, profile, fk, out_dir / f"k_profile_{name[:40]}.png")

    if not representatives_only and all_profiles:
        fitted_ks = {row["series"]: _f(row, "k") for row in bda_rows if row["series"] in all_profiles}
        plot_k_profile_grid(all_profiles, fitted_ks, out_dir / "k_profile_all_grid.png")

    long_rows = []
    for sname, rows in all_profiles.items():
        for r in rows:
            long_rows.append({"series": sname, **r})
    _write_csv(
        long_rows,
        out_dir / "k_profile_long.csv",
        [
            "series", "k",
            "rmse_normalized_prey", "rmse_normalized_predator", "rmse_normalized_total",
        ],
    )
    _write_csv(
        ident_rows,
        out_dir / "k_identifiability.csv",
        ["series", "group", "fitted_k", "k_min_rmse", "k_max_rmse", "width_10pct", "rmse_min"],
    )
    return all_profiles, ident_rows


def analyze_peacor(out_dir: Path) -> dict | None:
    try:
        df = load_peacor_plp()
    except FileNotFoundError as exc:
        print(f"  [skip Peacor] {exc}")
        return None

    effect_col = None
    for cand in df.columns:
        cl = str(cand).lower()
        if "year" in cl and "publish" in cl:
            continue
        if any(x in cl for x in ("hedges", "cohen", "effect size", "effect_size", "lnrr", "log response")):
            effect_col = cand
            break
    if effect_col is None:
        numeric_cols = [
            c for c in df.columns
            if df[c].dtype in ("float64", "int64")
            and "year" not in str(c).lower()
            and "id" not in str(c).lower()
        ]
        effect_col = numeric_cols[0] if numeric_cols else None

    summary: dict = {"n_studies": int(len(df))}
    fig, ax = plt.subplots(figsize=(7, 4))
    if effect_col and df[effect_col].dtype in ("float64", "int64"):
        summary["effect_col"] = effect_col
        summary["effect_type"] = "numeric"
        vals = df[effect_col].dropna().astype(float)
        vals = vals[np.isfinite(vals)]
        if len(vals):
            ax.hist(vals, bins=30, color="#4C72B0", alpha=0.85, edgecolor="white")
            ax.axvline(0, color="black", lw=0.8)
            summary["effect_median"] = float(median(vals))
            summary["effect_mean"] = float(mean(vals))
            summary["effect_n"] = int(len(vals))
        ax.set_xlabel(effect_col)
        ax.set_ylabel("count")
        ax.set_title("Peacor PLP meta-analysis: effect size distribution")
    else:
        # PLP studies 表为研究清单，Predation effect 为 TMIE/NCE 分类
        cat_col = "Predation effect" if "Predation effect" in df.columns else None
        if cat_col is None:
            cat_col = next((c for c in df.columns if df[c].dtype == object), None)
        summary["effect_col"] = cat_col or ""
        summary["effect_type"] = "categorical"
        if cat_col:
            counts = df[cat_col].value_counts()
            labels = [str(k) for k in counts.index]
            ax.bar(labels, counts.values, color=["#4C72B0", "#C44E52"][: len(labels)], alpha=0.85)
            for i, (lab, cnt) in enumerate(zip(labels, counts.values)):
                ax.text(i, cnt, str(cnt), ha="center", va="bottom", fontsize=9)
            tmie = int(counts.get("TMIE", 0))
            nce = int(counts.get("NCE", 0))
            summary["TMIE_count"] = tmie
            summary["NCE_count"] = nce
            summary["tmie_fraction"] = tmie / len(df) if len(df) else 0.0
        ax.set_xlabel(summary["effect_col"] or "category")
        ax.set_ylabel("study count")
        ax.set_title("Peacor PLP studies: TMIE vs NCE (study inventory)")
    fig.tight_layout()
    fig.savefig(out_dir / "peacor_effect_distribution.png", dpi=150)
    plt.close(fig)

    df.head(200).to_csv(out_dir / "peacor_plp_sample.csv", index=False, encoding="utf-8-sig")
    (out_dir / "peacor_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    taxon_summary = analyze_peacor_taxon(df, out_dir)
    if taxon_summary:
        summary["by_taxon"] = taxon_summary
    return summary


def analyze_peacor_taxon(df, out_dir: Path) -> dict | None:
    """Peacor PLP 按脊椎/无脊椎类群拆分 TMIE/NCE（B 轨补充）。"""
    tax_col = "Invertebrate or Vertebrate"
    eff_col = "Predation effect"
    if tax_col not in df.columns or eff_col not in df.columns:
        return None
    groups = sorted(df[tax_col].dropna().unique())
    tmie_counts, nce_counts = [], []
    for g in groups:
        sub = df[df[tax_col] == g]
        tmie_counts.append(int((sub[eff_col] == "TMIE").sum()))
        nce_counts.append(int((sub[eff_col] == "NCE").sum()))
    x = np.arange(len(groups))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, tmie_counts, w, label="TMIE", color="#4C72B0")
    ax.bar(x + w / 2, nce_counts, w, label="NCE", color="#C44E52")
    for i, (t, n) in enumerate(zip(tmie_counts, nce_counts)):
        ax.text(i - w / 2, t, str(t), ha="center", va="bottom", fontsize=9)
        ax.text(i + w / 2, n, str(n), ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel("study count")
    ax.set_title("Peacor PLP: TMIE vs NCE by prey taxon group")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "peacor_by_taxon.png", dpi=150)
    plt.close(fig)
    rows = [
        {"taxon_group": g, "TMIE": t, "NCE": n, "n_total": t + n}
        for g, t, n in zip(groups, tmie_counts, nce_counts)
    ]
    _write_csv(rows, out_dir / "peacor_by_taxon.csv", ["taxon_group", "TMIE", "NCE", "n_total"])
    return {r["taxon_group"]: {"TMIE": r["TMIE"], "NCE": r["NCE"]} for r in rows}


def build_dual_track_summary(
    cross: list[dict],
    peacor: dict | None,
    coral: dict | None,
    damsel: dict | None,
    out_path: Path,
) -> dict:
    """汇总 A 轨 η 与 B 轨实验/元分析先验，供论文双轨对照表。"""
    from statistics import median

    by_group: dict[str, list[float]] = {}
    for r in cross:
        by_group.setdefault(r["group"], []).append(_eta(r["k"], 5.0))
    track_a = {
        g: {
            "n": len(v),
            "eta_v5_median": float(median(v)) if v else float("nan"),
            "eta_v5_min": float(min(v)) if v else float("nan"),
            "eta_v5_max": float(max(v)) if v else float("nan"),
        }
        for g, v in by_group.items()
    }
    track_b = {
        "peacor_tmie_fraction": peacor.get("tmie_fraction") if peacor else None,
        "peacor_by_taxon": peacor.get("by_taxon") if peacor else None,
        "coral_herbivory_suppression": (
            coral.get("max_herbivory_suppression") if coral else None
        ),
        "damselfly_activity_suppression": (
            abs(damsel["suppression_vs_free_fish"]["cage"])
            if damsel and damsel.get("suppression_vs_free_fish", {}).get("cage") is not None
            else None
        ),
    }
    summary = {"track_A_eta_by_group": track_a, "track_B_priors": track_b}
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def analyze_coral_reef(out_dir: Path) -> dict | None:
    path = RAW / "08_coral_reef_fear" / "ratesperhour.csv"
    if not path.is_file():
        print("  [skip coral] ratesperhour.csv missing")
        return None
    rows = read_csv_dicts(path)
    by_pos: dict[int, list[float]] = {}
    for r in rows:
        try:
            pos = int(float(r["posi"]))
            rate = float(r["allbrate"])
        except (KeyError, ValueError):
            continue
        if rate >= 0:
            by_pos.setdefault(pos, []).append(rate)

    if not by_pos:
        return None

    pos_sorted = sorted(by_pos)
    means = {p: mean(by_pos[p]) for p in pos_sorted}
    ref = means[pos_sorted[0]]
    suppress = {p: 1.0 - means[p] / ref if ref > 0 else 0.0 for p in pos_sorted}

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([str(p) for p in pos_sorted], [means[p] for p in pos_sorted], color="#55A868")
    ax.set_xlabel("Fear position (posi)")
    ax.set_ylabel("Mean bite rate (allbrate)")
    ax.set_title("Coral reef: foraging rate vs fear treatment position")
    fig.tight_layout()
    fig.savefig(out_dir / "coral_foraging_by_position.png", dpi=150)
    plt.close(fig)

    # propremov: algae removal assay — pro.diff ≈ % biomass loss under fear position
    max_herbivory_suppression: float | None = None
    prop_path = RAW / "08_coral_reef_fear" / "propremov.csv"
    prop_rows_out: list[dict] = []
    if prop_path.is_file():
        for r in read_csv_dicts(prop_path):
            pos_raw = str(r.get("position", "")).strip().lower()
            if pos_raw in ("c", "p", ""):
                continue
            try:
                pct = float(r["pro.diff"])
            except (KeyError, ValueError):
                continue
            if np.isfinite(pct):
                prop_rows_out.append({"position": pos_raw, "pro_diff_pct": pct})
        if prop_rows_out:
            max_herbivory_suppression = max(r["pro_diff_pct"] for r in prop_rows_out) / 100.0
            _write_csv(
                prop_rows_out,
                out_dir / "coral_propremov_summary.csv",
                ["position", "pro_diff_pct"],
            )

    pos_suppress = [v for v in suppress.values() if v > 0]
    summary = {
        "mean_rate_by_posi": {str(k): v for k, v in means.items()},
        "suppression_vs_pos1": {str(k): v for k, v in suppress.items()},
        "max_suppression": max(pos_suppress) if pos_suppress else 0.0,
        "max_herbivory_suppression": max_herbivory_suppression,
    }
    _write_csv(
        [{"posi": p, "mean_rate": means[p], "suppression_vs_pos1": suppress[p]} for p in pos_sorted],
        out_dir / "coral_foraging_summary.csv",
        ["posi", "mean_rate", "suppression_vs_pos1"],
    )
    return summary


def analyze_damselfly(out_dir: Path) -> dict | None:
    path = RAW / "10_damselfly_predator_cues" / "actcage.csv"
    if not path.is_file():
        print("  [skip damselfly] actcage.csv missing")
        return None
    rows = read_csv_dicts(path)
    by_treat: dict[str, list[float]] = {}
    for r in rows:
        try:
            act = float(r["act.mm"])
        except (KeyError, ValueError):
            continue
        tr = r.get("treatment", "unknown")
        by_treat.setdefault(tr, []).append(act)

    means = {t: mean(v) for t, v in by_treat.items() if v}
    ref = means.get("free.fish")
    suppress = {}
    if ref and ref > 0:
        for t, m in means.items():
            suppress[t] = 1.0 - m / ref

    fig, ax = plt.subplots(figsize=(7, 4))
    labels = sorted(means.keys())
    ax.bar(labels, [means[t] for t in labels], color="#C44E52", alpha=0.85)
    ax.set_ylabel("Mean activity (act.mm)")
    ax.set_title("Damselfly: activity under predator cue treatments")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "damselfly_activity_by_treatment.png", dpi=150)
    plt.close(fig)

    summary = {
        "mean_activity": means,
        "suppression_vs_free_fish": suppress,
    }
    _write_csv(
        [{"treatment": t, "mean_act_mm": means[t],
          "suppression_vs_free_fish": suppress.get(t, float("nan"))} for t in labels],
        out_dir / "damselfly_activity_summary.csv",
        ["treatment", "mean_act_mm", "suppression_vs_free_fish"],
    )
    return summary


def plot_mechanism_prior(
    cross: list[dict],
    coral: dict | None,
    damsel: dict | None,
    peacor: dict | None,
    path: Path,
) -> None:
    """将拟合 η 与实验/元分析抑制量级放在同一图。"""
    etas = [_eta(r["k"], 5.0) for r in cross]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([etas], tick_labels=["fitted eta(v=5)\n12 ODE series"])
    refs: list[tuple[str, float]] = []
    if coral:
        coral_val = coral.get("max_herbivory_suppression")
        if coral_val is None or not np.isfinite(coral_val) or coral_val <= 0:
            coral_val = coral.get("max_suppression")
        if coral_val is not None and np.isfinite(coral_val) and coral_val > 0:
            refs.append(("coral herbivory↓ (propremov)", float(coral_val)))
    if damsel and damsel.get("suppression_vs_free_fish"):
        cage = damsel["suppression_vs_free_fish"].get("cage")
        if cage is not None and np.isfinite(cage):
            refs.append(("damselfly |activity↓| (cage)", abs(float(cage))))
    for _i, (label, val) in enumerate(refs):
        ax.axhline(val, ls="--", lw=1.2, label=f"{label}={val:.2f}")
    if peacor and peacor.get("effect_median") is not None:
        ax.axhline(abs(peacor["effect_median"]), ls=":", color="gray",
                   label=f"|Peacor median|={abs(peacor['effect_median']):.2f}")
    elif peacor and peacor.get("tmie_fraction") is not None:
        frac = peacor["tmie_fraction"]
        ax.axhline(frac, ls=":", color="gray",
                   label=f"Peacor TMIE share={frac:.2f}")
    ax.set_ylabel("relative suppression (0–1 scale)")
    ax.set_title("Fitted η vs experimental/meta-analysis priors")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_lter_extra(out_dir: Path, lake_id: int = LTER_DEFAULT_LAKE) -> list[dict]:
    out_rows: list[dict] = []
    for prey, pred in LTER_EXTRA_PAIRS:
        try:
            series = load_lter_fish_pair(
                lake_id=lake_id,
                prey_species=prey,
                predator_species=pred,
            )
        except ValueError as exc:
            print(f"  [skip LTER] lake={lake_id} {prey}/{pred}: {exc}")
            continue
        print(f"  [LTER fit] {series.name} ({series.n_points} pts)")
        res = fit_bda_fear_to_series(series)
        out_rows.append({
            "series": series.name,
            "prey": prey,
            "predator": pred,
            "lake_id": lake_id,
            "n_points": series.n_points,
            "rmse_normalized_total": res.rmse_normalized_total,
            "k": res.params.get("k", float("nan")),
            "p": res.params.get("p", float("nan")),
            "q": res.params.get("q", float("nan")),
        })
        from src.visualize import plot_fit_result
        plot_fit_result(res, out_dir / f"lter_{series.name}_bda_fear.png")
    _write_csv(out_rows, out_dir / "lter_extra_fits.csv",
               ["series", "prey", "predator", "lake_id", "n_points",
                "rmse_normalized_total", "k", "p", "q"])
    return out_rows


def _write_report(
    path: Path,
    cross: list[dict],
    improve: list[dict],
    ident: list[dict],
    peacor: dict | None,
    coral: dict | None,
    damsel: dict | None,
    lter: list[dict],
) -> None:
    lines = [
        "# Deep data analysis report",
        f"generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Tier 1",
        f"- cross_system rows: {len(cross)}",
        f"- RMSE improvement entries: {len(improve)}",
        f"- k identifiability entries: {len(ident)}",
        "",
        "### Top RMSE improvements",
    ]
    for r in sorted(improve, key=lambda x: x.get("improvement_ratio", 0), reverse=True)[:5]:
        lines.append(
            f"- {r['series']}: baseline/bda = {r.get('improvement_ratio', float('nan')):.2g}"
        )
    lines.extend(["", "## Tier 2"])
    lines.append(f"- Peacor: {peacor or 'skipped'}")
    lines.append(f"- Coral reef max foraging suppression: "
                 f"{coral.get('max_suppression') if coral else 'skipped'}")
    if damsel and damsel.get("suppression_vs_free_fish"):
        lines.append(f"- Damselfly cage suppression: "
                     f"{damsel['suppression_vs_free_fish'].get('cage', 'n/a')}")
    lines.append(f"- LTER extra fits: {len(lter)}")
    lines.extend([
        "",
        "## Figures",
        "- tier1/rmse_improvement.png",
        "- tier1/eta_by_group.png",
        "- tier1/killifish_sites.png",
        "- tier1/andren_regions.png",
        "- tier1/k_profile_*.png",
        "- tier2/peacor_effect_distribution.png",
        "- tier2/peacor_by_taxon.png",
        "- tier2/coral_foraging_by_position.png",
        "- tier2/damselfly_activity_by_treatment.png",
        "- tier2/mechanism_prior_comparison.png",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep analysis on existing datasets (tier 1+2)")
    parser.add_argument("--fit-summary", type=Path, default=None)
    parser.add_argument("--skip-lter", action="store_true", help="跳过 LTER 额外拟合（省时）")
    parser.add_argument("--skip-profile-all", action="store_true",
                        help="仅 profile 代表序列，不跑全 12 条")
    args = parser.parse_args()

    tier1 = OUT / "tier1"
    tier2 = OUT / "tier2"
    lter_dir = OUT / "lter_extra"
    for d in (tier1, tier2, lter_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Deep data analysis (tier 1 + tier 2)")
    print("=" * 60)

    fit_path = _find_fit_summary(args.fit_summary)
    print(f"\n[load] {fit_path}")
    fit_rows = _load_fit_table(fit_path)

    # --- Tier 1: tables & plots from fit_summary ---
    cross = build_cross_system_table(fit_rows)
    _write_csv(cross, tier1 / "cross_system_k_eta.csv",
               ["series", "group", "group_key", "k", "v0", "eta_v0", "eta_v5",
                "rmse_normalized_total", "rmse_normalized_prey",
                "rmse_normalized_predator", "p", "q", "r_param"])

    improve = build_rmse_improvement(fit_rows)
    _write_csv(improve, tier1 / "rmse_improvement.csv",
               ["series", "group", "rmse_normalized_baseline",
                "rmse_normalized_fear_memory", "rmse_normalized_bda_fear",
                "improvement_ratio"])

    plot_rmse_improvement(improve, tier1 / "rmse_improvement.png")
    plot_eta_by_group(cross, tier1 / "eta_by_group.png")
    plot_killifish_sites(cross, tier1 / "killifish_sites.png")
    plot_andren_regions(cross, tier1 / "andren_regions.png")
    print("[tier1] tables + RMSE/eta/group plots OK")

    # k profiles need loaded series
    print("\n[tier1] loading series for k profiles ...")
    series_list = discover_and_load(min_confidence=0.5)
    series_by_name = _series_index(series_list)

    bda_rows = [
        r for r in fit_rows
        if r.get("model") == "bda_fear" and _is_usable_fit(r)
    ]

    if args.skip_profile_all:
        _, ident = run_k_profiles(bda_rows, series_by_name, tier1, representatives_only=True)
    else:
        _, ident = run_k_profiles(bda_rows, series_by_name, tier1, representatives_only=False)
    print(f"[tier1] k profiles + identifiability ({len(ident)} series) OK")

    # --- Tier 2: mechanism priors ---
    print("\n[tier2] Peacor / coral / damselfly ...")
    peacor = analyze_peacor(tier2)
    coral = analyze_coral_reef(tier2)
    damsel = analyze_damselfly(tier2)
    plot_mechanism_prior(cross, coral, damsel, peacor, tier2 / "mechanism_prior_comparison.png")

    dual = build_dual_track_summary(cross, peacor, coral, damsel, tier2 / "dual_track_summary.json")
    print(f"[tier2] dual-track summary: {dual['track_A_eta_by_group']}")

    prior = {
        "peacor": peacor,
        "coral_reef": coral,
        "damselfly": damsel,
    }
    (tier2 / "mechanism_prior.json").write_text(
        json.dumps(prior, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    print("[tier2] mechanism prior analysis OK")

    # LTER extra
    lter_rows: list[dict] = []
    if not args.skip_lter:
        print("\n[tier2] LTER extra fish pairs ...")
        lter_rows = run_lter_extra(lter_dir)
    else:
        print("\n[tier2] LTER skipped (--skip-lter)")

    _write_report(OUT / "report.md", cross, improve, ident, peacor, coral, damsel, lter_rows)

    print("\n" + "=" * 60)
    print(f"DONE -> {OUT}")
    print("  tier1/: cross_system_k_eta.csv, rmse_improvement.png, k_profile_*.png, ...")
    print("  tier2/: peacor, coral, damselfly, mechanism_prior_comparison.png")
    if lter_rows:
        print(f"  lter_extra/: {len(lter_rows)} extra fits")
    print("=" * 60)


if __name__ == "__main__":
    main()
