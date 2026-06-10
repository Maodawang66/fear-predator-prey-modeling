"""
Phase 2 补充实验：退化检验(30)、ΔAICc 实质阈值(31)、研究级聚合(33)。

退化检验分三部分：
  (a) 数值验证 φ=0 时 fear-memory 轨迹与 baseline 完全重合
  (b) 对 AICc 支持 baseline 的序列，检查 fitted φ 是否集中在接近 0 的区域
  (c) 绘制 ΔAICc vs fitted φ 散点图

ΔAICc 阈值：
  计算 baseline 与最佳恐惧模型的 ΔAICc，报告 >2, >4, >7 的序列数

研究级聚合：
  研究内序列取验证 RMSE 中位数（Andrén 七区、Killifish 三站、Windermere 两湖盆）
  其余独立研究各算 1 单位，重新计算研究级胜负表
  同时报告中位数、最佳、最差三种聚合方式
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import json
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIT_CSV = os.path.join(ROOT, "results", "calibration_bda", "fit_summary.csv")
SIX_MODEL_CSV = os.path.join(
    ROOT, "results", "seven_model_real_fits",
    "report_protocol_seven_model_metrics.csv",
)
OUT_DIR = os.path.join(ROOT, "results", "phase2_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# study grouping
# ---------------------------------------------------------------------------
STUDY_GROUPS = {
    "Andrén": [
        "andren_lynx_roedeer_data_1",
        "andren_lynx_roedeer_data_2",
        "andren_lynx_roedeer_data_3",
        "andren_lynx_roedeer_data_4",
        "andren_lynx_roedeer_data_5",
        "andren_lynx_roedeer_data_6",
        "andren_lynx_roedeer_data_7",
    ],
    "Killifish": [
        "timeserieslogmeans_WRHW",
        "timeserieslogmeans_TP",
        "timeserieslogmeans_WRGP",
    ],
    "Windermere": [
        "windermere_north_pike_perch",
        "windermere_south_pike_perch",
    ],
}
# Identify independent studies (not in any group)
ALL_SERIES_15 = []
GROUP_MAP = {}  # series -> study name
for name, members in STUDY_GROUPS.items():
    for m in members:
        GROUP_MAP[m] = name

INDEPENDENT = {}  # will be filled after we see the data

FRIENDLY_NAMES = {
    "glerl_m110_zoop_1994-201": "GLERL",
    "isle_royale_wolf_moose_pre_2018": "Isle Royale",
    "komi_lynx_hare": "Komi lynx-hare",
}


def friendly(s):
    return FRIENDLY_NAMES.get(s, s)


def study_name(s):
    return GROUP_MAP.get(s, friendly(s))


# ===================================================================
# Load data
# ===================================================================

def load_fit_summary():
    """Return list of dicts from fit_summary.csv (baseline + fear_memory only)."""
    rows = []
    with open(FIT_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] in ("baseline", "fear_memory"):
                rows.append(row)
    return rows


def load_six_model_metrics():
    """Return list of dicts, excluding bda_fear."""
    rows = []
    with open(SIX_MODEL_CSV, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["model"] != "bda_fear":
                rows.append(row)
    return rows


# ===================================================================
# Item 30a — numerical verification: φ=0 gives same trajectory as baseline
# ===================================================================

def verify_phi_zero_degeneration():
    """Simulate baseline and fear_memory(phi=0) from matching initial conditions
    and check that trajectories are identical within numerical tolerance."""
    print("=" * 70)
    print("Item 30a: φ=0 degeneration check")
    print("=" * 70)

    import sys
    # Import as package (src is the package root)
    sys.path.insert(0, ROOT)
    from src.parameters import BaselineParams, FearMemoryParams
    from src.simulate import integrate_baseline, integrate_fear_memory

    # Use default parameters (same as in parameters.py defaults)
    bp = BaselineParams()
    fp = FearMemoryParams(phi=0.0, delta=1.0)

    t_span = (0, 100)
    t_eval = np.linspace(0, 100, 500)

    sol_base = integrate_baseline(bp, t_span, n_points=len(t_eval), rtol=1e-12, atol=1e-14)
    sol_fear = integrate_fear_memory(fp, t_span, n_points=len(t_eval), rtol=1e-12, atol=1e-14)

    dx = sol_base.y[0] - sol_fear.y[0]
    dy = sol_base.y[1] - sol_fear.y[1]
    max_err = max(np.max(np.abs(dx)), np.max(np.abs(dy)))

    print(f"  Max |x_base - x_fear| = {max_err:.2e}")
    print(f"  Max |y_base - y_fear| = {max_err:.2e}")
    print(f"  Coincident: {max_err < 1e-10}")

    # Plot overlay
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t_eval, sol_base.y[0], "b-", label="baseline", linewidth=2)
    ax1.plot(t_eval, sol_fear.y[0], "r--", label="fear-memory φ=0", linewidth=1.5)
    ax1.set_xlabel("t")
    ax1.set_ylabel("x (prey)")
    ax1.set_title("Prey trajectory")
    ax1.legend()

    ax2.plot(t_eval, sol_base.y[1], "b-", label="baseline", linewidth=2)
    ax2.plot(t_eval, sol_fear.y[1], "r--", label="fear-memory φ=0", linewidth=1.5)
    ax2.set_xlabel("t")
    ax2.set_ylabel("y (predator)")
    ax2.set_title("Predator trajectory")
    ax2.legend()

    fig.suptitle("Degeneration check: fear-memory(φ=0) vs baseline")
    fig.tight_layout()
    out = os.path.join(OUT_DIR, "degeneration_phi_zero.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")
    return max_err < 1e-10


# ===================================================================
# Item 30b/c + 31 — φ distribution and ΔAICc analysis
# ===================================================================

def analyze_phi_and_aicc():
    """For each series, compute baseline vs best-fear ΔAICc and plot vs fitted φ."""
    print("\n" + "=" * 70)
    print("Item 30b/c & 31: ΔAICc vs fitted φ, threshold analysis")
    print("=" * 70)

    # Load fitted φ from fit_summary (fear_memory rows)
    fit_rows = load_fit_summary()
    phi_fitted = {}
    aicc_base = {}
    aicc_fear = {}
    for row in fit_rows:
        s = row["series"]
        if row["model"] == "fear_memory":
            phi_fitted[s] = float(row["phi"])
            aicc_fear[s] = float(row["aicc"])
        elif row["model"] == "baseline":
            aicc_base[s] = float(row["aicc"])

    # Load six-model metrics to get best fear model AICc per series
    six = load_six_model_metrics()
    best_fear_aicc = {}  # series -> best AICc among fear models
    best_fear_model = {}
    for row in six:
        s = row["series"]
        aicc = float(row["aicc"])
        m = row["model"]
        if m != "baseline":
            if s not in best_fear_aicc or aicc < best_fear_aicc[s]:
                best_fear_aicc[s] = aicc
                best_fear_model[s] = m

    # Build table
    series_list = sorted(phi_fitted.keys())
    rows_data = []
    for s in series_list:
        base = aicc_base.get(s, np.nan)
        fm = aicc_fear.get(s, np.nan)
        best_f = best_fear_aicc.get(s, fm)
        best_fm = best_fear_model.get(s, "fear_memory")
        phi = phi_fitted[s]
        d_fm = base - fm  # ΔAICc: positive = fear-memory better
        d_best = base - best_f  # ΔAICc vs best fear model

        rows_data.append({
            "series": s,
            "friendly": friendly(s),
            "study": study_name(s),
            "aicc_base": base,
            "aicc_fear_memory": fm,
            "aicc_best_fear": best_f,
            "best_fear_model": best_fm,
            "phi_fitted": phi,
            "delta_aicc_fm": d_fm,
            "delta_aicc_best": d_best,
        })

    # --- Item 30b: φ concentration for AICc-baseline-win sequences ---
    baseline_win_series = [r for r in rows_data if r["delta_aicc_best"] > 0]
    baseline_lose_series = [r for r in rows_data if r["delta_aicc_best"] <= 0]
    phi_baseline_win = [r["phi_fitted"] for r in baseline_win_series]
    phi_baseline_lose = [r["phi_fitted"] for r in baseline_lose_series]

    print(f"\n  Sequences where baseline AICc > best fear AICc: {len(baseline_win_series)}")
    print(f"  Sequences where best fear AICc >= baseline: {len(baseline_lose_series)}")
    if phi_baseline_win:
        print(f"  Fitted φ in baseline-win sequences: "
              f"median={np.median(phi_baseline_win):.6f}, "
              f"mean={np.mean(phi_baseline_win):.6f}")
    if phi_baseline_lose:
        print(f"  Fitted φ in baseline-lose sequences: "
              f"median={np.median(phi_baseline_lose):.6f}, "
              f"mean={np.mean(phi_baseline_lose):.6f}")

    # --- Item 31: ΔAICc threshold counts ---
    print("\n  --- ΔAICc (baseline vs best fear model) thresholds ---")
    for thresh in [2, 4, 7, 10]:
        n_pos = sum(1 for r in rows_data if r["delta_aicc_best"] > thresh)
        n_neg = sum(1 for r in rows_data if r["delta_aicc_best"] < -thresh)
        print(f"    |ΔAICc| > {thresh:2d}:  baseline better={n_pos}, fear better={n_neg}")
    n_near_zero = sum(1 for r in rows_data if abs(r["delta_aicc_best"]) <= 2)
    print(f"    |ΔAICc| <= 2 (negligible): {n_near_zero}")

    # --- Print detailed table ---
    print(f"\n  {'Series':<35s} {'Study':<14s} {'Base AICc':>10s} {'Fear AICc':>10s} {'BestFear':>10s} {'Δbest':>8s} {'φ':>10s}")
    print("  " + "-" * 105)
    for r in rows_data:
        print(f"  {r['friendly']:<35s} {r['study']:<14s} {r['aicc_base']:10.2f} {r['aicc_fear_memory']:10.2f} {r['aicc_best_fear']:10.2f} {r['delta_aicc_best']:8.2f} {r['phi_fitted']:10.6f}")

    # --- Item 30c: ΔAICc vs fitted φ scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: ΔAICc(fear_memory) vs φ
    ax = axes[0]
    x = [r["phi_fitted"] for r in rows_data]
    y = [r["delta_aicc_fm"] for r in rows_data]
    labels = [r["friendly"] for r in rows_data]
    colors = ["#d62728" if r["study"] == "Andrén" else
              "#2ca02c" if r["study"] == "Killifish" else
              "#ff7f0e" if r["study"] == "Windermere" else
              "#1f77b4" for r in rows_data]
    ax.scatter(x, y, c=colors, s=60, edgecolors="k", linewidth=0.5, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x[i], y[i]), fontsize=6, alpha=0.8,
                     textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Fitted φ (fear-memory)")
    ax.set_ylabel("ΔAICc = AICc(baseline) − AICc(fear-memory)")
    ax.set_title("ΔAICc vs fitted φ (baseline vs fear-memory)")

    # Right: ΔAICc(best fear) vs φ
    ax = axes[1]
    y2 = [r["delta_aicc_best"] for r in rows_data]
    ax.scatter(x, y2, c=colors, s=60, edgecolors="k", linewidth=0.5, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    for i, lbl in enumerate(labels):
        ax.annotate(lbl, (x[i], y2[i]), fontsize=6, alpha=0.8,
                     textcoords="offset points", xytext=(4, 4))
    ax.set_xlabel("Fitted φ (fear-memory)")
    ax.set_ylabel("ΔAICc = AICc(baseline) − AICc(best fear)")
    ax.set_title("ΔAICc vs fitted φ (baseline vs best fear model)")

    import matplotlib.patches as mpatches
    legend_handles = [
        mpatches.Patch(color="#d62728", label="Andrén (7 regions)"),
        mpatches.Patch(color="#2ca02c", label="Killifish (3 sites)"),
        mpatches.Patch(color="#ff7f0e", label="Windermere (2 basins)"),
        mpatches.Patch(color="#1f77b4", label="Independent"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4, fontsize=8)

    fig.suptitle("Degeneration diagnostics: ΔAICc vs fitted φ", fontsize=13)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    out = os.path.join(OUT_DIR, "delta_aicc_vs_phi.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Saved: {out}")

    return rows_data


# ===================================================================
# Item 33 — study-level aggregation
# ===================================================================

def study_level_aggregation(six_model_rows, aicc_rows=None):
    """Aggregate six-model metrics to study level using median, best, worst.

    For grouped studies (Andrén, Killifish, Windermere):
      - Take median validation RMSE within group as the study-level metric
      - Also report best (min) and worst (max) for sensitivity
    For independent series: each is its own study unit.
    """
    print("\n" + "=" * 70)
    print("Item 33: Study-level aggregation")
    print("=" * 70)

    # Build per-series per-model validation RMSE and AICc
    series_metrics = {}  # series -> {model: {"val_rmse": ..., "aicc": ...}}
    for row in six_model_rows:
        s = row["series"]
        m = row["model"]
        if s not in series_metrics:
            series_metrics[s] = {}
        series_metrics[s][m] = {
            "val_rmse": float(row["validation_rmse"]),
            "aicc": float(row["aicc"]),
        }

    # Identify all series
    all_series = sorted(series_metrics.keys())
    models_list = ["baseline", "fear_memory", "fear_instant",
                   "fear_saturating", "fear_foraging", "fear_handling"]

    # Build study units
    studies = {}  # study_name -> list of series
    for s in all_series:
        sn = study_name(s)
        studies.setdefault(sn, []).append(s)

    print(f"\n  Study units: {len(studies)}")
    for sn, members in studies.items():
        print(f"    {sn}: {len(members)} series — {[friendly(m) for m in members]}")

    # Aggregate: for each study and each aggregation method (median/best/worst)
    # we compute the study-level val_rmse and AICc
    agg_methods = {
        "median": np.median,
        "best": np.min,
        "worst": np.max,
    }

    for agg_name, agg_fn in agg_methods.items():
        print(f"\n  --- Aggregation: {agg_name} ---")
        study_val = {}  # study -> {model: val_rmse}
        study_aicc = {}

        for sn, members in studies.items():
            study_val[sn] = {}
            study_aicc[sn] = {}
            for m in models_list:
                vals = []
                aiccs = []
                for s in members:
                    if s in series_metrics and m in series_metrics[s]:
                        vals.append(series_metrics[s][m]["val_rmse"])
                        aiccs.append(series_metrics[s][m]["aicc"])
                if vals:
                    study_val[sn][m] = agg_fn(vals)
                    study_aicc[sn][m] = agg_fn(aiccs)

        # Win table: for each study, find best model by val_rmse and AICc
        print(f"  {'Study':<18s} {'Val best':<18s} {'AICc best':<18s}")
        print("  " + "-" * 55)
        val_wins = {}
        aicc_wins = {}
        for sn in sorted(studies.keys()):
            if sn not in study_val:
                continue
            # Best val RMSE = min
            best_val_model = min(study_val[sn], key=lambda m: study_val[sn][m])
            best_aicc_model = min(study_aicc[sn], key=lambda m: study_aicc[sn][m])
            val_wins[best_val_model] = val_wins.get(best_val_model, 0) + 1
            aicc_wins[best_aicc_model] = aicc_wins.get(best_aicc_model, 0) + 1
            print(f"  {sn:<18s} {best_val_model:<18s} {best_aicc_model:<18s}")

        # Print tally
        print(f"\n  Validation RMSE wins ({agg_name}):")
        for m in models_list:
            n = val_wins.get(m, 0)
            print(f"    {m:<20s}: {n}")
        print(f"  AICc wins ({agg_name}):")
        for m in models_list:
            n = aicc_wins.get(m, 0)
            print(f"    {m:<20s}: {n}")

    return studies


# ===================================================================
# main
# ===================================================================

if __name__ == "__main__":
    # Item 30a
    ok = verify_phi_zero_degeneration()

    # Items 30b/c + 31
    rows_data = analyze_phi_and_aicc()

    # Item 33
    six = load_six_model_metrics()
    study_level_aggregation(six, rows_data)

    print("\n" + "=" * 70)
    print("Phase 2 analysis complete.")
    print(f"Outputs in: {OUT_DIR}")
    print("=" * 70)
