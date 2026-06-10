"""
Phase 2 Item 32: 参数不确定性量化。

方法（因参数触边和优化面平坦，FD Hessian 不可靠）：
  1. 参数边界触及分析：从 fit_summary.csv 提取触边参数
  2. 参数扰动敏感性：对每个参数 ±10% 扰动后重新积分，
     报告对训练段归一化 RMSE 的影响
  3. e-μ 相关性：从各序列点估计计算相关系数
"""

import os

os.environ["MPLBACKEND"] = "Agg"

import json
import csv
import sys
import warnings

import numpy as np
from scipy.stats import pearsonr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.parameters import BaselineParams
from src.model import baseline_rhs
from src.simulate import _integrate

OUT_DIR = os.path.join(ROOT, "results", "phase2_uncertainty")
os.makedirs(OUT_DIR, exist_ok=True)

PARAM_NAMES = ["r", "K", "a", "theta", "e", "mu"]


def load_fit_summary():
    """Load fit_summary.csv and extract baseline rows."""
    fit_csv = os.path.join(ROOT, "results", "calibration_bda", "fit_summary.csv")
    rows = []
    with open(fit_csv, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def load_baseline_params():
    """Load fitted baseline parameters from param JSON files."""
    params_dir = os.path.join(ROOT, "results", "calibration_bda", "params")
    params = {}
    for fname in os.listdir(params_dir):
        if not fname.endswith("_baseline.json"):
            continue
        if "WBIC" in fname:
            continue
        fpath = os.path.join(params_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        series = data["series"]
        p = data["params"]
        params[series] = {
            "r": p["r"], "K": p["K"], "a": p["a"],
            "theta": p["theta"], "e": p["e"], "mu": p["mu"],
            "x0": p["x0"], "y0": p["y0"],
        }
    return params


def load_observations(fit_row):
    """Load training observations from fit_summary row metadata."""
    src_file = fit_row.get("source_file", "")
    time_col = fit_row.get("time_col", "")
    prey_col = fit_row.get("prey_col", "")
    pred_col = fit_row.get("predator_col", "")
    n_train = int(fit_row.get("n_train_points", 20))

    if not src_file or not os.path.exists(src_file):
        return None, None, None, None, None

    try:
        import pandas as pd

        # GPDD long-format data: extract MainIDs from column labels
        if "gpdd_population_records" in src_file:
            import re
            prey_main_id = re.search(r"MainID=(\d+)", prey_col)
            pred_main_id = re.search(r"MainID=(\d+)", pred_col)
            if not prey_main_id or not pred_main_id:
                return None, None, None, None, None
            prey_id, pred_id = prey_main_id.group(1), pred_main_id.group(1)
            wanted = {prey_id, pred_id}
            df = pd.read_csv(src_file)
            by_year = {}
            for _, row in df.iterrows():
                try:
                    mid = str(int(float(row["MainID"])))
                except (ValueError, KeyError):
                    continue
                if mid not in wanted:
                    continue
                try:
                    year = int(float(row["SampleYear"]))
                    value = float(row["Population"])
                except (ValueError, KeyError):
                    continue
                if mid == prey_id:
                    by_year.setdefault(year, [None, None])[0] = value
                else:
                    by_year.setdefault(year, [None, None])[1] = value
            years = sorted(y for y, vals in by_year.items()
                          if vals[0] is not None and vals[1] is not None)
            t_all = np.array(years, dtype=float)
            x_all = np.array([by_year[y][0] for y in years])
            y_all = np.array([by_year[y][1] for y in years])
        elif src_file.endswith(".csv"):
            df = pd.read_csv(src_file)
            t_raw = df[time_col].values
            try:
                t_all = t_raw.astype(float)
            except (ValueError, TypeError):
                t_all = np.arange(len(t_raw), dtype=float)
            x_all = df[prey_col].values.astype(float)
            y_all = df[pred_col].values.astype(float)
        else:
            df = pd.read_csv(src_file, sep="\t")
            t_raw = df[time_col].values
            try:
                t_all = t_raw.astype(float)
            except (ValueError, TypeError):
                t_all = np.arange(len(t_raw), dtype=float)
            x_all = df[prey_col].values.astype(float)
            y_all = df[pred_col].values.astype(float)

        if len(t_all) < n_train:
            n_train = len(t_all)

        t_train = t_all[:n_train]
        x_obs = x_all[:n_train]
        y_obs = y_all[:n_train]

        x_min, x_max = x_obs.min(), x_obs.max()
        y_min, y_max = y_obs.min(), y_obs.max()
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0

        return t_train, x_obs, y_obs, (x_min, x_range), (y_min, y_range)
    except Exception as e:
        return None, None, None, None, None


def compute_normalized_rmse(params_lin, t_train, x_obs, y_obs, x_scale, y_scale, x0, y0):
    """Simulate baseline and return normalized total RMSE."""
    bp = BaselineParams(**{n: params_lin[i] for i, n in enumerate(PARAM_NAMES)})
    y0_vec = np.array([x0, y0], dtype=float)
    t_span = (float(t_train[0]), float(t_train[-1]))

    sol = _integrate(
        lambda t, s: baseline_rhs(t, s, bp),
        y0_vec, t_span, n_points=500,
    )

    x_sim = np.interp(t_train, sol.t, sol.y[0])
    y_sim = np.interp(t_train, sol.t, sol.y[1])

    x_min, x_range = x_scale
    y_min, y_range = y_scale

    x_norm = (x_sim - x_min) / x_range
    y_norm = (y_sim - y_min) / y_range
    x_obs_norm = (x_obs - x_min) / x_range
    y_obs_norm = (y_obs - y_min) / y_range

    n = len(t_train)
    rmse = np.sqrt((np.sum((x_norm - x_obs_norm)**2) + np.sum((y_norm - y_obs_norm)**2)) / (2 * n))
    return rmse


def perturbation_sensitivity(fit_row, fitted_params):
    """For each parameter, perturb ±10% and measure RMSE change."""
    series = fit_row["series"]
    t_train, x_obs, y_obs, x_scale, y_scale = load_observations(fit_row)
    if t_train is None:
        return None

    fp = fitted_params[series]
    x0, y0 = fp["x0"], fp["y0"]
    base = np.array([fp[n] for n in PARAM_NAMES])

    # Baseline RMSE
    try:
        rmse0 = compute_normalized_rmse(base, t_train, x_obs, y_obs, x_scale, y_scale, x0, y0)
    except Exception:
        return None

    result = {}
    for i, name in enumerate(PARAM_NAMES):
        p_up = base.copy()
        p_down = base.copy()
        p_up[i] *= 1.10
        p_down[i] *= 0.90

        try:
            rmse_up = compute_normalized_rmse(p_up, t_train, x_obs, y_obs, x_scale, y_scale, x0, y0)
        except Exception:
            rmse_up = rmse0
        try:
            rmse_down = compute_normalized_rmse(p_down, t_train, x_obs, y_obs, x_scale, y_scale, x0, y0)
        except Exception:
            rmse_down = rmse0

        pct_up = (rmse_up - rmse0) / rmse0 * 100 if rmse0 > 0 else 0
        pct_down = (rmse_down - rmse0) / rmse0 * 100 if rmse0 > 0 else 0
        max_pct = max(abs(pct_up), abs(pct_down))

        result[name] = {
            "estimate": float(base[i]),
            "rmse0": float(rmse0),
            "rmse_plus10pct": float(rmse_up),
            "rmse_minus10pct": float(rmse_down),
            "delta_pct_plus": float(pct_up),
            "delta_pct_minus": float(pct_down),
            "max_delta_pct": float(max_pct),
            "poorly_constrained": bool(max_pct < 1.0),
        }

    return result


def compute_e_mu_correlation(fitted_params):
    """Compute correlation between e and mu across all 15 series."""
    e_vals = []
    mu_vals = []
    for series, fp in fitted_params.items():
        e_vals.append(fp["e"])
        mu_vals.append(fp["mu"])
    r, p = pearsonr(e_vals, mu_vals)
    return r, p


def main():
    import sys
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 70)
    print("Item 32: Parameter uncertainty quantification")
    print("=" * 70)

    fit_rows_all = load_fit_summary()
    fitted_params = load_baseline_params()

    # Build lookup
    fit_rows = {}
    for row in fit_rows_all:
        if row["model"] == "baseline" and row["series"] in fitted_params:
            fit_rows[row["series"]] = row

    # 1. Boundary hit analysis
    print("\n--- Part A: Parameter boundary hits ---")
    bound_counts = {n: 0 for n in PARAM_NAMES}
    for series, row in fit_rows.items():
        hits_str = row.get("parameter_bound_hits", "")
        if hits_str:
            hits = [h.strip() for h in hits_str.split(",") if h.strip()]
            for h in hits:
                if h in bound_counts:
                    bound_counts[h] += 1

    print(f"  Out of {len(fit_rows)} series:")
    for n in PARAM_NAMES:
        print(f"    {n:6s}: hit bound in {bound_counts[n]} series")

    # 2. Perturbation sensitivity
    print("\n--- Part B: Perturbation sensitivity (±10%) ---")
    all_sensitivity = {}
    poorly_constrained_counts = {n: 0 for n in PARAM_NAMES}

    for i, (series, row) in enumerate(fit_rows.items()):
        print(f"\n  [{i+1}/{len(fit_rows)}] {series}")
        sens = perturbation_sensitivity(row, fitted_params)
        all_sensitivity[series] = sens
        if sens is None:
            print(f"    SKIP: cannot load or simulate")
            continue
        for name in PARAM_NAMES:
            s = sens[name]
            flag = " *** POORLY CONSTRAINED ***" if s["poorly_constrained"] else ""
            print(f"    {name:6s}: {s['estimate']:.4g}, "
                  f"ΔRMSE +10%={s['delta_pct_plus']:+.2f}%, "
                  f"-10%={s['delta_pct_minus']:+.2f}%{flag}")
            if s["poorly_constrained"]:
                poorly_constrained_counts[name] += 1

    print(f"\n  Poorly constrained (<1% RMSE change at ±10%):")
    for n in PARAM_NAMES:
        print(f"    {n:6s}: {poorly_constrained_counts[n]}/{len(fit_rows)}")

    # 3. e-μ correlation
    print("\n--- Part C: e-μ correlation across series ---")
    r, p = compute_e_mu_correlation(fitted_params)
    print(f"  Pearson r = {r:.4f}, p = {p:.4f}")
    e_vals = [fitted_params[s]["e"] for s in fitted_params]
    mu_vals = [fitted_params[s]["mu"] for s in fitted_params]
    print(f"  e range: [{min(e_vals):.4g}, {max(e_vals):.4g}]")
    print(f"  mu range: [{min(mu_vals):.4g}, {max(mu_vals):.4g}]")

    # 4. Parameter CV across series
    print("\n--- Part D: Cross-series parameter variability ---")
    for name in PARAM_NAMES:
        vals = [fitted_params[s][name] for s in fitted_params]
        cv = np.std(vals) / np.mean(vals) if np.mean(vals) > 0 else np.inf
        print(f"  {name:6s}: median={np.median(vals):.4g}, "
              f"CV={cv:.3f}, range=[{min(vals):.4g}, {max(vals):.4g}]")

    # Save results
    out = {
        "boundary_hits": {n: bound_counts[n] for n in PARAM_NAMES},
        "poorly_constrained_counts": {n: poorly_constrained_counts[n] for n in PARAM_NAMES},
        "e_mu_correlation": {"r": float(r), "p": float(p)},
        "series_sensitivity": {},
    }
    for series, sens in all_sensitivity.items():
        if sens is None:
            continue
        ss = {}
        for name in PARAM_NAMES:
            info = sens[name]
            ss[name] = {
                "estimate": float(info["estimate"]),
                "max_delta_pct": float(info["max_delta_pct"]),
                "poorly_constrained": bool(info["poorly_constrained"]),
            }
        out["series_sensitivity"][series] = ss

    out_file = os.path.join(OUT_DIR, "parameter_uncertainty.json")
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved: {out_file}")
    print("\nPhase 2 Item 32 complete.")


if __name__ == "__main__":
    main()
