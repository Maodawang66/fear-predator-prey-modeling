"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from .parameters import MechanismId

plt.rcParams["axes.unicode_minus"] = False


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_timeseries_compare(
    sol_base,
    sol_fear,
    out: Path,
    title: str = "Baseline vs fear + memory: time series",
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(sol_base.t, sol_base.y[0], "b-", label="baseline prey", lw=1.5)
    axes[0].plot(sol_fear.t, sol_fear.y[0], "r--", label="fear prey", lw=1.5)
    axes[0].set_ylabel("prey density x")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(sol_base.t, sol_base.y[1], "b-", label="baseline predator", lw=1.5)
    axes[1].plot(sol_fear.t, sol_fear.y[1], "r--", label="fear predator", lw=1.5)
    leg_lines, leg_labels = axes[1].get_legend_handles_labels()
    if sol_fear.y.shape[0] > 2:
        ax2 = axes[1].twinx()
        ax2.plot(sol_fear.t, sol_fear.y[2], "g:", label="memory M", lw=1.2)
        ax2.set_ylabel("memory M", color="green")
        ax2.tick_params(axis="y", labelcolor="green")
        m_lines, m_labels = ax2.get_legend_handles_labels()
        leg_lines += m_lines
        leg_labels += m_labels
    axes[1].set_xlabel("time t")
    axes[1].set_ylabel("predator density y")
    axes[1].legend(leg_lines, leg_labels, loc="upper right")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_phase_plane(
    sol,
    out: Path,
    title: str = "Phase-plane trajectory",
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(sol.y[0], sol.y[1], "k-", lw=1.2)
    ax.plot(sol.y[0, 0], sol.y[1, 0], "go", label="start")
    ax.plot(sol.y[0, -1], sol.y[1, -1], "rs", label="end")
    ax.set_xlabel("prey x")
    ax.set_ylabel("predator y")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_phi_scan(
    data: dict,
    out: Path,
    *,
    x_key: str = "phi",
    xlabel: str = "fear strength φ",
    ylabel: str = "long-run mean density",
    title: str = "φ parameter scan",
    prey_label: str = "prey time mean",
    pred_label: str = "predator time mean",
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = data[x_key]
    ax.plot(x, data["x_mean"], "b-o", ms=3, label=prey_label)
    ax.plot(x, data["y_mean"], "r-s", ms=3, label=pred_label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_delta_scan(data: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(data["delta"], data["x_mean"], "b-o", ms=4, label="prey")
    ax.plot(data["delta"], data["y_mean"], "r-s", ms=4, label="predator")
    ax.set_xlabel("memory decay rate δ")
    ax.set_ylabel("long-run mean density")
    ax.set_title("δ scan (fixed φ)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_sensitivity(sens: dict[str, float], out: Path) -> None:
    names = list(sens.keys())
    vals = np.array([sens[k] for k in names], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    colors = ["#4C72B0" if v >= 0 else "#C44E52" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_ylabel("∂(mean prey density)/∂param (normalized FD)")
    ax.set_title("Local sensitivity (raw)")
    ax.grid(True, axis="y", alpha=0.3)

    ax2 = axes[1]
    scale = np.max(np.abs(vals)) if np.any(np.abs(vals) > 0) else 1.0
    norm = vals / scale
    colors2 = ["#4C72B0" if v >= 0 else "#C44E52" for v in norm]
    ax2.bar(names, norm, color=colors2)
    ax2.axhline(0, color="gray", lw=0.8)
    ax2.set_ylim(-1.05, 1.05)
    ax2.set_ylabel("normalized |sensitivity| (max=1)")
    ax2.set_title("Local sensitivity (normalized to max |∂|)")
    ax2.grid(True, axis="y", alpha=0.3)
    for i, (n, v) in enumerate(zip(names, vals)):
        ax2.text(i, norm[i] + (0.04 if norm[i] >= 0 else -0.08), f"{v:.2e}", ha="center", fontsize=7)

    fig.suptitle("Local sensitivity analysis (fear+memory model)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mechanism_comparison(
    solutions: dict,
    labels: dict[MechanismId, str],
    out: Path,
) -> None:
    """Absolute time series split by model scale; not a cross-family comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col")
    for mid, sol in solutions.items():
        lab = labels.get(mid, mid.value)
        col = 1 if mid in (MechanismId.BDA_BASELINE, MechanismId.BDA_FEAR) else 0
        axes[0, col].plot(sol.t, sol.y[0], lw=1.2, label=lab)
        axes[1, col].plot(sol.t, sol.y[1], lw=1.2, label=lab)
    for col, family in enumerate(("Holling II (x, y scale)", "B-D (u, v scale)")):
        axes[0, col].set_title(family)
        axes[0, col].set_ylabel("prey density")
        axes[1, col].set_ylabel("predator density")
        axes[1, col].set_xlabel("time t")
        for row in range(2):
            axes[row, col].legend(fontsize=7, loc="best")
            axes[row, col].grid(True, alpha=0.3)
    fig.suptitle("Absolute mechanism trajectories (separate model scales)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_mechanism_bars(compare_rows: dict[str, list], out: Path) -> None:
    """Long-run mean changes relative to each model family's no-fear baseline."""
    labels = compare_rows["label"]
    x = np.arange(len(labels))
    w = 0.35
    x_means = compare_rows["x_mean_change_pct"]
    y_means = compare_rows["y_mean_change_pct"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, x_means, w, label="prey mean change", color="#4C72B0")
    ax.bar(x + w / 2, y_means, w, label="predator mean change", color="#C44E52")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("change from model-specific no-fear baseline (%)")
    ax.set_title("Mechanism comparison: relative long-run population changes")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_amplitude_comparison(compare_rows: dict[str, list], out: Path) -> None:
    """Prey relative-amplitude change from each model family's no-fear baseline."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(compare_rows["label"], compare_rows["relative_amplitude_x_change_pct"], color="#55A868")
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("change in prey relative amplitude A/mean (%)")
    ax.set_title("Mechanism comparison: relative oscillation change")
    plt.xticks(rotation=25, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_three_scenarios(
    scenarios: list[tuple[str, object]],
    out: Path,
    suptitle: str = "Three scenarios: baseline / moderate fear / strong fear",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for label, sol in scenarios:
        axes[0].plot(sol.t, sol.y[0], label=label)
        axes[1].plot(sol.t, sol.y[1], label=label)
    axes[0].set_ylabel("prey x")
    axes[0].set_xlabel("time t")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("predator y")
    axes[1].set_xlabel("time t")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(suptitle)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_turing_snapshot(
    result,
    out: Path,
    field: str = "prey",
    title: str | None = None,
) -> None:
    """Single 2D pattern snapshot (prey u or predator v)."""
    Z = result.u if field == "prey" else result.v
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(
        Z.T,
        origin="lower",
        extent=[0, result.config.lx, 0, result.config.ly],
        aspect="auto",
        cmap="viridis",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    lab = "prey u" if field == "prey" else "predator v"
    ttl = title or (
        f"{lab}  t={result.t_final:g}  d2={result.config.d2:g}  "
        f"a={result.params.a:g}  k={result.params.k:g}"
    )
    ax.set_title(ttl, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_turing_d2_panel(
    results: list,
    out: Path,
    field: str = "prey",
    suptitle: str | None = None,
) -> None:
    """Side-by-side d2 comparison (Myint Fig.3–6 style)."""
    n = len(results)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i, res in enumerate(results):
        Z = res.u if field == "prey" else res.v
        im = axes[i].imshow(
            Z.T,
            origin="lower",
            extent=[0, res.config.lx, 0, res.config.ly],
            aspect="auto",
            cmap="viridis",
        )
        axes[i].set_title(f"d2={res.config.d2:g}", fontsize=10)
        axes[i].set_xlabel("x")
        axes[i].set_ylabel("y")
        fig.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_turing_stability_d2(
    rows: list[dict],
    out: Path,
    d2_scan: np.ndarray | None = None,
    max_re_curve: np.ndarray | None = None,
    u_star: float | None = None,
    v_star: float | None = None,
    paper_ref: tuple[float, float] | None = None,
) -> None:
    """d2 -- max Re λ(k) Turing 线性稳定性诊断图。"""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if d2_scan is not None and max_re_curve is not None:
        ax.plot(d2_scan, max_re_curve, color="#4C72B0", lw=1.8, label=r"$\max_k \mathrm{Re}\,\lambda(k)$")
    else:
        d2s = [r["d2"] for r in rows]
        max_res = [r["max_re_lambda"] for r in rows]
        ax.plot(d2s, max_res, "o-", color="#4C72B0", lw=1.5, ms=5, label=r"$\max_k \mathrm{Re}\,\lambda(k)$")
    ax.axhline(0.0, color="black", lw=0.8, ls="--")
    ax.set_xlabel(r"predator diffusion $d_2$ ($d_1=1$)")
    ax.set_ylabel(r"$\max_k \mathrm{Re}\,\lambda(k)$")
    title_bits = ["Turing linear stability at numerical coexistence"]
    if u_star is not None and v_star is not None:
        title_bits.append(f"($u^*={u_star:.3f},\\ v^*={v_star:.3f}$)")
    ax.set_title(" ".join(title_bits), fontsize=10)
    if paper_ref is not None:
        ax.text(
            0.02,
            0.98,
            f"paper fig. note ({paper_ref[0]:.3f},{paper_ref[1]:.3f}): saddle, not used",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            color="#555555",
        )
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_turing_uv_pair(result, out: Path, title: str | None = None) -> None:
    """u and v panels for the same run."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, Z, lab in zip(
        axes,
        (result.u, result.v),
        ("prey u", "predator v"),
    ):
        im = ax.imshow(
            Z.T,
            origin="lower",
            extent=[0, result.config.lx, 0, result.config.ly],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_title(lab)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ttl = title or f"t={result.t_final:g}, d1={result.config.d1}, d2={result.config.d2}"
    fig.suptitle(ttl, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_fit_result(
    fit_result,
    out: Path,
    title: str | None = None,
) -> None:
    """Observed vs model fit (calibration output)."""
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    t = np.asarray(fit_result.t_obs, dtype=float)

    def _safe_series(*arrays):
        vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return 0.0, 1.0
        lo, hi = float(np.min(vals)), float(np.max(vals))
        if lo == hi:
            pad = max(abs(lo) * 0.1, 1.0)
            return lo - pad, hi + pad
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad

    def _fixed_ticks(ax, t_vals, y_lo, y_hi):
        if t_vals.size >= 2:
            ax.set_xticks(np.linspace(float(t_vals[0]), float(t_vals[-1]), min(6, t_vals.size)))
        ax.set_yticks(np.linspace(y_lo, y_hi, 5))

    prey_pred = np.asarray(fit_result.prey_pred, dtype=float)
    pred_pred = np.asarray(fit_result.predator_pred, dtype=float)
    prey_pred = np.where(np.isfinite(prey_pred), prey_pred, np.nan)
    pred_pred = np.where(np.isfinite(pred_pred), pred_pred, np.nan)

    y0_lo, y0_hi = _safe_series(fit_result.prey_obs, prey_pred)
    y1_lo, y1_hi = _safe_series(fit_result.predator_obs, pred_pred)

    axes[0].plot(t, fit_result.prey_obs, "ko", ms=4, label="obs")
    axes[0].plot(t, prey_pred, "b-", lw=1.5, label="model")
    axes[0].set_ylabel("prey")
    axes[0].set_ylim(y0_lo, y0_hi)
    _fixed_ticks(axes[0], t, y0_lo, y0_hi)
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, fit_result.predator_obs, "ko", ms=4, label="obs")
    axes[1].plot(t, pred_pred, "r-", lw=1.5, label="model")
    axes[1].set_xlabel("time")
    axes[1].set_ylabel("predator")
    axes[1].set_ylim(y1_lo, y1_hi)
    _fixed_ticks(axes[1], t, y1_lo, y1_hi)
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    ttl = title or f"{fit_result.series_name} — {fit_result.model} (RMSE={fit_result.rmse_total:.4g})"
    fig.suptitle(ttl)
    fig.subplots_adjust(hspace=0.32, top=0.90)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_k_eigenvalue_scan(rows, out: Path) -> None:
    """max Re(λ) and |Im(λ)| vs k (Hopf threshold)."""
    k = np.array([r.k for r in rows if r.re_max is not None])
    re_max = np.array([r.re_max for r in rows if r.re_max is not None])
    im_abs = np.array([r.im_abs for r in rows if r.re_max is not None])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(k, re_max, "b-o", ms=3)
    axes[0].axhline(0.0, color="k", lw=0.8, ls="--")
    axes[0].set_ylabel(r"max Re($\lambda$)")
    axes[0].set_title("Method 1: Jacobian eigenvalues vs k")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k, im_abs, "r-s", ms=3)
    axes[1].set_xlabel("fear parameter k")
    axes[1].set_ylabel(r"|Im($\lambda$)|")
    axes[1].grid(True, alpha=0.3)
    fig.subplots_adjust(hspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_k_amplitude_scan(rows, out: Path) -> None:
    """Numerical amplitude and relative amplitude vs k."""
    k = np.array([r.k for r in rows])
    amp = np.array([r.amplitude_u for r in rows])
    rel = np.array([r.rel_amplitude_u for r in rows])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    axes[0].plot(k, amp, "g-o", ms=3)
    axes[0].set_ylabel("prey amplitude (max-min)")
    axes[0].set_title("Method 2: long-run oscillation amplitude vs k")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(k, rel, "m-s", ms=3)
    axes[1].set_xlabel("fear parameter k")
    axes[1].set_ylabel("relative amplitude A / mean")
    axes[1].grid(True, alpha=0.3)
    fig.subplots_adjust(hspace=0.28)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_k_peak_decay(rows, out: Path) -> None:
    """Method 3: peak decay ratio vs k."""
    pts = [(r.k, r.peak_decay_ratio) for r in rows if r.peak_decay_ratio is not None]
    if not pts:
        return
    k, ratio = zip(*pts)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k, ratio, "c-o", ms=4)
    ax.axhline(1.0, color="k", lw=0.8, ls="--", label="no decay (ratio=1)")
    ax.set_xlabel("fear parameter k")
    ax.set_ylabel("last/first peak deviation ratio")
    ax.set_title("Method 3: peak decay ratio vs k (lower = weaker oscillation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_k_timeseries_panel(solutions: dict[str, object], out: Path) -> None:
    """Multi-k prey time series."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, sol in solutions.items():
        ax.plot(sol.t, sol.y[0], lw=1.4, label=label)
    ax.set_xlabel("time t")
    ax.set_ylabel("prey u")
    ax.set_title("Multi-k time series (Wang/Myint fear stabilization)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_k_phase_shrink(solutions: dict[str, object], out: Path) -> None:
    """Multi-k phase trajectories."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for label, sol in solutions.items():
        ax.plot(sol.y[0], sol.y[1], lw=1.2, label=label)
    ax.set_xlabel("prey u")
    ax.set_ylabel("predator v")
    ax.set_title("Phase plane: larger k -> spiral shrinks to equilibrium")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
