"""Regenerate six-model heatmap without B-D column (Phase 3 item 44).

Reads report_protocol_seven_model_metrics.csv, removes B-D column,
and regenerates with the same plotting parameters as the original.
"""
import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "seven_model_real_fits"
CSV_PATH = OUT / "report_protocol_seven_model_metrics.csv"

# Original MODEL_ORDER without bda_fear
MODEL_ORDER = (
    "baseline",
    "fear_instant",
    "fear_memory",
    "fear_saturating",
    "fear_foraging",
    "fear_handling",
)
LABELS = {
    "baseline": "Baseline",
    "fear_instant": "Instant reproduction fear",
    "fear_memory": "Memory reproduction fear",
    "fear_saturating": "Saturating reproduction fear",
    "fear_foraging": "Foraging/attack suppression",
    "fear_handling": "Handling-time extension",
}

# Read CSV
rows = []
with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
    for row in csv.DictReader(handle):
        if row["model"] == "bda_fear":
            continue
        rows.append({
            "series": row["series"],
            "model": row["model"],
            "validation_rmse": float(row["validation_rmse"]),
            "aicc": float(row["aicc"]),
            "optimization_status": row["optimization_status"],
            "usable_for_comparison": row["usable_for_comparison"] == "True",
        })

# Build data matrix
series_names = list(dict.fromkeys(row["series"] for row in rows))
lookup = {(row["series"], row["model"]): row for row in rows}
values = np.full((len(series_names), len(MODEL_ORDER)), np.nan)
for i, series in enumerate(series_names):
    for j, model in enumerate(MODEL_ORDER):
        row = lookup[(series, model)]
        if row["usable_for_comparison"]:
            values[i, j] = row["validation_rmse"]

row_min = np.nanmin(values, axis=1, keepdims=True)
ratios = values / np.maximum(row_min, np.finfo(float).eps)
masked = np.ma.masked_invalid(np.log10(ratios))

# Plot with same parameters as original
fig, ax = plt.subplots(figsize=(12.5, 7.5))
cmap = plt.get_cmap("viridis").copy()
cmap.set_bad("#d9d9d9")
image = ax.imshow(masked, cmap=cmap, aspect="auto", vmin=0)

for i, series in enumerate(series_names):
    for j, model in enumerate(MODEL_ORDER):
        row = lookup[(series, model)]
        text = f"{row['validation_rmse']:.3g}"
        if not row["usable_for_comparison"]:
            text = "failed"
        ax.text(
            j, i, text,
            ha="center", va="center", fontsize=7.5,
            color="black" if not row["usable_for_comparison"] else "white",
        )

ax.set_xticks(
    range(len(MODEL_ORDER)),
    [LABELS[model] for model in MODEL_ORDER],
    rotation=35,
    ha="right",
)
ax.set_yticks(range(len(series_names)), series_names)
ax.set_title(
    "Six-model 20% continuous multi-step holdout RMSE\n"
    "(colors: log10 ratio to best available model for each series; darker = lower RMSE)",
    fontsize=11.5,
    fontweight="bold",
)
# Colorbar
cbar = fig.colorbar(image, ax=ax, shrink=0.78, pad=0.02)
cbar.set_label("log10(validation RMSE / best-in-series)", fontsize=9)
fig.tight_layout()

# Save to the requested filename
out_path = OUT / "validation_rmse_heatmap_six_model.png"
fig.savefig(out_path, dpi=190)
plt.close(fig)
print(f"Regenerated heatmap: {out_path}")

# Also save as the default filename copy
default_out = OUT / "validation_rmse_heatmap.png"
fig2, ax2 = plt.subplots(figsize=(12.5, 7.5))
image2 = ax2.imshow(masked, cmap=cmap, aspect="auto", vmin=0)
for i, series in enumerate(series_names):
    for j, model in enumerate(MODEL_ORDER):
        row = lookup[(series, model)]
        text = f"{row['validation_rmse']:.3g}"
        if not row["usable_for_comparison"]:
            text = "failed"
        ax2.text(
            j, i, text,
            ha="center", va="center", fontsize=7.5,
            color="black" if not row["usable_for_comparison"] else "white",
        )
ax2.set_xticks(range(len(MODEL_ORDER)), [LABELS[model] for model in MODEL_ORDER], rotation=35, ha="right")
ax2.set_yticks(range(len(series_names)), series_names)
ax2.set_title(
    "Six-model 20% continuous multi-step holdout RMSE\n"
    "(colors: log10 ratio to best available model for each series; darker = lower RMSE)",
    fontsize=11.5, fontweight="bold",
)
cbar2 = fig2.colorbar(image2, ax=ax2, shrink=0.78, pad=0.02)
cbar2.set_label("log10(validation RMSE / best-in-series)", fontsize=9)
fig2.tight_layout()
fig2.savefig(default_out, dpi=190)
plt.close(fig2)
print(f"Saved default copy: {default_out}")
