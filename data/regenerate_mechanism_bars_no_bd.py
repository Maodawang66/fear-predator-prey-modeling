"""Regenerate mechanism bar chart without B-D mechanisms."""
import os, sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.literature import run_all_mechanisms, mechanism_labels
from src.parameters import MechanismId
from src.analysis import compare_mechanisms
from src.visualize import plot_mechanism_bars, plot_mechanism_comparison

# Only Holling II mechanisms (no B-D baseline, no B-D fear)
HOLLING_MECHANISMS = (
    MechanismId.BASELINE,
    MechanismId.FEAR_MEMORY,
    MechanismId.FEAR_INSTANT,
    MechanismId.FEAR_SATURATING,
    MechanismId.FEAR_FORAGING,
    MechanismId.FEAR_HANDLING,
)

out_dir = ROOT / "results"

# Regenerate mechanism time series
all_sols = run_all_mechanisms(t_span=(0.0, 80.0))
all_sols = {k: v for k, v in all_sols.items() if k in HOLLING_MECHANISMS}
labels = mechanism_labels()
labels = {k: v for k, v in labels.items() if k in HOLLING_MECHANISMS}
plot_mechanism_comparison(all_sols, labels, out_dir / "08_literature_mechanisms_timeseries.png")
print("Regenerated 08_literature_mechanisms_timeseries.png (no B-D)")

# Regenerate mechanism bar chart
cmp = compare_mechanisms(t_end=100.0, comparison_mode="equivalent", mechanisms=HOLLING_MECHANISMS)
plot_mechanism_bars(cmp, out_dir / "09_literature_mechanisms_bars.png")
print("Regenerated 09_literature_mechanisms_bars.png (no B-D)")
