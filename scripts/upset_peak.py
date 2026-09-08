"""
UpSet plot: which OVERLAPS of peak-favourable parameter conditions co-occur,
and how peak discharge is distributed in each overlap.
Standalone; does NOT touch any notebook. Run with the isolated venv:
    .venv_viz/Scripts/python.exe scripts/upset_peak.py

Your "low/med/high overlapping regions" idea, made concrete:
each parameter is turned into a boolean membership set for its PEAK-RAISING
extreme tercile (rain high, imperv high, pipe low, surf low, infil low).
The UpSet matrix shows every combination (intersection) of those conditions
that actually occurs; the box plot on top shows the peak distribution for each
combination -> combinations that push peak highest become obvious.

Uses the REAL simulated peak from data/dataset.csv (no surrogate needed).
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from upsetplot import UpSet, from_indicators

ROOT = Path(__file__).resolve().parent.parent
TARGET = "peak_discharge_cfs"

df = pd.read_csv(ROOT / "data" / "dataset.csv")
df = df[df["status"] == "ok"].reset_index(drop=True)

# peak-raising extreme = top tercile for rain/imperv, bottom tercile for pipe/surf/infil
def top(col):    return df[col] >= df[col].quantile(2 / 3)
def bottom(col): return df[col] <= df[col].quantile(1 / 3)

df["rain=high"]   = top("rain_mult")
df["imperv=high"] = top("imperv_mult")
df["pipe=low"]    = bottom("pipe_n")
df["surf=low"]    = bottom("surf_n")
df["infil=low"]   = bottom("infil_min")
IND = ["rain=high", "imperv=high", "pipe=low", "surf=low", "infil=low"]

data = from_indicators(IND, data=df)

u = UpSet(data, subset_size="count", min_subset_size=6,
          sort_by="cardinality", show_counts=True, element_size=42)
u.add_catplot(value=TARGET, kind="box", color="crimson")
u.plot()

fig = plt.gcf()
fig.suptitle("UpSet — overlaps of peak-raising parameter conditions "
             "(box plot = peak discharge in each overlap)", fontsize=12)
out = ROOT / "docs" / "figures" / "upset_peak.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved:", out)
print(f"(min_subset_size=6; conditions = {IND})")
