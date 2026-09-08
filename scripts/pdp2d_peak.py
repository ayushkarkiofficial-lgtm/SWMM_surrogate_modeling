"""
Standalone 2-D partial-dependence heatmaps for the peak-discharge surrogate.
Does NOT touch any notebook.

For each interesting parameter pair (A, B):
  - build a grid over A x B,
  - for every (a, b) cell, set feature A=a and B=b across a sample of real rows,
    hold the other 3 features at their real values, predict peak, average -> cell,
  - draw filled contour (color = predicted peak) with contour lines overlaid.

Reading it:
  straight, evenly-spaced, parallel bands  -> no interaction
  curved / twisting / uneven-spaced bands  -> interaction (effect of one
                                              parameter depends on the other)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FEATS = ["rain_mult", "imperv_mult", "surf_n", "infil_min", "pipe_n"]
PAIRS = [("rain_mult", "pipe_n"),
         ("rain_mult", "imperv_mult"),
         ("imperv_mult", "pipe_n")]
N_GRID = 40          # cells per axis
N_BG = 200           # background rows averaged per cell

df = pd.read_csv(ROOT / "data" / "dataset.csv")
df = df[df["status"] == "ok"].reset_index(drop=True)
model = joblib.load(ROOT / "models" / "peak_regressor.joblib")

lo, hi = df[FEATS].min(), df[FEATS].max()
bg = df if len(df) <= N_BG else df.sample(N_BG, random_state=42)

fig, axes = plt.subplots(1, len(PAIRS), figsize=(6.2 * len(PAIRS), 5.2))

for ax, (fa, fb) in zip(axes, PAIRS):
    ga = np.linspace(lo[fa], hi[fa], N_GRID)
    gb = np.linspace(lo[fb], hi[fb], N_GRID)
    Z = np.zeros((N_GRID, N_GRID))          # rows = b, cols = a
    base = bg[FEATS].to_numpy()
    ia, ib = FEATS.index(fa), FEATS.index(fb)
    for j, b in enumerate(gb):
        for i, a in enumerate(ga):
            X = base.copy()
            X[:, ia] = a
            X[:, ib] = b
            Z[j, i] = model.predict(pd.DataFrame(X, columns=FEATS)).mean()

    cf = ax.contourf(ga, gb, Z, levels=18, cmap="viridis")
    cl = ax.contour(ga, gb, Z, levels=9, colors="white", linewidths=0.7, alpha=0.7)
    ax.clabel(cl, inline=True, fontsize=7, fmt="%.0f")
    ax.set_xlabel(fa)
    ax.set_ylabel(fb)
    ax.set_title(f"peak  vs  {fa} x {fb}")
    fig.colorbar(cf, ax=ax, label="predicted peak (CFS)")

fig.suptitle("2-D partial dependence: parameter interactions on peak discharge "
             "(curved/uneven contours = interaction)", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out = ROOT / "docs" / "figures" / "pdp2d_peak.png"
fig.savefig(out, dpi=130)
print("saved:", out)
