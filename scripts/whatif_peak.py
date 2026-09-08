"""
Standalone what-if / ICE + PDP plot for the peak-discharge surrogate.

Does NOT touch any notebook. Loads the trained XGBoost peak regressor and the
dataset, then for each of the 5 input features:
  - sweeps that feature across its sampled range (x-axis),
  - draws 5 ICE lines: each line fixes the OTHER 4 features at the values of a
    real background scenario (chosen at the 5th/25th/50th/75th/95th percentile
    of observed peak), so you see how one variable moves peak while the rest are
    held fixed,
  - overlays the bold PDP line = average prediction over the WHOLE dataset as
    the feature is swept (the marginal effect),
  - marks the dataset mean of the feature with a dashed vertical line.

Parallel ICE lines  -> that feature acts additively (no interaction).
Fanning / crossing   -> that feature interacts with the ones held fixed.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
FEATS = ["rain_mult", "imperv_mult", "surf_n", "infil_min", "pipe_n"]
TARGET = "peak_discharge_cfs"
N_GRID = 60          # resolution of each sweep
N_PDP_BG = 300       # rows used to average the PDP (cap for speed)

df = pd.read_csv(ROOT / "data" / "dataset.csv")
df = df[df["status"] == "ok"].reset_index(drop=True)
model = joblib.load(ROOT / "models" / "peak_regressor.joblib")

lo = df[FEATS].min()
hi = df[FEATS].max()
mean = df[FEATS].mean()

# 5 background scenarios spread across the observed peak range
qs = [0.05, 0.25, 0.50, 0.75, 0.95]
bg = df.loc[[df[TARGET].sub(df[TARGET].quantile(q)).abs().idxmin() for q in qs]]

# rows for the PDP average (subsample for speed if large)
pdp_bg = df if len(df) <= N_PDP_BG else df.sample(N_PDP_BG, random_state=42)

colors = plt.cm.viridis(np.linspace(0, 0.9, len(qs)))

fig, axes = plt.subplots(1, 5, figsize=(22, 4.2), sharey=True)

for ax, f in zip(axes, FEATS):
    grid = np.linspace(lo[f], hi[f], N_GRID)

    # ---- ICE lines: one per background scenario ----
    for (idx, row), c, q in zip(bg.iterrows(), colors, qs):
        X = pd.DataFrame([row[FEATS].values] * N_GRID, columns=FEATS)
        X[f] = grid
        y = model.predict(X)
        ax.plot(grid, y, color=c, lw=1.6, alpha=0.9,
                label=f"peak p{int(q*100)} scenario")

    # ---- PDP: average prediction over many rows ----
    base = pd.concat([pdp_bg[FEATS]] * 1, ignore_index=True)
    pdp_y = []
    for g in grid:
        Xp = base.copy()
        Xp[f] = g
        pdp_y.append(model.predict(Xp).mean())
    ax.plot(grid, pdp_y, color="black", lw=3.0, label="PDP (average)")

    ax.axvline(mean[f], color="grey", ls="--", lw=1, alpha=0.7)
    ax.set_title(f)
    ax.set_xlabel(f)
    ax.grid(alpha=0.25)

axes[0].set_ylabel("predicted peak (CFS)")
axes[-1].legend(fontsize=8, loc="upper left", framealpha=0.9)
fig.suptitle("What-if / ICE profiles for peak discharge  "
             "(each line fixes the other 4 vars; bold = PDP average)",
             fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.94])

out = ROOT / "docs" / "figures" / "whatif_ice_peak.png"
fig.savefig(out, dpi=130)
print("saved:", out)
