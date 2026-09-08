"""
PRIM (Patient Rule Induction Method) scenario discovery for HIGH peak discharge.
Standalone; does NOT touch any notebook. Run with the isolated venv:
    .venv_viz/Scripts/python.exe scripts/prim_peak.py

PRIM finds an axis-aligned "box" = a conjunction of parameter ranges inside
which high-peak cases concentrate. Only SOME parameters get restricted, so the
box realizes the "sometimes 3 parameters intersect, sometimes 2" idea directly.

Uses the REAL simulated peak from data/dataset.csv (no surrogate needed).
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import prim

ROOT = Path(__file__).resolve().parent.parent
FEATS = ["rain_mult", "imperv_mult", "surf_n", "infil_min", "pipe_n"]
TARGET = "peak_discharge_cfs"
Q = 0.90                       # "high peak" = top 10%
DENSITY_MIN = 0.90             # purity we want when picking a box off the trajectory

df = pd.read_csv(ROOT / "data" / "dataset.csv")
df = df[df["status"] == "ok"].reset_index(drop=True)
x = df[FEATS]
y = df[TARGET].values
thr = np.quantile(y, Q)

p = prim.Prim(x, y, threshold=thr, threshold_type=">", peel_alpha=0.1)
box = p.find_box()
traj = box.peeling_trajectory.copy()

# --- pick a box: max coverage subject to density >= DENSITY_MIN (else max density) ---
ok = traj[traj["density"] >= DENSITY_MIN]
pick = (ok["coverage"].idxmax() if len(ok) else traj["density"].idxmax())
box.select(int(pick))
lim = box.limits                       # restricted dims only: min, max, qp values
cov = traj.loc[pick, "coverage"]
den = traj.loc[pick, "density"]
resdim = int(traj.loc[pick, "res dim"])

# full sampled ranges for normalisation
lo_all, hi_all = df[FEATS].min(), df[FEATS].max()

print(f"threshold (top {int((1-Q)*100)}%): peak > {thr:.2f} CFS  "
      f"({(y > thr).sum()} of {len(y)} runs)")
print(f"selected box: coverage={cov:.2f}  density={den:.2f}  restricted dims={resdim}")
print(lim)

# ----------------------------- figure -----------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5.2))

# LEFT: peeling trajectory (coverage vs density), colour = # restricted dims
sc = axL.scatter(traj["coverage"], traj["density"], c=traj["res dim"],
                 cmap="viridis", s=45, edgecolor="k", linewidth=0.3)
axL.scatter(cov, den, s=260, facecolors="none", edgecolors="red", linewidths=2.2,
            label="selected box", zorder=5)
axL.set_xlabel("coverage  (fraction of all high-peak cases captured)")
axL.set_ylabel("density  (fraction inside box that are high-peak = purity)")
axL.set_title(f"PRIM peeling trajectory — high peak (top {int((1-Q)*100)}%)")
axL.grid(alpha=0.25)
axL.legend(loc="lower left")
fig.colorbar(sc, ax=axL, label="# restricted parameters")

# RIGHT: the selected box as normalised range bars
axR.set_title(f"High-peak box:  peak > {thr:.1f} CFS   "
              f"(coverage {cov:.0%}, purity {den:.0%})")
for i, f in enumerate(FEATS):
    full = (0.0, 1.0)
    axR.barh(i, full[1] - full[0], left=full[0], height=0.5,
             color="lightgrey", zorder=1)
    if f in lim.index:                              # restricted -> dark band
        a = (lim.loc[f, "min"] - lo_all[f]) / (hi_all[f] - lo_all[f])
        b = (lim.loc[f, "max"] - lo_all[f]) / (hi_all[f] - lo_all[f])
        axR.barh(i, b - a, left=a, height=0.5, color="crimson", zorder=2)
        axR.text(1.02, i, f"[{lim.loc[f,'min']:.3g}, {lim.loc[f,'max']:.3g}]",
                 va="center", fontsize=9, color="crimson")
    else:
        axR.text(1.02, i, "unrestricted (full range)",
                 va="center", fontsize=9, color="grey")
axR.set_yticks(range(len(FEATS)))
axR.set_yticklabels(FEATS)
axR.set_xlim(0, 1)
axR.set_xlabel("normalised parameter range  (0 = min sampled, 1 = max sampled)")
axR.invert_yaxis()
axR.grid(axis="x", alpha=0.25)

fig.suptitle("PRIM scenario discovery — which overlapping parameter ranges "
             "produce HIGH peak discharge", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = ROOT / "docs" / "figures" / "prim_peak.png"
fig.savefig(out, dpi=130)
print("saved:", out)
