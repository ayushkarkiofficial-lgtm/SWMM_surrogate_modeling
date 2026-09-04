"""Generate phase4_montecarlo.ipynb (Phase 4: Monte Carlo + sensitivity).

Run from repo root:  python scripts/build_phase4.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))


def code(src):
    cells.append(nbf.v4.new_code_cell(src))


md(
    "# Phase 4 — Monte Carlo + Sensitivity Analysis\n"
    "\n"
    "Push 50,000 uncertain scenarios through the trained surrogates to get "
    "**probabilistic flood outputs**, then run **permutation + SHAP** sensitivity.\n"
    "\n"
    "Design: `docs/superpowers/specs/2026-09-04-phase4-montecarlo-sensitivity-design.md`.\n"
    "\n"
    "> **Prerequisite:** Phase 3 must have saved `models/*.joblib`. Needs `pip install shap`."
)

md("## 1. Config & load surrogates")
code(
    r'''from pathlib import Path
import numpy as np
import pandas as pd
import joblib

def _find_root(marker="PROJECT_BRIEF.md"):
    here = Path.cwd()
    for cand in [here, *here.parents]:
        if (cand / marker).exists():
            return cand
    return here

PROJECT    = _find_root()
MODELS_DIR = PROJECT / "models"
DATA_DIR   = PROJECT / "data"
SEED  = 42
N_MC  = 50_000

FEATURES = ["rain_mult", "imperv_mult", "surf_n", "infil_min", "pipe_n"]
PARAM_RANGES = {
    "rain_mult":   (0.7, 1.5),
    "imperv_mult": (0.7, 1.3),
    "surf_n":      (0.011, 0.030),
    "infil_min":   (0.05, 1.5),
    "pipe_n":      (0.011, 0.030),
}

peak_reg  = joblib.load(MODELS_DIR / "peak_regressor.joblib")
flood_clf = joblib.load(MODELS_DIR / "flood_classifier.joblib")
vol_reg   = joblib.load(MODELS_DIR / "flood_volume_regressor.joblib")
print("Loaded 3 surrogate models | N_MC =", N_MC)'''
)

md(
    "## 2. Monte Carlo sampling + hard two-stage prediction\n"
    "\n"
    "Draw 50k uniform samples over the training ranges, then: peak from the regressor, "
    "flood yes/no from the classifier, and volume from the volume regressor **only when the "
    "classifier says flood** (else 0)."
)
code(
    r'''rng   = np.random.default_rng(SEED)
lows  = np.array([PARAM_RANGES[f][0] for f in FEATURES])
highs = np.array([PARAM_RANGES[f][1] for f in FEATURES])
Xmc = pd.DataFrame(rng.uniform(lows, highs, size=(N_MC, len(FEATURES))), columns=FEATURES)

peak  = peak_reg.predict(Xmc)
flood = flood_clf.predict(Xmc).astype(int)
vol   = np.where(flood == 1, vol_reg.predict(Xmc), 0.0)
vol   = np.clip(vol, 0, None)          # flood volume can't be negative

mc = Xmc.copy()
mc["peak_discharge_cfs"] = peak
mc["flooded"]            = flood
mc["flood_vol"]          = vol
print(f"Monte Carlo complete: {N_MC:,} samples")
mc.head()'''
)

md(
    "## 3. Probabilistic outputs\n"
    "\n"
    "P(flooding), percentiles, and exceedance statements. Sanity: P(flood) should be near "
    "the ~35% training rate and mean peak near ~19.5 CFS."
)
code(
    r'''p_flood = mc["flooded"].mean()
flooded_vol = mc.loc[mc["flooded"] == 1, "flood_vol"]
pct = [50, 90, 95, 99]

print(f"P(flooding) = {p_flood*100:.1f}%   (training rate ~35% -> sanity check)")
print(f"mean peak   = {mc['peak_discharge_cfs'].mean():.2f} CFS  (baseline ~19.9)\n")

print("Peak discharge (CFS) percentiles:")
for q in pct:
    print(f"  P{q:<2d}: {np.percentile(mc['peak_discharge_cfs'], q):6.2f}")

print("\nFlood volume (10^6 gal), flooded samples only:")
for q in pct:
    print(f"  P{q:<2d}: {np.percentile(flooded_vol, q):.4f}  ({np.percentile(flooded_vol, q)*1e6:,.0f} gal)")

print("\nExceedance probabilities:")
for thr in [20, 25, 30]:
    print(f"  P(peak > {thr} CFS)          = {(mc['peak_discharge_cfs'] > thr).mean()*100:5.1f}%")
for v in [0.005, 0.010, 0.020]:
    print(f"  P(flood_vol > {v*1e6:>6,.0f} gal) = {(mc['flood_vol'] > v).mean()*100:5.1f}%")'''
)

md("### Distributions + exceedance curves")
code(
    r'''import matplotlib.pyplot as plt

fig, ax = plt.subplots(2, 2, figsize=(13, 9))
ax[0, 0].hist(mc["peak_discharge_cfs"], bins=50, color="#3b7dd8", edgecolor="white")
ax[0, 0].set_title("Peak discharge distribution"); ax[0, 0].set_xlabel("CFS")
ax[0, 1].hist(flooded_vol, bins=50, color="#d8663b", edgecolor="white")
ax[0, 1].set_title("Flood volume (flooded samples)"); ax[0, 1].set_xlabel("10^6 gal")

xs  = np.sort(mc["peak_discharge_cfs"].values)
exc = 1 - np.arange(1, len(xs) + 1) / len(xs)
ax[1, 0].plot(xs, exc); ax[1, 0].set_title("Peak exceedance  P(peak > x)")
ax[1, 0].set_xlabel("CFS"); ax[1, 0].set_ylabel("probability"); ax[1, 0].grid(alpha=0.3)

vs   = np.sort(flooded_vol.values)
excv = 1 - np.arange(1, len(vs) + 1) / len(vs)
ax[1, 1].plot(vs, excv, color="#d8663b"); ax[1, 1].set_title("Flood-volume exceedance (flooded)")
ax[1, 1].set_xlabel("10^6 gal"); ax[1, 1].set_ylabel("probability"); ax[1, 1].grid(alpha=0.3)
plt.tight_layout(); plt.show()'''
)

md(
    "## 4. Sensitivity — permutation importance (all 3 targets)\n"
    "\n"
    "Computed on the labeled dataset. This is where we finally see whether infiltration & "
    "imperviousness matter more for *flooding* than they did for peak discharge."
)
code(
    r'''from sklearn.inspection import permutation_importance

df = pd.read_csv(DATA_DIR / "dataset.csv")
df = df[df["status"] == "ok"].reset_index(drop=True)
Xd = df[FEATURES]
flooded_mask = df["total_flood_vol"] > 0

imp = {}
r = permutation_importance(peak_reg, Xd, df["peak_discharge_cfs"],
                           n_repeats=15, random_state=SEED, scoring="r2")
imp["peak"] = pd.Series(r.importances_mean, index=FEATURES)

r = permutation_importance(flood_clf, Xd, flooded_mask.astype(int),
                           n_repeats=15, random_state=SEED, scoring="roc_auc")
imp["floods?"] = pd.Series(r.importances_mean, index=FEATURES)

r = permutation_importance(vol_reg, Xd[flooded_mask], df.loc[flooded_mask, "total_flood_vol"],
                           n_repeats=15, random_state=SEED, scoring="r2")
imp["flood_volume"] = pd.Series(r.importances_mean, index=FEATURES)

imp_df = pd.DataFrame(imp)
print(imp_df.round(3))
imp_df.plot.bar(figsize=(10, 4))
plt.title("Permutation importance by target"); plt.ylabel("importance (score drop)")
plt.xticks(rotation=0); plt.tight_layout(); plt.show()'''
)

md(
    "## 5. Sensitivity — SHAP (per-feature contributions + direction)\n"
    "\n"
    "TreeExplainer on each XGBoost model. Beeswarm plots show both magnitude and "
    "direction (red = high feature value, blue = low)."
)
code(
    r'''import shap

Xexp = Xd.sample(min(800, len(Xd)), random_state=SEED)

for name, model in [("peak", peak_reg), ("floods?", flood_clf), ("flood_volume", vol_reg)]:
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xexp)
    print(f"\n===== SHAP summary: {name} =====")
    shap.summary_plot(sv, Xexp, show=False)
    plt.title(f"SHAP — {name}"); plt.tight_layout(); plt.show()'''
)

md("## 6. Save results")
code(
    r'''mc.to_csv(DATA_DIR / "mc_results.csv", index=False)          # full 50k (gitignored)

summary = {
    "N_MC":                 N_MC,
    "P_flood_pct":          round(p_flood * 100, 2),
    "peak_mean":            round(mc["peak_discharge_cfs"].mean(), 2),
    "peak_P90":             round(np.percentile(mc["peak_discharge_cfs"], 90), 2),
    "peak_P95":             round(np.percentile(mc["peak_discharge_cfs"], 95), 2),
    "peak_P99":             round(np.percentile(mc["peak_discharge_cfs"], 99), 2),
    "floodvol_P95_flooded": round(float(np.percentile(flooded_vol, 95)), 5),
    "floodvol_P99_flooded": round(float(np.percentile(flooded_vol, 99)), 5),
}
pd.DataFrame([summary]).to_csv(DATA_DIR / "mc_summary.csv", index=False)
imp_df.to_csv(DATA_DIR / "sensitivity_permutation.csv")

print("Saved -> data/mc_summary.csv, data/mc_results.csv, data/sensitivity_permutation.csv")
pd.DataFrame([summary])'''
)

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

with open("notebooks/phase4_montecarlo.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)
print("Wrote notebooks/phase4_montecarlo.ipynb with", len(cells), "cells")
