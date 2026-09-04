# Phase 6 Design — AD-Guided Adaptive Sampling

**Date:** 2026-09-05
**Project:** SWMM surrogate modeling (see `PROJECT_BRIEF.md`)
**Phase goal:** Close the loop — use the applicability domain to choose *where to run more
SWMM*, retrain, and show the surrogate improves in the weak regions Phase 5 identified.
Prove the AD-guidance beats plain "more data" via a controlled adaptive-vs-random comparison.

## 1. Setup

- Start from the 500-run training set + peak XGBoost surrogate + `scripts/swmm_utils.py`.
- **Fixed evaluation set:** reuse the 300 Phase-5 probes (`data/ad_probes.csv`), which already
  have SWMM ground-truth peak. Both strategies are judged on this identical held-out set.

## 2. Candidate pool

- ~3,000 fresh points over the wide box (1.5× training half-range each side, physically
  clipped), seed distinct from the probes. Candidates are proposals only — no SWMM yet.

## 3. Two selection strategies (add N_add = 150 each)

- **Adaptive (AD-guided maximin):** greedily select the 150 candidates farthest from the
  current training set in standardized feature space; each pick updates the running min-distance
  so selected points also spread from each other. Fills sparse / low-AD regions.
- **Random baseline:** 150 random candidates from the same pool.
- Run real SWMM on each selected set → true peak → new training rows.

## 4. Retrain & evaluate

- Retrain peak XGBoost (same hyperparameters) three ways: **baseline (500)**,
  **500 + 150 random**, **500 + 150 adaptive**.
- On the fixed eval set report **MAE, RMSE, R²** for all three.
- Supporting views: mean |error| by Mahalanobis-distance quintile (does adaptive flatten the
  far-region error?), and overall MAE bar chart.

## 5. Success criteria

- **Adaptive MAE < random MAE < baseline MAE** on the held-out set.
- Adaptive's largest gains fall in the **high-distance quintiles** (the out-of-AD tail Phase 5
  showed was ~2.6× worse).
- Headline: % error reduction of adaptive vs baseline, and adaptive vs same-budget random.

## 6. Artifacts

- `notebooks/phase6_adaptive_sampling.ipynb` via `scripts/build_phase6.py`.
- `data/adaptive_results.csv` — 3-way metrics comparison (tracked).
- ~300 new SWMM runs (150 adaptive + 150 random) in `runs/` (gitignored). No new dependencies.

## 7. Notes / honesty

- Eval and candidate pool use distinct seeds; both strategies see the identical eval set, so
  the adaptive-vs-random gap is a fair, apples-to-apples comparison (the key claim). Absolute
  reductions depend on the wide-box eval distribution.
- Single-batch selection + one retrain per strategy (iterative multi-round retraining is a
  possible future extension).
