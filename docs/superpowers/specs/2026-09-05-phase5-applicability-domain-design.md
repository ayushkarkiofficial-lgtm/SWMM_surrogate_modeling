# Phase 5 Design — Applicability Domain + Error-Link Validation

**Date:** 2026-09-05
**Project:** SWMM surrogate modeling (see `PROJECT_BRIEF.md`)
**Phase goal:** Define an applicability domain (AD) for the surrogate and validate the core
hypothesis — **distance from the training domain predicts surrogate error** — using real SWMM
ground truth. Mirrors the user's landslide-susceptibility AD work
(`ad-aware-surrogate-direction` memory). Adaptive sampling deferred to Phase 6.

## 1. AD definitions (fit on the 500 training scenarios)

- **Range-based** (baseline): inside if every feature is within its training min–max.
- **Mahalanobis distance** `D_M(x)=√((x−μ)ᵀΣ⁻¹(x−μ))` using training μ, Σ on raw features
  (scale-invariant; captures feature correlations). Threshold = 99th percentile of training
  D_M (χ² df=5 reported for reference).
- **k-NN distance**: mean **standardized** (z-scored) distance to the k=10 nearest training
  points. Threshold = 99th percentile of training kNN distance.

Mahalanobis & kNN are the continuous AD scores; range-based is the coverage baseline.

## 2. AD coverage on the existing Monte Carlo

Score the 50k MC samples (`data/mc_results.csv`); report AD coverage (% in-domain) per metric.
Expected contrast: range-based ≈ 100% (MC drawn inside ranges) but Mahalanobis/kNN < 100%
(flag unusual corner combinations) — demonstrating why range-based AD is too weak.

## 3. Core experiment — does AD distance predict error?

1. Generate **~300 probe points** from a box **1.5× the training half-range** each side
   (50% wider), clipped to physical limits (rainfall/imperv mult > 0.1; roughness > 0.005;
   `infil_min` in [0.01, 4.0] < Horton MaxRate).
2. Run **real SWMM** on each probe (reuse the pipeline via `scripts/swmm_utils.py`) →
   ground-truth peak discharge.
3. Surrogate predicts peak → compute **|error|** per probe.
4. Compute each probe's Mahalanobis & kNN distance.
5. **Correlate:** |error| vs AD distance — Spearman/Pearson coefficients, scatter colored by
   in/out of range, quintile-binned mean error, and in-domain vs out-of-domain error.
   **Hypothesis: distance ↑ → error ↑.**
6. **Practical payoff:** show mean error for in-AD vs out-AD probes (filtering to in-AD should
   cut error).

Primary error target: **peak discharge** (cleanest continuous target, defined for every point).

## 4. Artifacts

- `scripts/swmm_utils.py` — `write_scenario` / `run_swmm(inp, swmm_exe)` / `parse_rpt`
  refactored into an importable module (copied from the validated Phase 2 logic).
- `notebooks/phase5_applicability_domain.ipynb` via `scripts/build_phase5.py`.
- `data/ad_probes.csv` (probes + SWMM truth + preds + distances + error) and
  `data/ad_coverage.csv` — tracked (small).
- ~300 probe SWMM runs (~11 s) in `runs/` (gitignored). No new dependencies (scipy + sklearn).

## 5. Success criteria

- AD coverage table shows range-based ≈ 100% while Mahalanobis/kNN < 100% (weakness of range
  AD demonstrated).
- **Positive, significant correlation** between AD distance and |error| (Spearman ρ > 0,
  p small) — the headline finding.
- In-AD probes have visibly lower mean error than out-AD probes.

## 6. Out of scope → Phase 6

AD-guided adaptive sampling: find weak/low-AD regions, run extra SWMM there, retrain, show
improvement.
