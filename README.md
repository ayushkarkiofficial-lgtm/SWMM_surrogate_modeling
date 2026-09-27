# Surrogate-Assisted Monte Carlo Uncertainty Analysis of Urban Stormwater Drainage

Training machine-learning **surrogate models** (Random Forest, XGBoost) on EPA SWMM
runs to perform large-scale Monte Carlo flood-risk uncertainty analysis cheaply — instead
of running the slow physics model tens of thousands of times — with an
**applicability-domain reliability layer** that flags when the surrogate is extrapolating.

```
SWMM model  ->  LHS sampling  ->  automated batch runs  ->  dataset
            ->  ML surrogate  ->  50k Monte Carlo  ->  sensitivity analysis
            ->  applicability-domain gate  ->  AD-guided adaptive sampling
```

See [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for the full research plan and
[`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) for a figure-by-figure walkthrough.

## Status: complete (Phases 1–6)

All six phases are implemented and run end-to-end. Headline results below; the design
note for each phase is in [`docs/superpowers/specs/`](docs/superpowers/specs/).

- **Phase 1 — DONE.** Hand-built + validated the deterministic base model
  (`swmm/base_model.inp`): 3 subcatchments → J1 → C1 → J2 → C2 → Out1, 2-yr 2-hr design
  storm. Baseline peak outfall discharge ≈ 19.93 CFS, continuity < 0.5%. Spec in
  [`docs/base_model_spec.md`](docs/base_model_spec.md).
- **Phase 2 — DONE.** Latin-Hypercube sampling of 5 uncertain parameters → automated,
  fault-isolated SWMM batch → **500-run** `data/dataset.csv`.
  ([`notebooks/phase2_pipeline.ipynb`](notebooks/phase2_pipeline.ipynb))
- **Phase 3 — DONE.** Random Forest / XGBoost surrogates for peak discharge, flood
  occurrence and flood volume, with cross-validation and permutation importance.
  ([`notebooks/phase3_surrogate.ipynb`](notebooks/phase3_surrogate.ipynb))
- **Phase 4 — DONE.** **50,000-sample Monte Carlo** flood-risk uncertainty analysis on
  the trained surrogate, plus permutation and SHAP sensitivity.
  ([`notebooks/phase4_montecarlo.ipynb`](notebooks/phase4_montecarlo.ipynb))
- **Phase 5 — DONE.** Mahalanobis-distance **applicability-domain gate** — surrogate
  error rises with distance from the training set, so the fast model defers when it is
  out of distribution. ([`notebooks/phase5_applicability_domain.ipynb`](notebooks/phase5_applicability_domain.ipynb))
- **Phase 6 — DONE.** AD-guided **adaptive sampling** (greedy maximin) benchmarked
  against random augmentation, targeting sparse / low-AD regions.
  ([`notebooks/phase6_adaptive_sampling.ipynb`](notebooks/phase6_adaptive_sampling.ipynb))

## Headline results

Surrogate accuracy on held-out data (`data/phase3_metrics.csv`):

| Target | Model | Metric | Score |
|--------|-------|--------|-------|
| Peak discharge | XGBoost | R² | **0.98** (CV R² 0.98) |
| Flood occurrence | XGBoost | ROC-AUC | **0.99** (accuracy 0.97) |
| Flood volume | XGBoost | R² | **0.92** |

- **Monte Carlo (`data/mc_summary.csv`):** 50,000 surrogate evaluations → P(flood) ≈ 33%,
  peak-discharge P90 / P95 / P99 ≈ 25.0 / 27.2 / 30.8 CFS — a sweep that is impractical at
  SWMM's native runtime.
- **Applicability domain (`data/ad_coverage.csv`):** Mahalanobis gate covers ≈ 98.8% of the
  Monte Carlo cloud; held-out error rises monotonically with distance from the training set.
- **Adaptive sampling (`data/adaptive_results.csv`):** held-out MAE on peak discharge falls
  from **1.47 CFS** (baseline, 500 pts) → 0.76 (+150 random) → **0.73 (+150 AD-guided)**;
  the AD-guided budget cuts far-tail (out-of-domain) error hardest.

## Repository layout

| Path | Contents |
|------|----------|
| `swmm/` | Base SWMM model (`base_model.inp`) + validation experiments |
| `notebooks/` | `phase2_pipeline` → `phase3_surrogate` → `phase4_montecarlo` → `phase5_applicability_domain` → `phase6_adaptive_sampling` |
| `scripts/` | Helper / notebook-build scripts (`build_phase*.py`, `swmm_utils.py`) |
| `data/` | Generated datasets and result tables (`dataset.csv`, `phase3_metrics.csv`, `mc_summary.csv`, `ad_coverage.csv`, `adaptive_results.csv`) |
| `docs/` | Model spec, `WALKTHROUGH.md`, phase figures, and design specs (`docs/superpowers/specs/`) |
| `runs/` | Per-scenario `.inp`/`.rpt` files — **gitignored** (regenerable) |

## Setup

Requires **Python 3.10+** and a local **EPA SWMM 5.2** install (for `runswmm.exe`).

```bash
pip install -r requirements.txt
```

Then open the notebooks in JupyterLab, in phase order.

> **Machine-specific path:** In Cell 1 of `phase2_pipeline.ipynb`, set `SWMM_EXE` to your
> SWMM install, e.g. `D:\EPA SWMM 5.2.4 (64-bit)\runswmm.exe`. Everything else resolves
> relative to the repo root automatically.

## Data-generation pipeline (`notebooks/phase2_pipeline.ipynb`)

1. **Config** — paths, run count, seed, parameter ranges.
2. **LHS sampling** — Latin-Hypercube points over 5 parameters (`scipy.stats.qmc`).
3. **Edit `.inp`** — apply each sample to the base model.
4. **Run + parse** — `runswmm.exe` → parse `.rpt` for targets.
5. **Batch loop** — 500 fault-isolated runs.
6. **Assemble** — save `data/dataset.csv` + sanity plots.
7. **pyswmm demo** — optional in-process cross-check.

### Uncertain parameters

| Feature | Change | Range |
|---------|--------|-------|
| `rain_mult` | Rainfall timeseries multiplier | 0.7 – 1.5 |
| `imperv_mult` | %Imperv multiplier (per subcatchment) | 0.7 – 1.3 |
| `surf_n` | Impervious overland Manning n | 0.011 – 0.030 |
| `infil_min` | Horton min infiltration rate (in/hr) | 0.05 – 1.5 |
| `pipe_n` | Conduit Manning roughness | 0.011 – 0.030 |

### Targets

Peak outfall discharge (CFS), total flooding volume (10⁶ gal), number of flooded nodes.
