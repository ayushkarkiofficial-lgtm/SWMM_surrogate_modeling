# Surrogate-Assisted Monte Carlo Uncertainty Analysis of Urban Stormwater Drainage

Training machine-learning **surrogate models** (Random Forest, XGBoost) on EPA SWMM
runs to perform large-scale Monte Carlo flood-risk uncertainty analysis cheaply — instead
of running the slow physics model tens of thousands of times.

```
SWMM model  ->  LHS sampling  ->  automated batch runs  ->  dataset
            ->  ML surrogate  ->  Monte Carlo  ->  sensitivity analysis
```

See [`PROJECT_BRIEF.md`](PROJECT_BRIEF.md) for the full research plan.

## Repository layout

| Path | Contents |
|------|----------|
| `swmm/` | Base SWMM model (`base_model.inp`) + validation experiments |
| `notebooks/` | Analysis notebooks — `phase2_pipeline.ipynb` (LHS + automation) |
| `scripts/` | Helper scripts — `build_notebook.py` regenerates the notebook |
| `data/` | Generated datasets (`dataset.csv`) |
| `runs/` | Per-scenario `.inp`/`.rpt` files — **gitignored** (regenerable) |
| `docs/` | Model spec + design docs (`docs/superpowers/specs/`) |

## Project phases

- **Phase 1 — DONE.** Hand-built + validated the deterministic base model
  (`swmm/base_model.inp`): 3 subcatchments → J1 → C1 → J2 → C2 → Out1, 2-yr 2-hr design
  storm. Baseline peak outfall discharge ≈ 19.93 CFS, continuity < 0.5%. Spec in
  [`docs/base_model_spec.md`](docs/base_model_spec.md).
- **Phase 2 — IN PROGRESS.** LHS sampling of 5 uncertain parameters → automated SWMM
  batch → `dataset.csv`. See
  [`docs/superpowers/specs/2026-09-04-phase2-lhs-automation-design.md`](docs/superpowers/specs/2026-09-04-phase2-lhs-automation-design.md).
- **Phase 3+** — surrogate training, Monte Carlo, sensitivity (permutation / SHAP).

## Setup

Requires **Python 3.10+** and a local **EPA SWMM 5.2** install (for `runswmm.exe`).

```bash
pip install -r requirements.txt
```

Then open `notebooks/phase2_pipeline.ipynb` in JupyterLab.

> **Machine-specific path:** In Cell 1 of the notebook, set `SWMM_EXE` to your SWMM
> install, e.g. `D:\EPA SWMM 5.2.4 (64-bit)\runswmm.exe`. Everything else resolves
> relative to the repo root automatically.

## Phase 2 pipeline (`notebooks/phase2_pipeline.ipynb`)

1. **Config** — paths, run count, seed, parameter ranges.
2. **LHS sampling** — 100 points over 5 parameters (`scipy.stats.qmc`).
3. **Edit `.inp`** — apply each sample to the base model.
4. **Run + parse** — `runswmm.exe` → parse `.rpt` for targets.
5. **Batch loop** — 100 fault-isolated runs.
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
