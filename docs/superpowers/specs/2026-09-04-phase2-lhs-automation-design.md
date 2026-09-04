# Phase 2 Design — LHS Sampling + SWMM Automation Pipeline

**Date:** 2026-09-04
**Project:** Surrogate-Assisted Monte Carlo Uncertainty Analysis of Urban Stormwater Drainage (see `PROJECT_BRIEF.md`)
**Phase goal:** Build the full Python automation pipeline in a Jupyter notebook and validate it end-to-end with a 100-run pilot ("quick automation test", Brief §14).

## 1. Scope

In scope this session:
- Latin Hypercube sampling of 5 uncertain parameters.
- Programmatic editing of `base_model.inp` per sample.
- Running each scenario via the `runswmm.exe` CLI.
- Parsing outputs from the `.rpt` report.
- Assembling `dataset.csv` (features + targets + QC).
- A short `pyswmm` demo cell for learning the in-process API.

Out of scope (later phases): surrogate training, Monte Carlo, sensitivity analysis, the full 300–500 run training set.

## 2. Baseline model facts (from `base_model.inp`)

- 3 subcatchments S1/S2/S3 → J1 → C1 → J2 → C2 → Out1.
- Infiltration method: **HORTON**. Per-subcatchment params: MaxRate 4.5, MinRate 0.2, Decay 6.5/hr, DryTime 7, MaxInfil 0.
- Rainfall: `[TIMESERIES]` named `2-yr` (24 intensity values, in/hr), peak 2.85 at t=30 min. Raingage named `1`, INTENSITY format.
- %Imperv: S1=56.8, S2=63, S3=39.5.
- Surface roughness N-Imperv = 0.015 (all); pipe Roughness = 0.016 (C1, C2).
- Baseline outputs: peak outfall discharge ≈ 19.93 CFS, zero flooding, continuity error < 0.5%.

**Key modeling note:** Horton decay 6.5/hr drops infiltration from 4.5 → ~0.37 in/hr within ~30 min, before the storm peak. Therefore MaxRate barely influences peak runoff; **MinRate is the storm-long loss knob and is the parameter we vary.**

## 3. Uncertain parameters (5)

| # | Feature name | `.inp` change | Baseline | Range | Distribution | Justification |
|---|--------------|---------------|---------|-------|--------------|---------------|
| X1 | `rain_mult` | Multiply every `[TIMESERIES] 2-yr` value | 1.0 | 0.7 – 1.5 | Uniform | Storm severity |
| X2 | `imperv_mult` | Multiply each subcatchment %Imperv, clip to [0,100] | 1.0 | 0.7 – 1.3 | Uniform | Urbanization; preserves S1/S2/S3 pattern |
| X3 | `surf_n` | Set `[SUBAREAS]` N-Imperv on all subcatchments | 0.015 | 0.011 – 0.030 | Uniform | Overland roughness |
| X4 | `infil_min` | Set Horton MinRate (Param2) on all subcatchments (in/hr) | 0.2 | 0.05 – 1.5 | Uniform | Saturated soil infiltration |
| X5 | `pipe_n` | Set `[CONDUITS]` Roughness on C1 & C2 | 0.016 | 0.011 – 0.030 | Uniform | Pipe interior condition |

**Feature encoding:** X1 and X2 are recorded in `dataset.csv` as multipliers (e.g. 1.15); X3–X5 are recorded as absolute values. This is the natural knob for each; documented so ML feature semantics are unambiguous.

**Constraint:** X4 MinRate must stay below MaxRate (4.5). Range max 1.5 satisfies this with margin.

## 4. Targets (parsed from `.rpt`)

1. `peak_discharge_cfs` — peak flow at outfall Out1 [CFS].
2. `total_flood_vol` — total system flooding volume [10⁶ gal, as reported by SWMM].
3. `num_flooded_nodes` — count of nodes reporting flooding.

Plus QC columns:
- `continuity_error_pct` — flow routing continuity error.
- `status` — `ok` / `failed` / `flagged` (e.g., continuity error above a threshold).

## 5. Pipeline architecture (notebook `phase2_pipeline.ipynb`)

- **Cell 1 — Config:** paths (`runswmm.exe`, base model, `runs/` dir), `N_RUNS=100`, `SEED`, parameter-range dict.
- **Cell 2 — LHS sampling:** `scipy.stats.qmc.LatinHypercube(d=5, seed=SEED)` → scale to ranges → `samples` DataFrame; display for coverage check.
- **Cell 3 — `.inp` editing:** `write_scenario(base_text, params) -> str`. Section-aware edits for the 5 parameters. Unit-tested on one sample and diffed against base before batch run.
- **Cell 4 — Run + parse:** `run_scenario(inp_path) -> dict`. `subprocess` call to `runswmm.exe`; regex parse of `.rpt` for the 3 targets + continuity error, using section anchors.
- **Cell 5 — Batch loop:** iterate 100 samples → write `runs/scenario_NNN.inp`, run, parse, collect rows. Progress bar, timing, per-run try/except so one failure flags a row instead of crashing the batch.
- **Cell 6 — Assemble & save:** merge features + targets + QC → `dataset.csv`; sanity plots (target histograms, feature-vs-target scatter); verify near-baseline sample reproduces ~19.93 CFS.
- **Cell 7 — pyswmm demo:** re-open one scenario with `pyswmm`, read peak discharge back to compare against `.rpt` parsing (learning aid).

## 6. Design decisions

- **Run + extract method:** `runswmm.exe` subprocess + `.rpt` text parsing (transparent, no new deps). `pyswmm` used only as a learning demo in Cell 7.
- **Artifact retention:** keep `runs/*.inp` and `runs/*.rpt` (100 runs is small, aids auditability); delete `.out` binaries after parsing to save space.
- **Reproducibility:** fixed LHS seed → identical 100 samples on re-run.
- **Robustness:** batch loop isolates per-run failures.
- **Dependencies:** none new for the core pipeline (scipy/pandas/numpy already installed); `pip install pyswmm` only for Cell 7.

## 7. Success criteria

- 100 scenarios run; `dataset.csv` has 100 rows × (5 features + 3 targets + 2 QC) columns.
- Near-baseline sample reproduces peak discharge ≈ 19.93 CFS (within routing tolerance).
- Continuity errors within acceptable bounds for the large majority of runs; outliers flagged, not silently included.
- Output distributions span a meaningful range (some runs flood, some do not) — confirms the parameter ranges exercise the model.

## 8. Risks (Brief §26)

- SWMM automation/CLI invocation quirks on Windows paths (spaces in `EPA SWMM 5.2.4 (64-bit)`).
- Reliable flood-output extraction from `.rpt` (section headers vary when zero flooding occurs — the "Node Flooding Summary" table is absent when nothing floods; parser must handle its absence → 0 flooded nodes, 0 volume).
- Keeping parameter ranges physically realistic (addressed in §3).
