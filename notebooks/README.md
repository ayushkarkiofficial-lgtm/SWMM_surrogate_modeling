# SWMM surrogate modelling

Run the notebooks in order: Phase 2 builds the dataset, Phase 3 trains the
surrogates, and Phases 4–6 analyse uncertainty, applicability, and adaptive sampling.
`Untitled` notebooks are learning drafts.

Install Python dependencies with:

```sh
pip install -r notebooks/requirements.txt
```

Install EPA SWMM 5.2 locally and set `SWMM_EXE` in Phases 2, 5, and 6 to your
`runswmm.exe` path. Open the notebooks from the repository root or `notebooks/`.

The base input is `notebooks/base_model.inp`. The readable SWMM helpers are in
`notebooks/swmm_helpers.py`.

Datasets and analysis results are written to `output/`. Existing figures are in
`output/figures/`. Phase 3 creates `output/models/` for trained models, which are
ignored by Git. Run Phase 3 before Phase 4 or Phase 5, and Phase 5 before Phase 6.
Temporary SWMM scenarios are written to the local, ignored `runs/` folder.
