# SWMM Surrogate Modeling Research Project Brief

## Working Title

**Surrogate-Assisted Monte Carlo Uncertainty Analysis of Urban Stormwater Drainage Systems Using SWMM**

## 1. Project Summary

This mini project investigates whether machine-learning surrogate models can approximate important EPA SWMM outputs accurately enough to support large-scale uncertainty analysis.

In simple terms:

> Instead of running a slow drainage model 10,000 to 100,000 times, we run SWMM a smaller number of times, train a machine-learning model to imitate SWMM behavior, and then use the fast surrogate model for probabilistic flood-risk analysis.

The central workflow is:

```text
Uncertain drainage parameters
        ↓
SWMM simulations
        ↓
Dataset of inputs and outputs
        ↓
Machine-learning surrogate model
        ↓
Large Monte Carlo analysis
        ↓
Flood probability and sensitivity results
```

## 2. Background

Urban drainage simulations are commonly performed using numerical models such as EPA SWMM. SWMM can simulate rainfall-runoff, pipe flow, junction flooding, surcharge, infiltration, and outfall discharge.

However, uncertainty analysis may require thousands or hundreds of thousands of simulations. This can become computationally expensive for large drainage networks.

This research will test whether machine-learning surrogate models can reproduce selected SWMM outputs while greatly reducing computational cost.

## 3. Main Research Question

Can a machine-learning surrogate accurately reproduce important SWMM outputs while significantly reducing the computational cost of Monte Carlo uncertainty analysis?

## 4. Secondary Research Questions

- Which uncertain hydrological and drainage parameters have the greatest influence on urban flooding?
- How many SWMM simulations are required to develop an accurate surrogate?
- Which surrogate model provides the best balance between accuracy and computational efficiency?
- How much computational time can surrogate-assisted Monte Carlo simulation save compared with direct SWMM simulation?
- Does the surrogate remain accurate for extreme flooding cases, not just average cases?

## 5. Beginner Mental Model

Think of SWMM as the teacher.

SWMM understands the drainage system using hydrology and hydraulic equations, but it can be slow if we run it many times.

We ask SWMM many questions:

```text
What happens if rainfall is high?
What happens if imperviousness is low?
What happens if pipes are rough?
What happens if infiltration is weak?
```

Then we collect SWMM's answers.

The machine-learning surrogate is the student. It studies those answers and learns to predict similar results quickly.

Once the student is accurate enough, we use it to answer thousands of new uncertainty questions very quickly.

## 6. Key Concepts

### EPA SWMM

EPA SWMM, or Storm Water Management Model, is a numerical model used for stormwater and drainage-system simulation.

It can simulate:

- rainfall
- runoff
- infiltration
- pipe flow
- junction flooding
- surcharge
- outfall discharge
- storage and hydraulic structures

### Hydrology

Hydrology describes how rainfall becomes runoff.

Example questions:

- How much rain falls?
- How much water infiltrates into the soil?
- How much water runs over the ground?
- How quickly does runoff reach the drainage system?

### Hydraulics

Hydraulics describes how water moves through pipes, channels, junctions, and outfalls.

Example questions:

- How much water flows through a pipe?
- Does the pipe become full?
- Does water back up at a junction?
- Does flooding occur at a manhole?

### Surrogate Model

A surrogate model is a faster approximation of a more detailed model.

For this project:

```text
SWMM = detailed but slower physics-based model
ML surrogate = faster approximation trained from SWMM results
```

### Monte Carlo Simulation

Monte Carlo simulation means repeatedly sampling uncertain inputs and analyzing the resulting output distribution.

Instead of one fixed answer, Monte Carlo gives probabilistic results such as:

```text
Probability of flooding = 35%
Probability flood volume exceeds 500 m3 = 12%
95th percentile peak discharge = 3.8 m3/s
```

### Latin Hypercube Sampling

Latin Hypercube Sampling, or LHS, is a structured sampling method used to cover the uncertain input space efficiently.

Monte Carlo analysis is the overall uncertainty-analysis method. LHS is one possible sampling strategy used inside that process.

The distinction is:

```text
Monte Carlo analysis = repeated simulation under uncertainty
LHS = efficient method for selecting uncertain input combinations
```

In this project:

```text
Use LHS for the smaller SWMM training dataset.
Use larger Monte Carlo sampling after the surrogate is trained.
```

## 7. Numerical Model

EPA SWMM will be used as the numerical stormwater model.

The first model should be small and understandable. A representative synthetic urban drainage catchment will be created or recreated from an official example.

The model will contain:

- subcatchments
- rainfall input
- junctions
- conduits or pipes
- outfall
- infiltration
- surface runoff
- possible node flooding or surcharge

## 8. Should We Download Or Build A SWMM Model?

The recommended approach is:

> Find a reliable SWMM example online, study it, then rebuild a simplified version ourselves.

This gives us a working reference model while still teaching the basics of SWMM.

We should not blindly use a complex downloaded real-world model because it may contain too many assumptions, nodes, pipes, rainfall files, controls, and calibration decisions.

We should also not build entirely from nothing at first because that increases the chance of beginner errors.

The best path is:

```text
Step 1: Get an official EPA example SWMM model
Step 2: Open and inspect the .inp file
Step 3: Rebuild a smaller version ourselves
Step 4: Run the copied or rebuilt model
Step 5: Modify parameters manually
Step 6: Automate parameter modification with Python
Step 7: Use it for LHS, surrogate modeling, and Monte Carlo analysis
```

## 9. Candidate SWMM Model Sources

Reliable sources to use as references:

- Official EPA SWMM software and documentation
- EPA SWMM Applications Manual example models
- EPA sample models installed with SWMM
- Open Water Analytics / public SWMM example models
- Public SWMM model libraries only after we understand basic model structure

Good beginner direction:

```text
Start with an official EPA site-drainage or small catchment example.
Use it as a template.
Rebuild a simple model manually so we understand every section.
```

## 10. Proposed First SWMM Model

A simple learning model could look like:

```text
Rainfall
   ↓
Subcatchment S1 ─┐
Subcatchment S2 ─┼→ Junction J1 → Pipe C1 → Junction J2 → Pipe C2 → Outfall
Subcatchment S3 ─┘
```

This is enough to learn the core SWMM logic:

- rainfall falls on land areas
- some rainfall infiltrates
- the rest becomes runoff
- runoff enters junctions
- water moves through pipes
- water exits through an outfall
- flooding occurs if the drainage system cannot carry the flow

## 11. Important SWMM Input File Sections

A SWMM model is stored in a `.inp` text file.

Important sections include:

```text
[TITLE]
[OPTIONS]
[RAINGAGES]
[SUBCATCHMENTS]
[SUBAREAS]
[INFILTRATION]
[JUNCTIONS]
[OUTFALLS]
[CONDUITS]
[XSECTIONS]
[TIMESERIES]
[REPORT]
[COORDINATES]
[POLYGONS]
```

Each section controls a different part of the model.

Examples:

- `[RAINGAGES]` defines rainfall source information.
- `[TIMESERIES]` stores rainfall values over time.
- `[SUBCATCHMENTS]` defines land areas.
- `[SUBAREAS]` defines impervious and pervious surface behavior.
- `[INFILTRATION]` defines soil infiltration behavior.
- `[JUNCTIONS]` defines manholes or connection points.
- `[CONDUITS]` defines pipes.
- `[XSECTIONS]` defines pipe shapes and diameters.
- `[OUTFALLS]` defines where water leaves the system.

## 12. Initial Uncertain Parameters

The first version should use a small number of uncertain inputs.

| Parameter | Physical Meaning | Why It Matters |
| --- | --- | --- |
| Rainfall intensity or rainfall multiplier | Storm uncertainty | Stronger rainfall creates more runoff |
| Impervious percentage | Urbanization and runoff generation | More impervious area means less infiltration and more runoff |
| Surface Manning roughness | Overland-flow resistance | Affects how quickly runoff reaches the drainage system |
| Infiltration parameter | Rainfall loss into soil | More infiltration reduces runoff |
| Pipe Manning roughness | Hydraulic resistance inside pipes | Rougher pipes reduce conveyance capacity |

Possible first parameter set:

```text
X1 = rainfall multiplier
X2 = impervious percentage
X3 = surface Manning roughness
X4 = infiltration conductivity or curve number
X5 = pipe Manning roughness
```

Each parameter will need:

- lower bound
- upper bound
- distribution type
- physical justification

## 13. SWMM Outputs

Potential surrogate prediction targets include:

- peak discharge
- total flooding volume
- maximum node flooding rate
- number of flooded nodes
- maximum surcharge depth
- runoff volume

For the first project version, focus on 2 or 3 outputs:

```text
1. Peak discharge at the outfall
2. Total flooding volume
3. Number of flooded nodes or maximum node flooding
```

## 14. Experimental Design

Latin Hypercube Sampling will initially be used to efficiently sample combinations of uncertain parameters.

Initial SWMM simulation stages:

```text
100 runs: quick automation test
300 runs: first surrogate comparison
500 to 1000 runs: final convergence and validation experiment
```

The required sample size should be determined through model performance and learning curves rather than assumed beforehand.

Learning-curve idea:

```text
Train with 50 samples
Train with 100 samples
Train with 200 samples
Train with 400 samples
Train with 800 samples
Compare accuracy improvement
```

## 15. SWMM Automation

Python will be used to automatically:

1. Generate parameter combinations.
2. Modify SWMM input parameters.
3. Execute SWMM simulations.
4. Extract simulation outputs.
5. Store results in a structured dataset.

Automation flow:

```text
base_model.inp
      ↓
modify parameters
      ↓
scenario_001.inp
      ↓
run SWMM
      ↓
scenario_001 results
      ↓
extract outputs
      ↓
dataset.csv
```

Possible tools:

- Python
- pandas
- NumPy
- scipy or pyDOE-style LHS sampling
- PySWMM
- swmm-toolkit
- SWMM command-line executable

## 16. Simulation Dataset

After running SWMM simulations, we will create a dataset containing input parameters and SWMM outputs.

Example structure:

| rainfall_multiplier | impervious_pct | pipe_roughness | peak_discharge | total_flooding_volume |
| ---: | ---: | ---: | ---: | ---: |
| 0.85 | 45 | 0.013 | 1.2 | 0 |
| 1.10 | 60 | 0.015 | 2.1 | 120 |
| 1.45 | 85 | 0.017 | 3.8 | 950 |

In machine-learning terms:

```text
Features = uncertain SWMM input parameters
Targets = SWMM simulation outputs
```

## 17. Machine-Learning Surrogate Models

The surrogate relationship is:

```text
Hydrological + hydraulic parameters
        ↓
ML surrogate
        ↓
SWMM response
```

Models to compare:

- Random Forest
- XGBoost or another gradient boosting method
- Artificial Neural Network
- Gaussian Process Regression, if computationally practical

Recommended first comparison:

```text
Random Forest vs XGBoost vs Neural Network
```

### Random Forest

Useful as a strong baseline.

Strengths:

- handles nonlinear behavior
- works well with small and medium tabular datasets
- needs limited preprocessing
- gives feature importance

### XGBoost / Gradient Boosting

Likely to perform very well for tabular data.

Strengths:

- high predictive accuracy
- handles nonlinear interactions
- fast prediction after training

### Neural Network

Useful for comparison, but may require more tuning and more data.

### Gaussian Process Regression

Useful for uncertainty-aware surrogate modeling, but may become computationally expensive as the dataset grows.

## 18. Surrogate Validation

A portion of SWMM simulations will remain completely unseen during model training.

Possible split:

```text
80% training
20% testing
```

If we run 500 SWMM simulations:

```text
400 simulations: train surrogate
100 simulations: test surrogate
```

Performance will be evaluated using:

- R2
- RMSE
- MAE
- prediction-versus-SWMM plots
- residual plots
- performance for extreme flooding scenarios
- runtime comparison between SWMM and surrogate

Extreme-case validation is especially important. A model with high average accuracy may still underpredict rare high-flooding events.

## 19. Monte Carlo Simulation

After validation, the best-performing surrogate will be used instead of SWMM for large-scale Monte Carlo simulation.

Example:

```text
10,000 to 100,000+ uncertain drainage scenarios
```

The surrogate can evaluate these much faster than direct SWMM simulation.

Example runtime logic:

```text
SWMM runtime per simulation = 10 seconds
100,000 SWMM simulations = about 11.6 days

Surrogate prediction for 100,000 scenarios = seconds or minutes
```

## 20. Probabilistic Outputs

The Monte Carlo analysis may estimate quantities such as:

- probability of node flooding
- probability of exceeding a specified flooding volume
- probability of drainage-system surcharge
- distribution of peak runoff
- distribution of total flooding volume
- 90th, 95th, and 99th percentile flood outputs

Example output statements:

```text
Probability of flooding = 35%
Probability total flooding volume exceeds 500 m3 = 12%
95th percentile flood volume = 900 m3
```

## 21. Sensitivity Analysis

Sensitivity analysis will determine which uncertain parameters control flooding predictions.

Possible techniques:

- permutation importance
- SHAP
- Sobol sensitivity indices

Recommended first approach:

```text
Use permutation importance first.
Use SHAP for model interpretation.
Use Sobol later if the sampling design supports it.
```

Expected interpretation examples:

```text
High rainfall multiplier strongly increases flood volume.
High imperviousness increases runoff.
Higher infiltration reduces flooding.
Higher pipe roughness increases flooding.
```

## 22. Overall Workflow

```text
Urban Catchment
      ↓
Build or recreate SWMM Model
      ↓
Validate deterministic simulation
      ↓
Define uncertain parameters
      ↓
Latin Hypercube Sampling
      ↓
Automated SWMM simulations
      ↓
Create simulation dataset
      ↓
Train ML surrogate models
      ↓
Independent validation
      ↓
Select best surrogate
      ↓
Large Monte Carlo simulation
      ↓
Uncertainty analysis
      ↓
Sensitivity analysis
      ↓
Probabilistic urban flood assessment
```

## 23. Recommended Mini Project Scope

The first complete version should be:

```text
SWMM model: small synthetic urban drainage network
Uncertain inputs: 5
Training simulations: 300 to 500
Surrogate models: Random Forest and XGBoost
Outputs: peak discharge and total flooding volume
Monte Carlo size: 10,000 to 50,000
Sensitivity: permutation importance and SHAP
```

This is realistic, complete, and suitable for a mini project.

## 24. Expected Project Outputs

Expected files and artifacts:

- base SWMM `.inp` model
- generated scenario `.inp` files
- SWMM report and output files
- simulation dataset CSV
- trained surrogate models
- validation metrics
- prediction-versus-SWMM plots
- Monte Carlo output distributions
- exceedance probability plots
- sensitivity-analysis figures
- final report

Example model comparison table:

| Model | R2 | RMSE | MAE | Runtime |
| --- | ---: | ---: | ---: | --- |
| Random Forest | 0.91 | 80 | 42 | fast |
| XGBoost | 0.95 | 55 | 30 | fast |
| Neural Network | 0.89 | 95 | 50 | fast |

## 25. Future Extension

After establishing the methodology using SWMM, the same framework can be extended to natural river-basin flood modeling.

```text
Rainfall
   ↓
HEC-HMS
   ↓
Runoff hydrograph
   ↓
HEC-RAS 2D
   ↓
Numerical flood simulations
   ↓
ML surrogate
   ↓
Monte Carlo uncertainty analysis
   ↓
Probabilistic flood inundation mapping
```

This creates a broader research direction in computational hydrology, hydroinformatics, surrogate modeling, and uncertainty quantification.

## 26. Main Risks

Important risks to watch:

- SWMM automation may take time to configure correctly.
- Extracting flooding outputs reliably can be tricky.
- Parameter ranges must remain physically realistic.
- Surrogate accuracy for extreme events may be weaker than average accuracy.
- Synthetic catchment results may not generalize directly to real-world drainage systems.
- Using a complex downloaded model too early may slow learning.

## 27. Practical Next Steps

1. Download or locate an official EPA SWMM example model.
2. Study the `.inp` structure section by section.
3. Recreate a small version manually.
4. Run the base model once.
5. Modify one parameter manually and observe output changes.
6. Create Python scripts for parameter sampling and model modification.
7. Automate batches of SWMM runs.
8. Build the first SWMM-generated dataset.
9. Train the first Random Forest surrogate.
10. Validate predictions against unseen SWMM simulations.

