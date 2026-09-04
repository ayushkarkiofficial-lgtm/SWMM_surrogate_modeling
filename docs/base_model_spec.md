# Base SWMM Model — Specification Sheet

First learning model for the SWMM surrogate-modeling project. Built by hand in the
EPA SWMM 5.2 GUI, imitating a trimmed version of the official EPA
`Site_Drainage_Model.inp` (Applications Manual site-drainage example).

Reference model:
`C:\Users\Acer\Documents\EPA SWMM Projects\Samples\Site_Drainage_Model.inp`

## Network layout

```
   S1 ┐
   S2 ┼──► J1 ──C1──► J2 ──C2──► Out1   (free outfall)
   S3 ┘
         ▲ RainGage feeds all three subcatchments
```

## Project options

| Option        | Value          |
|---------------|----------------|
| Flow units    | CFS (US units) |
| Infiltration  | Horton         |
| Flow routing  | Dynamic Wave   |
| Start         | 01/01/1998 00:00 |
| End           | 01/01/1998 06:00 (6-hour simulation) |
| Routing step  | 15 s           |
| Report step   | 1 min          |

## Rain gage

- **Name:** RainGage
- **Format:** Intensity · **Interval:** 5 min · **SCF:** 1.0
- **Data source:** Time Series named `2-yr`

### Time series `2-yr` (rainfall intensity, in/hr) — 2-year 2-hour design storm

| Time  | Value | Time  | Value | Time  | Value | Time  | Value |
|-------|-------|-------|-------|-------|-------|-------|-------|
| 0:00  | 0.29  | 0:30  | 2.85  | 1:00  | 0.20  | 1:30  | 0.15  |
| 0:05  | 0.33  | 0:35  | 1.18  | 1:05  | 0.19  | 1:35  | 0.15  |
| 0:10  | 0.38  | 0:40  | 0.71  | 1:10  | 0.18  | 1:40  | 0.14  |
| 0:15  | 0.64  | 0:45  | 0.42  | 1:15  | 0.17  | 1:45  | 0.14  |
| 0:20  | 0.81  | 0:50  | 0.35  | 1:20  | 0.17  | 1:50  | 0.13  |
| 0:25  | 1.57  | 0:55  | 0.30  | 1:25  | 0.16  | 1:55  | 0.13  |

Peak intensity 2.85 in/hr at 0:30.

## Subcatchments (all drain to J1)

| Name | Rain Gage | Outlet | Area (ac) | %Imperv | Width (ft) | %Slope |
|------|-----------|--------|-----------|---------|------------|--------|
| S1   | RainGage  | J1     | 4.55      | 56.8    | 1587       | 2      |
| S2   | RainGage  | J1     | 4.74      | 63.0    | 1653       | 2      |
| S3   | RainGage  | J1     | 3.74      | 39.5    | 1456       | 3.1    |

Shared sub-area properties (all three):

| N-Imperv | N-Perv | S-Imperv | S-Perv | %Zero | Route To |
|----------|--------|----------|--------|-------|----------|
| 0.015    | 0.24   | 0.06     | 0.30   | 25    | OUTLET   |

Shared Horton infiltration (all three):

| MaxRate | MinRate | Decay | DryTime | MaxInfil |
|---------|---------|-------|---------|----------|
| 4.5     | 0.2     | 6.5   | 7       | 0        |

## Junctions

| Name | Invert elev (ft) | Max depth (ft) |
|------|------------------|----------------|
| J1   | 100              | 4              |
| J2   | 98               | 4              |

## Conduits (circular concrete pipes, Manning n = 0.016, offsets 0)

| Name | From | To   | Length (ft) | Diameter (ft) |
|------|------|------|-------------|---------------|
| C1   | J1   | J2   | 185         | 2.0           |
| C2   | J2   | Out1 | 89          | 2.0           |

## Outfall

| Name | Invert elev (ft) | Type |
|------|------------------|------|
| Out1 | 96               | FREE |

## Deliberate simplifications vs. the reference

- Clean round elevations (100 / 98 / 96 ft) for an obvious downhill slope.
- Circular 2.0 ft pipes (reference uses swales/culverts) so the first run completes
  cleanly. Later we can shrink diameters to intentionally induce flooding for study.
- Dropped: TSS pollutant, land uses, coverages, buildup/washoff, subcatchments
  S4–S7, junctions J3–J11.

## Save location

Save the SWMM project as: `D:\Claude\surrogate_modeling\base_model.inp`
