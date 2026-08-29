# Flagship Research Dashboard

The dashboard is a Streamlit research-control surface over the existing CyberPhotonics-SPR scientific APIs. It does not replace the CLI pipelines or weaken their evidence rules.

## Install

```powershell
pip install -e ".[dashboard]"
```

For COMSOL verification also install the COMSOL extra and use a licensed local installation:

```powershell
pip install -e ".[dashboard,comsol]"
```

Launch:

```powershell
streamlit run src/sprpcf/dashboard/app.py
```

## Workflow

### 1. Design Studio

Enter the requested:

- wavelength sensitivity,
- FOM,
- resonance wavelength,
- analyte refractive index.

The dashboard calls the same `optimize_target_table()` implementation used by the advanced CLI. It shows:

- Pareto rank,
- calibrated confidence score,
- Mahalanobis OOD score,
- fabrication-projection distance,
- predicted sensing metrics,
- a schematic 3D PCF geometry.

The 3D drawing is a schematic for communication and inspection. It is **not** a COMSOL mesh, fabrication mask, or electromagnetic solution.

### 2. Exact-candidate Physics Gate

The selected Pareto row is passed into `run_closed_loop_iteration()` through a fixed designer callback. This is deliberate: the geometry shown in Design Studio is the same geometry that enters the RI sweep.

Backends:

- `synthetic`: CI/demo/software validation only.
- `comsol`: real COMSOL automation using the configured `.mph` model and sweep YAML.

Sensitivity/FOM acceptance uses an odd target-centered RI sweep. A single operating-point simulation is not treated as independent sensitivity validation.

The dashboard displays:

- acceptance/rejection,
- sensitivity/FOM/resonance errors,
- RI-to-resonance linearity,
- all successful spectra.

### 3. Edge / XAI Evidence

The dashboard can load already-generated evidence files, including:

- scientific-validation `summary.json`,
- XAI attribution CSV,
- hardware `benchmark.json`.

These are hash-bound in the report when present. The dashboard does not invent missing benchmark, sensor, COMSOL, or XAI evidence.

### 4. Report

The report includes:

- requested sensing target,
- selected geometry,
- Pareto/calibration/OOD information,
- physics acceptance result,
- SHA-256 hashes for loaded evidence,
- an explicit evidence-boundary statement.

A synthetic result is always labelled software-flow evidence. Only a verified COMSOL backend may support simulation claims, and experimental claims still require measured sensor data.

## Output layout

Default dashboard outputs:

```text
outputs/dashboard/
  design/
    pareto_candidates.csv
    pareto_selected_designs.csv
    design_calibration.json
  physics/
    dashboard_target.csv
    ...
  dashboard_evidence_report.md
```

Closed-loop artifacts under `physics/` retain the standard closed-loop manifest and accepted-only dataset rules.

## CI contract

The dashboard has two layers:

1. `sprpcf.dashboard.core`: Streamlit-independent transformation, plotting, parsing, hashing, and report functions. These are unit tested.
2. `sprpcf.dashboard.app`: thin Streamlit orchestration over the research APIs.

CI installs the optional dashboard dependencies on Python 3.11, imports the app, and runs dashboard-specific tests. The existing Python 3.10–3.13, Edge/LiteRT, release metadata, and package-build gates remain independent.
