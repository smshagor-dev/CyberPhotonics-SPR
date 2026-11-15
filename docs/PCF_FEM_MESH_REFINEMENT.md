# Exploratory numerical mesh-refinement audit — full-vector PCF (not journal-ready)

This Stage-B subtest executes real Femwell full-vector eigenmode solves at three Au-interface mesh resolutions on the same **exploratory** geometry (one hole ring, 50 nm gold/analyte channel) and saves each full complex vector eigenfield, mesh, material/config, provenance hashes, complex effective-index comparison and solver run report.

Start by running the independent solver feasibility control documented in docs/PCF_FEMWELL_FEASIBILITY.md.

## Actual run

Using the optional Python 3.11 FEM tools in the CI workflow:

    python scripts/probe_femwell_mesh_refinement.py --out outputs/open_physics/mesh_audit --gold-mesh-steps-um 0.03,0.025,0.02 --num-modes 2 --strict

The new workflow pcf-mesh-refinement.yml **executes the solver**, archives all raw mode data and fails on missing/failed eigenmode runs. A workflow PASS means only this numerical experiment ran and produced a comparison report, not that Maxwell physics has been fully validated.

Output:

- level_00, level_01, level_02: independent real Femwell mesh/mode material files, raw_eigenfields.npz and FEMWELL_PILOT.json.
- matched_eigenvalue_changes.csv: absolute/relative changes between adjacent mesh levels.
- REFINEMENT_READINESS.json: all runs, hashes, matching algorithm, maximum last-step difference, explicit science rejection flags.

## Critical interpretation

1. **Matching by nearest complex n_eff is not field-overlap mode identity tracking**. Two closely spaced modes can swap, and small changes do not automatically establish convergence to the right mode.
2. Thin Au, polygon approximation, finite remote boundaries without PML and approximate dispersive Au remain major numerical limitations.
3. No fabricated sensor, measured optical constants, reference geometry, RI sensing calibration, physical sensitivity, FOM or leakage loss is verified.
4. Even a decreasing complex n_eff between meshes cannot prove eigenmode convergence, unless an independent field/residual and outgoing boundary comparison also passes.

Next physical acceptance gates are field-overlap branch tracking; finite-domain/PML stability; material source provenance; a literature-defined topology; a full wavelength by RI matrix; spectral grid convergence; independent cross-solver agreement; source-bound ML statistical evaluation.

**Paper status: NOT READY.** Do not insert this experiment's raw values into the sensor performance table as verified physical results.
