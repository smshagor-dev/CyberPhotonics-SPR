# Executed independent planar TM finite-volume numerical check

Local execution on 2026-11-17: `OPENBLAS_NUM_THREADS=1 python -m pytest -q tests/test_planar_tm_fd_eigenmode.py` gave **9 passed**; `OPENBLAS_NUM_THREADS=1 python -m research.planar_tm_fd_eigenmode --output numerical_results.json` completed with all residuals below 1e-8.

Inputs: 0.8 µm wavelength, RI 1.36, illustrative constant metal ε=-24+1.5i, ±3.2 µm domain, Dirichlet outer walls, six shift-invert eigenvalue candidates per grid. The exact solver command regenerates machine-readable native numerical results locally. These values are **not** PCF FEM, not a PML study, not independently sourced Au optical constants, and not qualified sensor evidence.

| Cell width (µm) | Numeric Re β (µm⁻¹) | Numeric Im β (µm⁻¹) | Relative error vs planar analytic β | Relative matrix residual |
|---:|---:|---:|---:|---:|
| 0.08 | 10.804985772122041 | 0.013464767633097504 | 0.028058274463502977 | 5.94e-16 |
| 0.04 | 10.948226611804053 | 0.023796003171813768 | 0.015145519746216708 | 1.87e-15 |
| 0.02 | 11.057214969138164 | 0.02795488366329112 | 0.005335105130096712 | 5.20e-15 |
| 0.01 | 11.099997924634067 | 0.02872028788837686 | 0.0014859416786688739 | 2.78e-14 |

Reference planar analytic β=11.116515649515604+0.02888535935473879i µm⁻¹. Numerical agreement improves with refinement but the finite domain, interface discretization and material approximation have **not** been independently qualified. The initial test run failed because a ±3.0 µm domain generated 75 cells at 0.08 µm and violated the interface alignment constraint; the domain was corrected to ±3.2 µm and tests were rerun successfully.

**Fail-closed flags:** `physics_qualified=false`, `publication_ready=false`, `comsol_verified=false`. No physical resonance, FWHM, sensitivity, PCF mode identity, or COMSOL validation is claimed. A true full-vector PCF Maxwell FEM experiment with literature-supported geometry, measured dispersive Au, PML and independent solver convergence remains required.
