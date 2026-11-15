# Archived native Maxwell material-source diagnostic (NOT sensor loss)

Date checked: 2026-11-15. Numerical evidence class: **unqualified Femwell eigenmode feasibility**.

## Independent raw sources

- Fitted Au, GitHub Actions run 43675419748, artifact 14146447038, ZIP SHA256 5576492fb4eeab99804bd0472bdc8a87ba7f466e92dfd0f0a65463976ad4df27.
- Johnson–Christy Au, GitHub Actions run 43685290196, artifact 14149464285, ZIP SHA256 30b28d83cef11faca79dae829c3a931edc73da2bcc9e69cd12af50b66f3cf4eb.
- Both full-vector Femwell 0.1.12 archives have 9 completed solves, 18 eigenpairs and 9 nonempty native complex E/H NPZ files; all 14 manifest-listed SHA256 artifacts checked on each.
- Both contain the **same Gmsh mesh SHA256**: ef21c1972285093f8fbac9cfa817ef1ab4fda07abbebb3ef472f6cd8665717ec (48,940 elements).
- Their fixed geometry and wavelength/RI coordinate arrays are identical; only the optical Au material model changes. The JC input manifest identifies 49 measured-reference knots and the upstream DOI 10.1103/PhysRevB.6.4370.
- No COMSOL data, no PML and no fabricated gold film were involved.

## Actual calculation, permutation-invariant over the two computed raw eigenmodes

For each frozen RI × wavelength pair, take the arithmetic mean of **Im(n_eff)** for the two raw eigenpairs in each source archive. This summary is **permutation invariant** and does not assume the native mode IDs identify the same physical core/SPP mode. Compute relative change as 100 × (mean_JC / mean_fitted − 1). This is a numerical eigenvalue/material-source diagnostic, **not confinement loss, absorption spectrum, or physical sensitivity**.

| Analyte RI | Wavelength (um) | Mean Im(neff), fitted Au | Mean Im(neff), JC Au | Relative difference |
|---|---:|---:|---:|---:|
| 1.35 | 0.75 | 8.05650648e-8 | 4.30034414e-8 | -46.62% |
| 1.35 | 0.80 | 9.84050296e-8 | 6.93653306e-8 | -29.51% |
| 1.35 | 0.85 | 1.42736646e-7 | 1.51597229e-7 | +6.21% |
| 1.36 | 0.75 | 8.20457342e-8 | 4.34672514e-8 | -47.02% |
| 1.36 | 0.80 | 9.94913201e-8 | 6.97770115e-8 | -29.87% |
| 1.36 | 0.85 | 1.43592625e-7 | 1.51508335e-7 | +5.51% |
| 1.37 | 0.75 | 8.37786882e-8 | 4.40035478e-8 | -47.48% |
| 1.37 | 0.80 | 1.00736771e-7 | 7.02458954e-8 | -30.27% |
| 1.37 | 0.85 | 1.44565379e-7 | 1.51453653e-7 | +4.76% |

The material assumption changes mean Im(neff) by **−47.48% to +6.21%** in these finite-box, unselected modal solutions. This does **not** imply the same change in an experimentally observable SPR attenuation peak. It demonstrates why a fitted optical Au model must not silently substitute for a source-traceable measured-reference model.

## Reproduction on the *real* GitHub Actions artifacts

Download each run's exact artifact from GitHub and unzip to fitted/ and measured/ respectively. The raw files are raw_eigenmode_grid.csv, GRID_READINESS.json, experiment_protocol.json, pcf_frozen_exploratory_grid_mesh.msh and nine raw_vector_eh_ri*_wl*.npz files. Recalculate mean Im(neff) using the two per-grid CSV rows, after verifying every SHA256 in GRID_READINESS.json and the identical mesh hashes.

For each RI/wavelength combination, calculate:

    mean_imag = sum(float(row["n_eff_imag"]) for row in raw_modes_at_gridpoint) / 2
    percent_change = 100 * (mean_imag_measured_Au / mean_imag_fitted_Au - 1)

Do not use this 3-point spectral pilot to fit peaks, FWHM, wavelength sensitivity or FOM; three samples cannot resolve a proper resonance. Do not train or evaluate a geometry-generalizing inverse model on these 9 samples (only one exploratory geometry). These numbers are derived from actual FEM artifacts, but the **original physical publication-readiness flags remain FALSE**, because external boundary, mesh/spectral convergence, exact literature topology, physical mode identification and independent solver checks are absent.
