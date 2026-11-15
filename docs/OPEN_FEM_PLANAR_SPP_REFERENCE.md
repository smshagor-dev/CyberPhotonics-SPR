# Analytical planar SPP Maxwell reference for measured-Au PCF feasibility runs

Status: source-traceable optical reference, **not** an independent full-vector FEM solver and **not** PCF-SPR validation.

Under the exp(-i omega t) time convention, the TM surface-plasmon effective index for two ideal semi-infinite homogeneous media meeting at a flat interface is:

    n_eff,SPP = sqrt(epsilon_Au * epsilon_analyte / (epsilon_Au + epsilon_analyte))
    epsilon_analyte = n_analyte^2

Source for dispersive Au: P. B. Johnson and R. W. Christy, Optical Constants of the Noble Metals, Phys. Rev. B 6, 4370 (1972), DOI https://doi.org/10.1103/PhysRevB.6.4370. The code uses the SHA256-traced, measured-reference optical table in this repository, with no extrapolation. This is **not** the optical dispersion of a specifically fabricated 50 nm coating. It applies only under the stipulated planar half-space approximation; the actual PCF geometry has a curved finite gold shell.

## Actual numerical check against immutable native archive

Reference: successful GitHub Actions run 43685290196, artifact 14149464285. Nine authentic measured-Au Femwell grid solves, RI 1.35/1.36/1.37 and wavelengths 0.75/0.80/0.85 um, each storing two raw eigenvalues. The archived finite-box mesh contains 48,940 elements, with no PML.

Independent local evaluation of this analytical expression against Au permittivity values recorded by that real native archive gives these sample values:

| RI | wavelength (um) | planar Re(n_eff,SPP) | planar Im(n_eff,SPP) | minimum |Re(planar) - Re(raw FEM)| |
|---|---:|---:|---:|---:|
| 1.35 | 0.75 | 1.41524878 | 0.00433699 | 0.03788097 |
| 1.36 | 0.80 | 1.41524750 | 0.00367443 | 0.03681360 |
| 1.37 | 0.85 | 1.41767571 | 0.00310000 | 0.03340821 |
| 1.37 | 0.75 | 1.43834152 | 0.00455278 | 0.01478824 |

Across the nine sampled pairs, the minimum real-index mismatch with *unidentified* raw FEM eigenpairs is between **0.01479 and 0.05554**. There is no documented planar-interface phase match among these sampled raw eigenpairs, but this neither excludes a PCF-SPR resonance elsewhere nor identifies the actual PCF SPP mode. Different boundary conditions, mode families, finite gold thickness and curved geometry preclude a direct solver equivalence assertion.

The new source-specific pipeline downloads the actual immutable native artifact, checks its SHA256/content provenance, recomputes all nine planar reference points, and archives PLANAR_SPP_REFERENCE.json. The equation tests use invented arrays only as non-evidentiary software fixtures.

**Never** classify the planar equation as an independently validated PCF full-vector solver, loss/resonance/FWHM/sensitivity measurement, or COMSOL output. Required next: exact published PCF topology, PML/radiation boundary, mode-localization/core-SPP tracking, converged spectral grid and a genuinely independent second Maxwell solver. Scientific submission readiness remains **FALSE**.
