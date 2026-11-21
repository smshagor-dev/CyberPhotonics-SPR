# Independent native-eigenpair separation control (2026-11-21)

## Source and scope
Source: genuine exploratory Femwell PR #64 GitHub Actions ZIP artifact `14146447038`, SHA256 `5576492fb4eeab99804bd0472bdc8a87ba7f466e92dfd0f0a65463976ad4df27`. This is **not** a COMSOL archive. The experiment is the same-mesh 3×3 analyte-RI/wavelength grid, two native-index eigenpairs at each point.

A separate local Python audit opened the ZIP, checked unique and safe member names, verified every registered SHA256, checked grid coordinates and completeness, checked finite passive complex effective indices, and computed the absolute complex difference between the two native-index effective indices at each point. It executed successfully on the actual 53 MB ZIP; three local regression tests passed. The audit script and machine-readable output are retained as local artifacts pending repository integration; this note is **not** a substitute for an immutable, CI-produced native audit archive.

## Observed native-index separation

Absolute complex effective-index separation `|n_eff,0 - n_eff,1|`, dimensionless:

| Analyte RI | 0.75 µm | 0.80 µm | 0.85 µm |
|---|---:|---:|---:|
| 1.35 | 1.5571180e-6 | 1.9101067e-6 | 2.3615145e-6 |
| 1.36 | 1.5550979e-6 | 1.9081382e-6 | 2.3591456e-6 |
| 1.37 | 1.5528065e-6 | 1.9059579e-6 | 2.3565815e-6 |

These small gaps make naive nearest-eigenvalue mode association fragile. This does **not** demonstrate a core/SPP avoided crossing: physical identity, normalization, full-vector field-overlap assignment and branch continuity are not established. Mode-index continuity cannot be assumed.

## Qualification boundary
- The topology is exploratory, not literature-frozen.
- Au is fitted rather than authenticated from an independently sourced tabulation.
- No PML, mesh/domain/spectral convergence or independent full-vector solver comparison.
- Three wavelength samples are insufficient for resolved resonance and FWHM.
- Raw attenuation from Im(n_eff) is not qualified confinement loss.
- No geometry-disjoint solver-backed training data or publication performance claims follow from these diagnostics.

`physics_qualified=false`, `publication_ready=false`, `comsol_verified=false`.

## Reproducibility gap
Promote the separately executed audit source, adversarial tests and SHA256-registered report into a future PR with exact-head CI and genuine-archive compatibility before treating this diagnostic as a repository-integrated result. Never promote the diagnostic into sensor performance evidence.
