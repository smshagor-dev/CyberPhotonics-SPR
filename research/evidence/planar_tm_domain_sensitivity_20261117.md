# Executed planar TM domain-truncation sensitivity — 2026-11-17

**Scope:** independently executed local Python/SciPy sparse eigenvalue calculation using the same cell-centered TM finite-volume discretization as `research/planar_tm_fd_eigenmode.py`. This is **not** a PCF FEM, PML, COMSOL, measured-Au, or journal-quality sensor calculation.

## Reproduction

```bash
python -m research.planar_tm_domain_sensitivity --output reports/planar_tm_domain_sensitivity.json
pytest -q tests/test_planar_tm_domain_sensitivity.py
```

Parameters: 0.8 µm wavelength; RI=1.36; illustrative metal epsilon=-24+1.5i; dx=0.02 µm; aligned planar interface at x=0; homogeneous Dirichlet walls; half-domain sizes below. Numerical beta is in µm⁻¹. Local experiment used NumPy and SciPy (SciPy 1.17.0).

| Half-domain (µm) | Relative beta error vs analytic planar SPP | Relative matrix residual | Re(beta) (µm⁻¹) | Im(beta) (µm⁻¹) |
|---:|---:|---:|---:|---:|
| 1.6 | 0.005348757418930869 | 8.166978236636401e-15 | 11.057062532700613 | 0.027997480509231915 |
| 2.4 | 0.005335235088071316 | 6.103936633542861e-15 | 11.057213513440873 | 0.027955574450912457 |
| 3.2 | 0.005335105130096712 | 5.800305147142276e-15 | 11.057214969138164 | 0.027954883663291186 |
| 4.0 | 0.005335103925151409 | 7.199916333925793e-15 | 11.057214982685093 | 0.027954874075465125 |

Last-pair relative complex-beta drift: **1.5009659847986674e-9** (3.2→4.0 µm). Domain truncation effect is small at fixed dx, but **discretization error remains ~0.5335%**. Thus domain stability does not establish continuum convergence. The computation was executed locally; the committed regression tests and GitHub CI remain to be verified independently.

**Hard blockers:** literature-frozen PCF topology, independently sourced wavelength-dependent Au, vectorial PCF Maxwell solve, PML convergence, core/SPP field identity, RI-resolved spectra, geometry-disjoint inverse benchmark. All publication flags are **FALSE**. No COMSOL claim.
