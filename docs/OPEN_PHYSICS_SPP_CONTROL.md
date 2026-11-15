# Stage A: Real planar TM surface-plasmon numerical control (not a PCF-SPR result)

This module performs an **actual** numerical finite-difference frequency-domain (FDFD) eigenmode calculation of a **one-dimensional, semi-infinite gold/dielectric interface** and compares it to the exact planar TM surface-plasmon dispersion relation. Its intended use is checking material sign convention, eigenvalue formulation, modal attenuation units, numerical stability and reproducibility **before** investing in a full-vector PCF-SPR FEM implementation.

It is NOT a COMSOL substitute for a 2D gold-coated photonic-crystal fiber. In particular there is **no** PCF lattice, Au thin film, analyte channel resonance, vectorial two-dimensional Maxwell mode, PML, confinement-loss spectrum, physical sensor sensitivity, independent experimental measurement, or publication-grade dataset.

## Run without COMSOL, hardware, or NGSolve

Python 3.10–3.13. Requires only numpy and scipy, already project dependencies.

    python scripts/probe_open_physics.py --out outputs/open_physics/control --strict

With the default 0.8 μm wavelength, dielectric RI 1.36, and 6 μm half-domain, four mesh steps (80 nm, 40 nm, 20 nm, 10 nm) are evaluated.

**This command executes numerical physics**, not synthetic curves or replayed COMSOL fixtures. The numerical eigenproblem uses finite-volume discretization of

    d_x[(1/epsilon) d_x H_y] + (k0^2 - beta^2/epsilon) H_y = 0

with an interface at x=0 and remote Dirichlet boundaries. Interface coefficients use harmonic averaging of 1/epsilon; ARPACK shift-invert finds eigenpairs closest to the analytical planar reference, and each numerical eigenvector is archived. The analytical benchmark is

    n_eff = sqrt(epsilon_Au * epsilon_d / (epsilon_Au + epsilon_d))

under E~exp(i beta z - i omega t). The signed attenuation is

    loss_db_per_cm = (20/ln(10)) * (2*pi/(lambda_um * 1e-6)) * Im(n_eff) / 100.

Negative Im(n_eff) is rejected, not relabeled by taking absolute value.

The Au values are a repository-consistent **fitted Rakić-style Drude-Lorentz approximation** (see src/sprpcf/simulation/dispersion.py), *not* a freshly imported independently measured dataset. See https://doi.org/10.1364/AO.37.005271. All fields/curve/error data are written by a real numerical solve.

## Files produced by the command

- CONTROL_READINESS.json: exact and numerical complex n_eff, eigen residual, four-grid errors, solver/runtime versions, artifact SHA256 hashes and **false** paper/sensor qualification flags.
- mesh_convergence.csv: raw result for each spatial resolution including complex n_eff, attenuation, residual, error, mode localization.
- planar_mode_field.csv: final numerical mode shape with material side, x coordinate, real/imaginary H_y, magnitude.

These files are created **only when the command actually runs**; the GitHub Actions workflow also uploads them as a CI artifact.

### Acceptance and invalidation

- Minimum 3 strictly decreasing mesh steps.
- Every successive mesh refinement must reduce the relative error against the analytical planar TM solution.
- Finest-grid error <= 0.5%, eigen residual <= 1e-8; passive decaying propagation and localization near the interface.
- Invalid geometry, metal dispersion, sign convention, unresolved mode or increasing error => no PASS.
- No claim of PML convergence or PCF sensor performance is ever inferred.

## Tests

    PYTHONPATH=src pytest -q -W error tests/test_open_physics.py

The test suite covers numerical convergence, loss units and sign, invalid inputs, source flags and absence of fictional PCF evidence.

## Mandatory next stage

**A passing planar SPP control does not clear Issue #42.** Still required for a publishable PCF-SPR numerical paper:

1. Freeze a cited, complete two-dimensional PCF topology and fitted vs measured Au optical constants, source data, units, temperature and wavelength range.
2. Independently implement and verify a true **full-vector 2D Maxwell eigenmode solver** for that cross-section with dispersive materials, mesh and transparent boundaries/PML. Candidate: NGSolve+fibermode, with gold/PML support verified in actual runs; alternatively Femwell after similar checks.
3. Recompute solver-native complex propagation constants, track the core-like branch across wavelength and *fixed-geometry* analyte RI, show identifiable resonance and resolved FWHM, mesh/PML/wavelength convergence.
4. Independently cross-check selected examples (e.g. Meep with physically equivalent kz convention), without equating temporal damping to spatial attenuation without derivation.
5. Expand source-verified training/test geometry sets, retrain and compare inverse methods with independent numerical evaluations and statistical tests, then revise the manuscript.
6. Introduce a **separate** open-numerical-science registry and science gate. Do NOT rename output as COMSOL, do NOT weaken existing COMSOL-origin validation, and do NOT promote these control files to open_fem_physics evidence.

Until these stages are completed and reviewed, **scientific_submission_ready remains false**. The current paper's synthetic 820 nm/RIU demonstration is not a physically verified sensor result.
