# Measured-Au 8×2 native campaign: independently checked execution inventory (2026-11-23)

## Exact source
- Base main: `a58407ee694ed38625644fe969ab29d7650194f0`.
- Draft PR #216 head: `be0b6bbbd5a673ff80927eef8cc9bab428df3289`.
- GitHub Actions run: https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/runs/45076667894
- Run conclusion: success; job `measured-au-native-fem`, started 2026-11-22 22:15:18 UTC and finished 22:21:34 UTC.
- Native ZIP artifact ID: `14775991240`; archive size 97,656,281 bytes; GitHub reported SHA256 `b318310ceff177754835110d0fd85270a6402b096fda3c61fbf2b71815cf0d90`; expiry 2027-02-20 22:15:15 UTC.
- Workflow assertion checks `completed=True`, `solver_calls_completed=16`, `raw_eigenmode_rows=32`, existence of 16 E/H NPZ files, and explicitly false PML, physical mode tracking, FWHM, sensitivity, submission gates.

## Evidence boundary
The workflow **passed** and an actual 97.7 MB native artifact was uploaded. These facts establish execution of the workflow's assertions, **not independent inspection of the ZIP's individual files**. The ZIP contents and internal SHA256 registry were not downloaded and independently recomputed in this audit. Avoid claims of independent physical validation, COMSOL agreement, mode identity, converged loss or resonance.

The mesh is an exploratory one-ring finite-box geometry without validated PML. The 8-wavelength samples (0.74–0.88 μm, 0.02 μm spacing) and two RI values (1.35, 1.37) do not support verified FWHM, derivative sensitivity, mode anticrossing, or sensor figures of merit.

## Next mandatory native checks
1. Download ZIP by artifact ID and verify GitHub ZIP digest, member names, duplicate/path-traversal entries, per-file SHA256 against GRID_READINESS.json, and numerical row uniqueness/coverage.
2. Inspect finite complex eigenvalues, E/H shapes, solver failures, passive measured Au epsilon and loss dimensional identity.
3. Freeze literature-grounded geometry and source DOI/table; implement outgoing PML and domain/mesh convergence with independently identified core and SPP branches.
4. Independently cross-check selected eigenpairs against a second full-vector Maxwell solver, then acquire dense fixed-geometry RI spectra with converged resonance and FWHM.
5. Only after physics qualification, register source-specific immutable evidence and run geometry-disjoint inverse benchmarks.

## Publication flags
`physics_qualified=false`; `publication_ready=false`; `comsol_verified=false`. This document is an execution inventory, not a publication result.
