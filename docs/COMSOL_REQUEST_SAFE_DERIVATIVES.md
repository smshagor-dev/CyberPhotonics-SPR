# Stage 13 — Request-safe COMSOL sensitivity and FOM diagnostics

The genuine COMSOL backend previously attempted `grouped_finite_difference_sensitivity` after **every** explicit geometry batch. A batch may contain samples from different inverse-design targets, incomplete RI sweeps, repeated candidates or only one RI point. Crucially, two targets with the same geometry and RI could cause an exception **after expensive COMSOL solves had finished**, discarding all completed spectra.

Stage 13 separates **raw FEM solve results** from finite-difference analysis.

## Correct production behavior

- `run_comsol_geometries(model, config, geometries)` returns **only raw solver observations** for the exact requested geometry/RI samples, with stable sample IDs, status, resonance/FWHM and source spectra. It does **not** infer `sensitivity_nm_per_riu` or `fom_per_riu` by pooling unrelated target requests. This is used by Stage 7–12, the remote API and arbitrary active-learning/campaign batches.
- `run_comsol_sweep(model, config)` processes the explicit configured Cartesian RI grid. Finite differences are computed **only** when every solve for a fixed geometry succeeded and its sampled RI coordinates are unique and sufficient. These values are **unqualified grid diagnostics**, not independent scientific evidence. The output includes `sensitivity_diagnostic_status`.
- If a configured grid repeats an RI point, successful raw results and their wavelength/loss spectra are preserved, but its entire geometry group receives `duplicate_ri_points_no_derivative` with NaN sensitivity and FOM.
- If any RI point fails, successful rows remain preserved and the full geometry group receives `incomplete_grid_solver_failure`, with no sensitivity/FOM. The original failed row remains `solver_failed`.
- A single RI point yields `insufficient_unique_ri_points` and no derivative.
- Successful complete unique-RI groups receive `grid_diagnostic_only_unqualified`, without any claim of COMSOL authenticity or numerical optical convergence.

**Do not substitute** these grid diagnostics for publication claims. The physical validation pathway continues to require complete **request-matched** fixed-geometry RI sweeps, authentic native solver logs, mesh/spectral refinement, independent model review and qualified external checks.

## Reproducible software regression

```bash
pytest -q -W error tests/test_comsol_request_derivatives.py tests/test_comsol_sweep.py tests/test_comsol_campaign.py tests/test_resumable_campaign.py
```

The tests inject a deliberately **fake** MPh model and synthetic Lorentzian spectra, and cover duplicate samples from different requests, explicit duplicate RI grids, mid-grid solver failure, insufficient data and valid complete grids. These are software control-plane tests only; they are not evidence of Maxwell-model correctness, a real COMSOL license or optical sensor performance.

Scientific blocker: [issue #42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42).
