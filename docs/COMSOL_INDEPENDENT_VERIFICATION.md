# Stage 5 — Independent COMSOL inverse-design verification

Stage 4 (#48) exported **immutable individual solver requests**. This stage
imports independently archived fixed-geometry RI spectra, binds every result
to a frozen request, recomputes optical metrics and reports an honest,
failure-accounted comparison between inverse-design methods.

**No COMSOL solver is run by the Stage 5 importer.** The user must provide an
authentic saved model, solver configuration/logs and actual exported spectra.
Neither SHA-256 nor consistent synthetic data can prove a COMSOL run occurred.

## Inputs and evidence requirements

1. Stage 4 benchmark directory with `INVERSE_BENCHMARK_MANIFEST.json`,
   `selected_designs.csv`, `PHYSICS_VERIFICATION_REQUESTS.csv`. Never edit
   them after results are generated. The benchmark hashes must match.
2. A **real** `refinement` archive passing the Stage 2
   [mesh and spectral convergence check](INDEPENDENT_PHYSICS_VALIDATION.md).
   `refinement_manifest.json` must include the model SHA-256 of the same
   physical model used for the verification sweeps.
3. A separate solver archive with a saved authentic `model.mph`, export
   configuration, COMSOL execution log, sweep table and a frozen manifest:

```text
solver_archive/
  verification_manifest.json
  model.mph
  config.yaml
  solver.log
  sweeps.csv
```

### Verification manifest

```json
{
  "schema_version": 1,
  "backend": "comsol",
  "evidence_class": "comsol_physics",
  "solver_version": "Actual installed COMSOL release",
  "benchmark_manifest_sha256": "SHA256_OF_STAGE4_MANIFEST",
  "requests_sha256": "SHA256_OF_STAGE4_REQUESTS_CSV",
  "refinement_manifest_sha256": "SHA256_OF_REFINEMENT_MANIFEST",
  "ri_points": 5,
  "ri_span": 0.04,
  "normalization_scales": {
    "sensitivity_nm_per_riu": 100.0,
    "fom_per_riu": 10.0,
    "lambda_res_nm": 10.0
  },
  "model": "model.mph",
  "model_sha256": "SHA256_OF_REAL_MODEL",
  "config": "config.yaml",
  "config_sha256": "SHA256_OF_REAL_CONFIG",
  "solver_log": "solver.log",
  "solver_log_sha256": "SHA256_OF_REAL_SOLVER_LOG",
  "sweeps": "sweeps.csv",
  "sweeps_sha256": "SHA256_OF_EXPORTED_SWEEPS"
}
```

**Important:** Normalization values shown above are illustrative and **not**
scientifically justified thresholds. Pre-register actual units/scales from the
training reference distribution or defensible engineering tolerances **before**
accessing solver outcomes. Keep the external frozen manifest and raw source
archive alongside the manuscript.

Archive files must be inside the archive root; path escapes/symlinks are
rejected. Every file requires a matching SHA-256. The model hash must match
the archived refinement model hash.

### Required `sweeps.csv` columns

```text
request_id,analyte_ri,pitch_um,d_over_lambda,metal_thickness_nm,
channel_radius_um,status,source,evidence_class,wavelength_nm,
loss_db_per_cm,lambda_res_nm,fwhm_nm
```

These are **comma-separated CSV headers on one line** (wrapped for reading).

- `request_id`: exact ID from the frozen Stage 4 request table.
- Each request needs a complete, centered, odd-length RI grid matching the
  manifest (`ri_points` and `ri_span`). **Geometry must not vary** across RI.
- `status` must be `ok` or `failed: reason`; failed solver rows are retained,
  not deleted. Successful rows need at least 16 strictly increasing wavelengths
  and equally sized loss arrays in the two quoted CSV cells. Use comma-separated
  numeric arrays *inside the quoted CSV field*. Store full original COMSOL
  export files and mode selections outside this derived table.
- `source=comsol`, `evidence_class=comsol_physics` must be declared for
  every record, but declaration is never treated as proof of authenticity.
- `lambda_res_nm` and `fwhm_nm` are recomputed from the archived spectra;
  mismatch above a preregistered tolerance blocks the audit.

The importer recomputes a global RI-to-resonance slope (nm/RIU), absolute
slope divided by mean FWHM (FOM), exact-target-RI resonance, linearity R²,
and absolute/normalized target errors, all per original request.

### Run

```bash
python scripts/verify_inverse_comsol.py \
  --benchmark outputs/research/stage4 \
  --solver-archive /path/to/real/solver_archive \
  --refinement-archive /path/to/real/refinement_archive \
  --out outputs/research/stage5 \
  --spectrum-tolerance-nm 0.05 \
  --bootstrap-resamples 2000 \
  --strict
```

Without authentic solver archives, strict mode returns **exit 2** with a JSON
blocker report; this is expected, not a software failure.

## Output and interpretation

- `PHYSICS_VERIFICATION_AUDIT.json`: all provenance checks, total requests,
  completed and failed requests, numerical consistency, explicit blockers,
  method completion rates, and pending scientific authenticity review.
- `request_outcomes.csv`: one record per selected design, including
  per-target physical numerical errors **only when the entire RI sweep passes
  consistency checks**.
- `method_completion_summary.csv`: all requested designs remain in the
  denominator. Completed-only error means are labelled vulnerable to
  incomplete-sweep bias.
- `paired_comsol_error_differences.csv`: matched seed and target-geometry
  contrasts vs tandem, with resampling by independent held-out **geometry**
  rather than correlated RI rows. Failed or incomplete pairs are counted.
- Every generated CSV is hashed in the JSON audit.

A recorded solver failure may leave `numerical_archive_consistent=true`
(i.e. no forged or misaligned records) but it forces
`all_requested_sweeps_successful=false`, strict exit 2 and no complete
method-level acceptance. Missing/extra IDs, changed geometry, unexpected
RI grids, tampered model/log/exports or inconsistent spectra fail closed.

Even if every numerical consistency check passes, the report
**always** preserves `source_authenticity_verified=false`,
`physical_correctness_verified=false` and
`scientific_submission_ready=false`. A reviewer must inspect the actual
COMSOL model, Maxwell/eigenmode formulation, materials/dispersion, PML,
mode selection, mesh convergence and source logs; then document an
independent physics sign-off. A green CI or a local report is not that sign-off.

## Further publication blockers

- All Stage 4 selection used a **tandem-trained surrogate**; a fresh untouched
  external geometry cohort and multi-training-seed validation are still needed.
- Equal surrogate query budgets are not equal total runtime, compute or solver
  budgets; measure actual solver calls and end-to-end cost.
- Compare method failures and acceptance in addition to errors. No
  completed-case-only superiority claims if failures are nonzero.
- Record calibration/OOD and method ablations on a truly independent
  dataset, full numerical traceability and reviewer-qualified optics.
- Physical sensor fabrication/measurement is necessary **only when claims
  are made about real manufactured devices**, not for a numerical-only paper.

All regression tests create **fake** `.mph` bytes and synthetic Lorentzian
spectra solely to challenge the audit. Those files must never appear in a
journal submission or be described as real COMSOL evidence.

Research blocker: [#42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42).
