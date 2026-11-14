# Archived COMSOL spectrum consistency audit

This is a **numerical-record integrity check**, not scientific verification of the COMSOL electromagnetic model. The SoftwareX submission used constructed examples; those cannot be converted into COMSOL evidence by relabeling.

## What the audit verifies

- A manifest claims `backend=comsol` and `evidence_class=comsol_physics`.
- The model and configuration are available and match the SHA-256 hashes recorded at run time.
- `simulation_results.csv` and its `.meta.json` sidecar agree on checksum, source and class.
- Every run has a unique `sample_id` and carries row-level `source=comsol` and `evidence_class=comsol_physics`.
- Returned geometry and exact fixed-geometry RI grid agree with the selected targets.
- Raw wavelength/loss arrays are numeric and support independently re-extracted resonance/FWHM.
- Recomputed sensitivity, FOM, RI linearity and target acceptance match `verification.csv`.
- `accepted_targets` is consistent with the recorded results.

The audit **cannot** independently prove that COMSOL was run, that the `.mph` model is physically correct, that material values/boundaries/mesh are converged, or that a fabricated sensor achieved a predicted result. The archive is controlled by its creator; modified results plus modified manifests may still be internally consistent. The evidence registry separately validates provenance but is not peer review.

## Run (on a machine with a completed COMSOL iteration)

```bash
python scripts/research_completion.py audit-comsol \
  --iteration outputs/real_validation/comsol \
  --json-out outputs/real_validation/comsol/SPECTRUM_CONSISTENCY_AUDIT.json

# Equivalent standalone strict command:
python scripts/audit_comsol_iteration.py \
  --iteration outputs/real_validation/comsol \
  --out outputs/real_validation/comsol/SPECTRUM_CONSISTENCY_AUDIT.json \
  --strict
```

A failed audit exits with code 2 in either strict path. Never claim publication-quality COMSOL results until **both** scientific model validation and archived-data consistency checks pass.

### Review the physical model independently

1. Record the actual COMSOL version, geometry, wavelength definition, material optical constants and loss extraction expression.
2. Review mesh independence, boundary conditions, PML and spectral-grid convergence.
3. Confirm geometry and RI sweep are the exact candidate selected by inverse design.
4. Compare against literature or independent reference solutions, report sensitivity/FOM accuracy, error distributions and numerical limitations.
5. Archive model, configuration, raw spectra, dataset, trained checkpoint and run metadata with stable identifiers.

### Test limitation

`tests/test_physics_audit.py` uses **generated curves and a dummy model fixture** to exercise parsing and integrity rejections. Those tests do not provide physical evidence and must not be cited as a real PCF-SPR validation result.

Related: [Scientific evidence readiness](SCIENTIFIC_EVIDENCE_READINESS.md), [Real validation to stable release](REAL_VALIDATION_TO_STABLE_RELEASE.md).
