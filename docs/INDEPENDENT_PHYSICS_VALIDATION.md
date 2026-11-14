# Independent COMSOL Numerical-Physics Evidence: Stage 2

This stage implements **two independent, complementary audits**. Neither of them
runs COMSOL unless a separately configured real solver campaign has been executed.
They cannot certify that a fabricated sensor exists or prove any electromagnetic
model is correct. They are conservative prereview checks on actual archives.

## 1. Archived closed-loop spectrum consistency

PR #44 implemented a spectral consistency auditor on a branch other than main.
Stage 2 carries that implementation onto main through this branch.

```bash
python scripts/audit_comsol_iteration.py \
  --iteration outputs/real_validation/comsol \
  --out outputs/real_validation/comsol/spectrum_audit.json \
  --strict

# Equivalent existing research CLI:
python scripts/research_completion.py audit-comsol \
  --iteration outputs/real_validation/comsol \
  --json-out outputs/real_validation/comsol/spectrum_audit.json
```

This reconstructs resonance/FWHM from stored spectra and checks selected
geometry and RI sweep, independently recomputed sensitivity/FOM, reported
acceptance, per-row provenance labels, source/config hashes, sidecar content
hashes, and missing/misaligned records.

A consistency pass does **not** prove that COMSOL was actually run. A dishonest
person could produce internally consistent fake data; no software audit can turn
these data into physical evidence.

## 2. Mesh and spectral convergence studies

The second audit requires an archive directory with:

```text
archive/
  refinement_manifest.json
  model.mph                   # actual saved COMSOL model, not a dummy
  config.yaml                 # complete materials/units/study settings
  solver.log                  # solver-produced run log
  refinement.csv              # exported independent refinement table
```

The `refinement_manifest.json` schema is:

```json
{
  "schema_version": 1,
  "backend": "comsol",
  "solver_version": "REAL_COMSOL_VERSION",
  "materials_reference": "Source of material dispersion and constants",
  "boundary_conditions": "Exact exterior and port boundary settings",
  "pml_description": "Exact PML settings and thickness",
  "geometry_definition": "Frozen geometry and parameterization",
  "study_protocol": "How mesh/spectral refinements were run and mode selected",
  "model": "model.mph",
  "model_sha256": "SHA256_OF_ACTUAL_MODEL",
  "configuration": "config.yaml",
  "configuration_sha256": "SHA256_OF_CONFIG",
  "solver_log": "solver.log",
  "solver_log_sha256": "SHA256_OF_SOLVER_LOG",
  "results": "refinement.csv",
  "results_sha256": "SHA256_OF_EXPORTED_CSV"
}
```

All four files must be stored inside the archive directory. Hashes refer to the
exact file bytes. Never put "test" or placeholder model/config values into
the submission archive.

### Required CSV columns

```text
study_kind,case_id,refinement_index,mesh_elements,wavelength_step_nm,
analyte_ri,d_over_lambda,pitch_um,metal_thickness_nm,channel_radius_um,
lambda_res_nm,fwhm_nm,peak_loss_db_per_cm,source
```

The CSV header must be a single line; the line wrapping above is for reading.

- `study_kind`: `mesh` or `spectral`. **Both** are required for each `case_id`.
- `case_id`: identifies a fixed geometry and analyte RI across both studies.
- `refinement_index`: unique integer 1, 2, 3 ... in coarsest-to-finest order.
- `mesh_elements`: increasing positive element count in a mesh study; constant
  during a spectral study.
- `wavelength_step_nm`: constant in a mesh study; decreasing in a spectral study.
- `lambda_res_nm`, `fwhm_nm`, and `peak_loss_db_per_cm`: from each actual run.
- `source`: literal `comsol`, and *must* match genuinely solver-produced data.

For both studies, a minimum of **three** refinement levels per case is required.
The last two levels must satisfy the **preregistered tolerances** and the final
refinement change must not diverge relative to the previous change.

Default thresholds: resonance absolute change <= **0.1 nm**, relative FWHM
change <= **1%**, relative peak-loss change <= **2%**. These are conservative
engineering starting points, not universal scientifically sufficient limits.
Set study-specific tolerances before viewing outcomes and justify them in the
paper according to measurement resolution and expected effect size.

```bash
python scripts/audit_comsol_convergence.py \
  --archive /path/to/real/COMSOL/archive \
  --json-out outputs/real_validation/mesh_spectral_audit.json \
  --resonance-tol-nm 0.1 \
  --fwhm-rel-tol 0.01 \
  --loss-rel-tol 0.02 \
  --strict
```

`convergence_consistent=true` means recorded refinement trends and hashes
meet these requirements, **not** independent electromagnetic correctness.
`scientific_correctness_verified` and `source_authenticity_verified` intentionally
remain `false`.

## Independent scientific review requirements

1. Verify the actual `.mph` COMSOL study file opens, solves and uses the
   documented Maxwell formulation and eigenmode/propagation selection.
2. Inspect dielectric and metal-dispersion data, complex refractive indices,
   wavelength/loss conversions, parameterization and fabrication constraints.
3. Review PML/exterior boundaries and mesh localization near the metal
   and fiber/analyte interfaces; record mesh quality, DOF and solver tolerances.
4. For representative fixed-geometry RI values, run mesh *and* spectral-grid
   convergence studies on the real solver; keep raw logs and all exported spectra.
5. Compare resonance, sensitivity, FWHM, FOM and mode profiles to relevant,
   independent published methods or reference solutions.
6. Freeze hashes, versions, seeds, any failures, selected/rejected designs,
   and supplementary tables. Do not hide failed physics runs.
7. Ensure the target journal's device/optoelectronics scope supports the
   paper's actual photonics contribution. Physical hardware is only required
   for claims about manufactured sensors or exact-device performance.

## Known blocker as of initial Stage 2 implementation

The checked-in repository contains **no validated .mph model, no real archived
COMSOL fixed-geometry RI sweeps, and no recorded refinement campaign**.
Those external physics artifacts must be provided or genuinely generated before
physics evidence can pass a scientific review. CI uses simulated **test fixtures
only** for tampering regressions; those fixtures are never publishable evidence.
See [issue #42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42)
and [scientific readiness](SCIENTIFIC_EVIDENCE_READINESS.md).
