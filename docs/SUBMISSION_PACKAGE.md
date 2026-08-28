# Manuscript Supplementary and Submission Package

CyberPhotonics-SPR can convert an existing reviewer evidence package into a journal-oriented supplementary archive without inventing missing scientific results.

## 1. Purpose

The submission builder organizes already-produced evidence into a form that is easier to inspect during manuscript preparation, revision, peer review, or release archiving.

It generates:

- `README_FIRST.md` — reviewer/editor starting point,
- `SUPPLEMENTARY_INFORMATION.md` — evidence-aware supplementary narrative,
- `MANUSCRIPT_CHECKLIST.md` — pre-submission verification checklist,
- `TABLE_S1_VALIDATION_METRICS.csv/.md` — scalar values extracted from supplied validation summary JSON,
- `TABLE_S2_CLAIMS_TO_EVIDENCE.csv/.md` — reviewer claims matrix,
- `TABLE_S3_ARTIFACT_PROVENANCE.csv` — artifact hashes and evidence classes,
- `FIGURE_INDEX.csv` and numbered files under `figures/`,
- mirrored `reviewer_evidence/`,
- selected citation/reproducibility/release metadata,
- `submission_manifest.json`,
- `submission_checksums.sha256`.

The builder copies figures without resampling. Final journal dimensions, DPI, font size, file format, and color-mode requirements must still be checked against the chosen venue.

## 2. Build a submission package

First generate the reviewer package from completed evidence. Then run:

```powershell
python scripts/build_submission_package.py `
  --reviewer-package outputs/reviewer_package `
  --validation-dir outputs/validation `
  --ablation-dir outputs/ablation `
  --design-dir outputs/multiobjective `
  --closed-loop-dir outputs/closed_loop/iteration_001 `
  --hardware-dir outputs/experimental_session `
  --journal "Target journal name" `
  --out outputs/submission_package
```

An optional manuscript can be included with `--manuscript path/to/manuscript.pdf` (also supports `.docx`, `.tex`, and `.md`). The file is copied and SHA-256 bound; the builder does not rewrite it.

## 3. Evidence interpretation

The submission package inherits evidence classes from the reviewer package.

- `software_only` supports software/methodology execution claims only.
- `surrogate_model` supports learned-model/design reporting but not independent physics proof.
- `comsol_physics` supports numerical-physics evidence when the underlying model/configuration are valid.
- `experimental_sensor` requires genuine measured sensor evidence.
- `device_benchmark` requires measurements on the exact named device.
- `reproducibility` supports computational traceability.
- `release` covers citation, licensing, validation, and packaging metadata.

The submission manifest exposes explicit readiness flags for each class. A `false` flag means that evidence family is absent, not that the underlying scientific claim has been disproved.

## 4. Tables and figures

`TABLE_S1_VALIDATION_METRICS` is generated only from scalar values actually present in `validation/summary.json`. No missing metric is guessed or interpolated.

`TABLE_S2_CLAIMS_TO_EVIDENCE` is copied from the reviewer manifest so manuscript wording can be checked against the evidence available at submission time.

`TABLE_S3_ARTIFACT_PROVENANCE` records hashes of the reviewer evidence artifacts.

Figures from supplied validation, ablation, design, closed-loop, and hardware directories are copied under `figures/` and numbered deterministically as supplementary figures. The index records source role, evidence class, SHA-256, conservative caption text, and a journal-quality reminder.

## 5. Release candidate use

Version `1.0.0rc1` is a software/research-package release candidate. It does not imply that COMSOL, laboratory, fabricated-device, or target-device evidence exists.

The tag-time release workflow can build:

- wheel and source distribution,
- release validation JSON,
- reproducibility evidence,
- reviewer evidence package,
- deterministic publication demo,
- manuscript submission package.

A future stable `v1.0.0` should be created only after the intended software/release gates are deliberately accepted. Physical manuscript claims still require their corresponding real evidence whether the software version is RC or stable.

## 6. Pre-submission boundary

Before uploading to a journal, verify the generated `MANUSCRIPT_CHECKLIST.md`, the claims matrix, figure requirements, journal declarations, data/code availability text, and exact artifact hashes. Do not claim a DOI until one has actually been minted.
