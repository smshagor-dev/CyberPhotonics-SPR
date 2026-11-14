# Stage 6 — Claim-traceable publication results

Stage 6 adds evidence-to-claim traceability on top of the existing generic reviewer and submission builders. A green CI job is NOT evidence of optical physics, independently executed COMSOL, or manuscript acceptance.

## What the builder verifies

- Requires the frozen Stage 4 inverse-design manifest and request, selected-design, target and split tables; confirms their SHA-256 digests.
- Optionally consumes the Stage 5 audit. Numerical claims are **withheld** unless its source/input hashes agree with Stage 4.
- To release even review-only numerical values, both real external solver and mesh-refinement archives and a hash-valid evidence registry are required. Stage 6 reruns Stage 5 with the supplied physical archives, then compares recomputed file hashes, request counts and statuses with the original Stage 5 audit.
- Human inspection of the authentic COMSOL electromagnetic model, boundary conditions, complex dispersion, PML, mesh, logs and actual solver execution is a **separate mandatory sign-off**. A registry alone confirms file integrity, not physical correctness.
- Failed or missing solver sweeps cannot be silently omitted. Surrogate-trained selection is not an independent physical benchmark; multiple training seeds, untouched external targets, OOD and matched total cost are still pending.

## Run without real physical archives

The following produces a blocker-first reviewer draft; it will not invent missing COMSOL results.

    python scripts/build_claim_evidence_package.py --benchmark outputs/research/stage4 --stage5 outputs/research/stage5 --out outputs/publication/stage6

Use a different output directory for subsequent runs. The builder refuses to overwrite an existing non-empty output.

For an explicit fail-closed availability check, add --strict-numerical-candidates. The command then exits with code 2 if no independently re-audited numerical values are available, but still writes its blocker report.

## Run with genuine archived COMSOL results

    python scripts/build_claim_evidence_package.py --benchmark outputs/research/stage4 --stage5 outputs/research/stage5 --solver-archive /path/to/real/solver_archive --refinement-archive /path/to/real/refinement_archive --evidence-registry /path/to/evidence_registry.json --bootstrap-resamples 500 --out outputs/publication/stage6_source_bound

Set the same bootstrap count as the original Stage 5 run. Do not edit earlier reports to force hashes to match. Original solver/model/log/spectra must remain preserved separately.

## Files produced

| File | Scope |
|---|---|
| CLAIMS_TO_EVIDENCE.csv | Claim IDs C01–C06 and supported, blocked, or unqualified status |
| CANDIDATE_NUMERICAL_VALUES_FOR_REVIEW.csv | Empty without genuine source-bound re-audit; always marked unqualified |
| SOLVER_COMPLETION_FOR_REVIEW.csv | Total and completed/failed requests per method |
| MANUSCRIPT_RESULTS_DRAFT.md | Manuscript working text and explicit blockers |
| PUBLICATION_EVIDENCE_MANIFEST.json | Commit, artifact digests, provenance and submission gate |
| checksums.sha256 | Tamper-detection for generated tables, figures and manifest |
| figures/solver_request_completion_unqualified.png | Generated only for externally re-audited solver completion counts; NOT a sensor-performance figure |

The scientific_submission_ready, independent_physics_review_completed and source_authenticity_verified flags remain false. C03 (independently reviewed physics), C04 (verified tandem superiority) and C06 (journal submission readiness) remain blocked. C05 (fabricated sensor measurement) is not claimed.

Numerical values may be prepared *for reviewer examination* only after the source-bound re-audit. Their presence never constitutes publication-ready independent optics evidence.

Once real experiments and reviewer scientific sign-off are complete, use the existing scripts/build_reviewer_package.py, scripts/build_submission_package.py and scripts/finalize_evidence_package.py for further journal preparation. Avoid interpreting a generated supplementary package as evidence of scientific correctness.

Research blocker: https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42
