# Scientific evidence readiness for journal submission

This is an **evidence gap checklist**, not a claim that the research has passed independent validation.
The SoftwareX manuscript used a constructed five-point RI sweep as a software API example;
those numbers are not COMSOL output, measured PCF-SPR performance, or evidence of generalization.

## What is already implemented

- Grouped geometry/RI training and validation splits
- Tandem inverse generator and forward surrogate
- Ridge baseline, validation pack, and ablation scripts
- COMSOL `mph` integration and exact-candidate closed-loop verification
- Evidence registry, provenance hashes, reviewer/submission packaging
- Explicit software-only evidence labeling for synthetic and injected test runners

## Blocking experiments before numerical-photonics journal submission

1. **Qualified independent numerical physics.** Supply a verified COMSOL `.mph` model,
   geometry parameterization, mesh convergence, boundary conditions, correct units,
   materials and raw wavelength/loss outputs. Compare predicted sensitivity, FOM and
   resonance against fixed-geometry RI sweeps from COMSOL, not against the same forward
   surrogate used to optimize the designs.
2. **Reference dataset provenance.** Identify the origin of every training/evaluation
   sample. Clearly separate synthetic spectra, qualified simulation, and measurements.
   Freeze an immutable dataset version and experiment/split hashes.
3. **Quantitative baselines.** Run Ridge and at least one relevant inverse-design
   comparison on the **same geometry-group-disjoint split**, same targets and budgets.
   Report per-metric MAE/RMSE, verified target error, fabrication-valid rate and run cost.
4. **Uncertainty and ablations.** Measure held-out coverage and interval widths;
   report OOD behavior, pre- and post-projection validity, and multiple seeds.
   Compare full method with constraints/calibration/ensemble components removed.
   Calibrate across geometry-group residual maxima (not independent RI rows), and report whether the number of held-out groups supports nominal conformal coverage.
5. **Evidence-bound results.** Report numerical values only after results, configurations,
   model/dataset hashes and test scripts are archived. Do not label synthetic or
   surrogate outputs as electromagnetic or experimental results.
6. **Manuscript revision.** Replace the constructed demonstration as the main result
   with independent physics-backed tables, figures, limitations and a reproducible case
   study. Archive the tested software version and supplementary datasets with stable IDs.

## Suggested commands (after valid data and COMSOL resources exist)

```bash
python scripts/validation_campaign.py preflight \
  --config configs/real_validation_campaign.yaml --strict

python scripts/validation_campaign.py init \
  --config configs/real_validation_campaign.yaml \
  --out outputs/real_validation/campaign

python scripts/research_completion.py run-comsol \
  --campaign outputs/real_validation/campaign

python scripts/run_validation_pack.py \
  --data data/processed/qualified_comsol_training.parquet \
  --checkpoint models/tandem.pt --out outputs/validation
```

The full campaign initialization and COMSOL workflow are documented in
[`REAL_VALIDATION_TO_STABLE_RELEASE.md`](REAL_VALIDATION_TO_STABLE_RELEASE.md)
and [`REAL_VALIDATION_CAMPAIGN.md`](REAL_VALIDATION_CAMPAIGN.md).
The example config contains placeholders and is **not** executable physics evidence.

## Acceptance rule

- A green CI run establishes **software behavior**, not optical validity.
- A valid COMSOL evidence registry establishes **file provenance**, not mesh
  convergence or independent peer-reviewed physical correctness.
- The manuscript is **not ready for a physics-performance claim** until the
  independent COMSOL or experimental results have been generated, reviewed,
  compared and archived.
- Laboratory and exact-device results are necessary **only for claims** about
  fabricated sensors or physical device performance. Do not imply that these
  have been completed when only a numerical study has been performed.

## Known repository example

`reports/phase4_hil_benchmark.json` explicitly records `"protocol": "mock"`.
It may support software testing, never a physical hardware latency or sensor
benchmark claim.


## Stage 2: archived independent numerical-physics audits

The executable spectrum and refinement consistency audits are documented in
[INDEPENDENT_PHYSICS_VALIDATION.md](INDEPENDENT_PHYSICS_VALIDATION.md).
Run both on **actual** completed COMSOL archives (not CI fixtures):

```bash
python scripts/audit_comsol_iteration.py \
  --iteration /path/to/real/closed_loop_iteration \
  --out /path/to/reports/spectrum_consistency.json --strict
python scripts/audit_comsol_convergence.py \
  --archive /path/to/real/refinement_archive \
  --json-out /path/to/reports/refinement_consistency.json --strict
```

Both consistency checks must pass *in addition to* independent scrutiny of
the actual COMSOL physics model, material dispersion, mode selection, boundaries
and mesh/spectral convergence design. These checks do not automatically
establish a physically validated sensor, and no such evidence is currently
present in the tracked repository.


## Stage 3: actual grouped ML comparisons (engineering evidence)

Reproducible forward baselines and RI-feature ablation with identical held-out
geometry groups across methods, per-target physical-unit errors, split hashes
and paired group-bootstrap CIs are documented in
[ML_STATISTICAL_VALIDATION.md](ML_STATISTICAL_VALIDATION.md).

```bash
python scripts/run_grouped_ml_study.py \
  --data /path/to/frozen_source_labeled_dataset.parquet \
  --out outputs/research/stage3 \
  --seeds 7,17,29,41,53 \
  --trees 120 --bootstrap-resamples 2000
```

Explicit `--allow-software-only` is required for synthetic demonstrations.
This forward-model comparator is **not** the final published tandem
inverse-design benchmark, physical validation or evidence of improved PCF-SPR
sensor sensitivity. Real independent solver results and a matched-budget
inverse-method comparison are still pending.


## Stage 4: matched-query inverse-design benchmark

The executable side-by-side tandem vs reference-retrieval vs uniform-random
comparison and zero-latent ablation are in
[MATCHED_INVERSE_BENCHMARK.md](MATCHED_INVERSE_BENCHMARK.md).
A newly trained checkpoint binds its exact dataset SHA-256 and geometry
train/held-out split IDs to the evaluation; the benchmark refuses old unbound
checkpoints or inconsistent partitions.

```bash
python scripts/run_inverse_benchmark.py \
  --data /path/to/frozen_training_dataset.parquet \
  --checkpoint /path/to/retrained_tandem.pt \
  --out outputs/research/stage4 \
  --seeds 7,17,29 --candidates-per-target 16
```

All methods get the same forward-surrogate ranking-query budget. This
does **not** equalize total runtime or independent COMSOL solver cost.
Selected designs enter `PHYSICS_VERIFICATION_REQUESTS.csv` with
`verification_status=not_run`. The model's own validation geometries are
not an untouched external test, and generation seeds are **not**
independent model-training replicates. An actual solver study, external
holdout and physical claim review remain mandatory.


## Stage 5: independent COMSOL request-to-spectrum comparison

The Stage 5 importer connects each frozen `PHYSICS_VERIFICATION_REQUESTS.csv`
entry from Stage 4 to a hash-bound COMSOL execution archive. It recomputes
sensor resonance/sensitivity/FOM from complete RI spectra, verifies fixed
geometry and source/refinement model hashes, and counts **all** solver
failures in the denominator. See
[COMSOL_INDEPENDENT_VERIFICATION.md](COMSOL_INDEPENDENT_VERIFICATION.md).

```bash
python scripts/verify_inverse_comsol.py \
  --benchmark outputs/research/stage4 \
  --solver-archive /path/to/real/solver_archive \
  --refinement-archive /path/to/real/refinement_archive \
  --out outputs/research/stage5 \
  --strict
```

**Important:** Without authentic external solver model/log/export archives,
this command correctly produces a failed/blocked audit instead of inventing
physics. Consistent file hashes and recomputed metrics are *not* proof of
physical correctness or real COMSOL execution. The paper's independent
physics validation and publication gate remain open research blockers.


## Stage 6: claim-traceable manuscript results draft

An executable manuscript-working package now checks frozen Stage 4/5 hashes,
reruns Stage 5 when genuine external COMSOL archives and registry are provided,
and creates a claim-level evidence ledger and manuscript draft. See
[CLAIM_TRACEABLE_PUBLICATION.md](CLAIM_TRACEABLE_PUBLICATION.md).

The builder deliberately withholds optical metrics from a Stage 5 JSON file
alone. All numbers are review-only even after a numerical archive re-audit,
and independent solver/source authenticity plus electromagnetic model
scientific review still block publication approval.

A blocker-only package can be generated without real COMSOL archives:

    python scripts/build_claim_evidence_package.py --benchmark outputs/research/stage4 --out outputs/publication/stage6

C03 physical correctness, C04 independently tested tandem superiority and
C06 journal readiness remain blocked until external experiments, independent
review and stronger statistical design are completed.


## Stage 7: run actual COMSOL evaluation of frozen inverse designs

The first real COMSOL request execution entrypoint is now documented in
[REAL_COMSOL_CAMPAIGN.md](REAL_COMSOL_CAMPAIGN.md). It freezes Stage 4
request/model/config hashes and RI/normalization protocol **before**
execution, invokes the existing local `mph` COMSOL backend for all queued
RI points (no synthetic fallback), records all solver failures and outputs
raw wavelength/loss spectra. Final sealing requires a separately sourced
native solver log and a matching, numerically consistent refinement archive.

```bash
python scripts/run_comsol_verification_campaign.py --help
python scripts/run_comsol_verification_campaign.py plan --help
python scripts/run_comsol_verification_campaign.py run --help
python scripts/run_comsol_verification_campaign.py seal --help
```

The executable path and unit-tested failure handling **do not** mean
COMSOL has been run on the repository's test infrastructure: the real
model, license, genuine native solver logs and physical review have not
been supplied. The publication evidence gate remains closed.


## Stage 8: resumable real COMSOL sweep campaign

A checkpointed solver runner can now commit batches with SHA-256 integrity,
validate saved spectra and design identities before resuming, and reject
altered or missing batch evidence. See
[RESUMABLE_COMSOL_CAMPAIGN.md](RESUMABLE_COMSOL_CAMPAIGN.md).

```bash
python scripts/run_comsol_verification_campaign.py run-batched --help
python scripts/run_comsol_verification_campaign.py inspect --help
```

The Stage 7 seal path now checks checkpointed source archives against the
individual batch journal before proceeding to Stage 5. This does **not**
run COMSOL in GitHub CI, certify electromagnetic correctness, or solve
the absence of genuine model, native log and independent refinement inputs.


## Stage 9: native solver log per COMSOL execution session

[COMSOL_SESSION_PROVENANCE.md](COMSOL_SESSION_PROVENANCE.md) describes
new session-bound log receipts, explicit crashed-session closure,
mandatory all-session coverage before sealing, and sealed Stage 5
reverification. The physical authenticity and independent scientific
review blocker **remains open**; no COMSOL was run by CI.


## Stage 10: live model and one-RI pilot readiness

[COMSOL_MODEL_PILOT_GATE.md](COMSOL_MODEL_PILOT_GATE.md) documents a
separate fail-closed structural check and optional *real* local solver
pilot before committing time to a large RI campaign. The gate requires
the frozen Stage 4/7 inputs, explicit unit conversion, model parameter
names, matching study, materials, mesh and physics interface presence,
and one physically generated, numerically consistent spectrum to enable
**operational** solver-run readiness.

CI uses a dummy model and template config to verify rejection, not to
claim physics results. Even a successful genuine one-point pilot does
**not** establish sensor sensitivity, FOM, mesh convergence, model
scientific validity or journal readiness.


## Stage 11: RI parameter coupling and repeatability before long solves

After passing the Stage 10 one-RI model gate, run the separate
[RI response sanity diagnostic](COMSOL_RI_RESPONSE_GUARD.md). It solves
the first frozen geometry at low, center, high, then **repeated center**
analyte RI, checking that the resonance actually responds monotonically
and repeatably to RI changes on a resolved wavelength grid.

Both `run` and `run-batched` now require **two** source-bound reports:
`--readiness-report` and `--ri-response-report`. CI exercises software
fixtures and cannot pass them off as physical simulations. A diagnostic
pass does not establish full-sweep sensitivity, mesh convergence, solver
authenticity or journal-ready numerical evidence.


## Stage 12: protect solver spectra and geometry values against false success

[COMSOL_SPECTRUM_QUALITY.md](COMSOL_SPECTRUM_QUALITY.md) describes
engineering checks that refuse flattened multi-eigenmode outputs,
non-real/negative loss, unresolved or boundary-truncated resonance peaks,
invalid units and nonfinite geometric parameters. The original COMSOL
runner and downstream archived/checkpointed spectrum readers use the
same numerical gate. Synthetic tests of its failure handling do **not**
verify genuine optical FEM outputs or publication-readiness.
