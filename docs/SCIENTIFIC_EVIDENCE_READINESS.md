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
