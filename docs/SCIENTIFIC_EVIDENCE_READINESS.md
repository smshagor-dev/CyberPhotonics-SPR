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
