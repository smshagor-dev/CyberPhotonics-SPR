# Stage 7 — Run authentic COMSOL RI sweeps for frozen inverse designs

This is the first workflow that actually **calls the existing COMSOL mph
solver backend** for Stage 4's frozen request queue. It does not synthesize
optical spectra or treat CI tests as experimental physics. To execute the
real solver, you need an accessible licensed COMSOL Multiphysics installation,
a validated physical photonic-crystal-fiber/surface-plasmon-resonance
`.mph` model, and the model-specific expressions and units in the YAML
configuration.

The repository and GitHub Actions have **not** provided these external
assets. No actual COMSOL performance results are claimed here.

## 1. Preregister the campaign before solving

Freeze all methods/seeds/target geometries, the RI protocol, source hashes
and normalization scales *before viewing simulation results*.

```bash
python scripts/run_comsol_verification_campaign.py plan \
  --benchmark outputs/research/stage4 \
  --model /path/to/validated/real_pcf_spr.mph \
  --config /path/to/real_comsol_settings.yaml \
  --solver-version "COMSOL Multiphysics ACTUAL VERSION" \
  --ri-span 0.04 --ri-points 5 \
  --sensitivity-scale <predeclared-positive-scale> \
  --fom-scale <predeclared-positive-scale> \
  --resonance-scale <predeclared-positive-scale> \
  --out outputs/research/stage7_plan
```

Replace each placeholder with a justified actual value. Units for scales
must match the expected metric units. The underlying model must use validated
physical dispersive materials, appropriate boundary/PML setup, exact unit
conversion, mode selection and mesh settings; the CLI cannot inspect these
scientific details and does not claim it has done so.

The plan requires Stage 4's frozen `INVERSE_BENCHMARK_MANIFEST.json`,
`PHYSICS_VERIFICATION_REQUESTS.csv`, and `selected_designs.csv`. It
rejects mismatched hashes, edited request status, duplicate identities, changed
geometry, impossible fabrication parameters, invalid target values and RI
protocols. For **every** selected method/target/seed, it generates an
individual centered fixed-geometry RI sweep. Duplicate geometries arising
from different methods are deliberately kept as distinct requests so
per-method completion costs can be accounted for consistently.

Outputs:

- `campaign_plan.json`: Stage 4 request/manifest SHA-256, model/config
  hashes, installed solver-version declaration, unit-specific normalization
  scales, immutable planned count and missing scientific qualifications.
- `planned_sweeps.csv`: exact request ID, target, generation seed, method,
  fixed geometry, analyte RI and sample index for every scheduled solver call.

**Planning does not run COMSOL.**

## 2. Execute the real solver

```bash
python scripts/run_comsol_verification_campaign.py run \
  --plan outputs/research/stage7_plan \
  --benchmark outputs/research/stage4 \
  --model /path/to/validated/real_pcf_spr.mph \
  --config /path/to/real_comsol_settings.yaml \
  --out outputs/research/stage7_execution
```

The production path directly invokes
`sprpcf.simulation.comsol_sweep.run_comsol_geometries`, which uses the
installed `mph` package, opens the actual `.mph` file, sets physical
parameters and solves the configured COMSOL study. This is *not* a command
that silently falls back to synthetic spectra, mock geometry or replay.
If COMSOL is not installed, the license cannot start, or model loading
fails, execution raises and does **not** write a successful execution
manifest.

It validates one returned spectrum per exact planned sample, geometry
and RI; requires at least 16 finite, monotonically increasing wavelengths;
recomputes and checks resonance/FWHM; and records all per-sample solver
failure statuses. No failed row is dropped, and all results remain
unqualified until later scientific checks.

Outputs:

- `sweeps.csv`: direct model-returned loss/wavelength arrays, result
  metrics, original request IDs, RI, source and status for each solver call.
- `execution_manifest.json`: frozen hashes, sample counts, run UTC
  timestamps, success/failure counts and an explicit
  `source_authenticity_verified=false` scientific boundary.

The current COMSOL backend submits the campaign as one sequential sweep,
not a distributed checkpointed run. A crash or license interruption may
require rerunning it into a new output directory. Do not discard failures.
The execution manifest is an *API invocation record*, not an independently
obtained native COMSOL solver log.

## 3. Preserve native COMSOL solver log and convergence evidence

Get the **genuine** native COMSOL solver log from the actual execution,
not a manually written test log or Python wrapper log. Independently
complete the mesh/spectral refinement archive using Stage 2. It must
contain at least three levels for each refinement type, model/config/log,
source hashes and recorded outputs, all tied to the **same model bytes**.

Then seal an immutable Stage 5 archive:

```bash
python scripts/run_comsol_verification_campaign.py seal \
  --plan outputs/research/stage7_plan \
  --run outputs/research/stage7_execution \
  --benchmark outputs/research/stage4 \
  --model /path/to/validated/real_pcf_spr.mph \
  --config /path/to/real_comsol_settings.yaml \
  --native-solver-log /path/to/actual-comsol-solver.log \
  --refinement-archive /path/to/real-refinement-archive \
  --out outputs/research/stage7_sealed
```

Sealing refuses an injected test runner, edited execution manifest or
changed model/config, missing native log, unmatched model refinement hash
or a failed numerical refinement archive. It copies original model,
config, log and exported sweeps unchanged into the sealed archive and
writes the Stage 5-compatible `verification_manifest.json` with SHA-256
for all four files and frozen Stage 4/refinement bindings.

### Perform Stage 5 scientific numerical-consistency audit

```bash
python scripts/verify_inverse_comsol.py \
  --benchmark outputs/research/stage4 \
  --solver-archive outputs/research/stage7_sealed \
  --refinement-archive /path/to/real-refinement-archive \
  --out outputs/research/stage7_independent_check --strict
```

A successful numerical audit **is not physical authenticity or Maxwell-model
correctness verification**. Even after sealing, a researcher/reviewer must
inspect raw native logs, saved model, study geometry, gold/material
dispersion, PML, mode tracking, unit conversion, mesh convergence, fixed-RI
spectra and comparisons to independent published physics.

### Engineering tests are NOT research results

Regression tests inject Lorentzian spectra and dummy `.mph` bytes.
Those results are explicitly `test_fixture`/`software_only` and
**cannot be sealed** as real COMSOL evidence. GitHub Actions only tests
the plan builder with a deliberately fake file; it does not run or claim
a COMSOL simulation, and cannot provide raw optical results.

Scientific research blocker: https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42


## Stage 8: crash-resilient batch execution

For lengthy COMSOL sweeps, prefer the source-bound and resumable
`run-batched` command described in
[RESUMABLE_COMSOL_CAMPAIGN.md](RESUMABLE_COMSOL_CAMPAIGN.md).
Its per-batch journal allows interrupted sessions to resume without
rerunning completed samples, and the original `seal` command now
verifies the **entire batch ledger** against the final results CSV.
These checks do not replace retention of authentic native solver logs
for every run session and independent model/source qualification.
