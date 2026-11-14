# Stage 8 — Resumable COMSOL campaigns with verifiable batches

The Stage 7 `run` command performs the entire scheduled RI sweep in a single
solver call. A workstation crash or license interruption can lose that work.
Stage 8 provides **incremental COMSOL batches**, persistent per-batch SHA-256
evidence, **read-only audit** of the committed progress and fail-closed resume.

This is actual production code for the local `mph` COMSOL backend when
licensed COMSOL and a real model are available. The CI tests inject software
fixtures instead. No solver is running on GitHub Actions and no physical
performance has been validated by this change.

## Execute in batches

First prepare a frozen plan with the Stage 7 `plan` command as described
in [REAL_COMSOL_CAMPAIGN.md](REAL_COMSOL_CAMPAIGN.md).

```bash
python scripts/run_comsol_verification_campaign.py run-batched \
  --plan outputs/research/stage7_plan \
  --benchmark outputs/research/stage4 \
  --model /path/to/real/pcf_spr.mph \
  --config /path/to/real/solver.yaml \
  --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json \
  --batch-size 8 \
  --out outputs/research/stage8_run
```

The production path invokes `run_comsol_geometries` for **only the next
unsolved batch**. The backend returns sample indices relative to the current
batch; the journal stores the original global IDs to preserve the Stage 4
frozen request identity.

A successful batch is checked for matching geometry, RI, full strictly
increasing finite spectra, and independently recomputed resonance/FWHM.
An individual `failed: reason` solver result is a committed failure, not
silently retried or dropped. When COMSOL cannot even launch, the function
raises and does **not** write a successful execution manifest.

Each committed batch contains:

- `chunks/batch_<start>_<stop>.csv`: one row per original planned sample,
  including request ID, complete spectrum or explicit solver failure.
- `chunks/batch_<start>_<stop>.json`: source mode, exact frozen plan SHA,
  range, success/failure counts and CSV SHA-256.
- `campaign_progress.json`: interim counts; recomputed from saved chunk
  files, not an independently trusted claim of completion.
- `checkpoint_manifest.json`: immutable plan/model/config/batch-size/source
  contract. Changing batch size or backend between runs is forbidden.

The output directory is never overwritten. While a campaign runs, a lock
file prevents accidental simultaneous writers. An interrupted run retains
fully committed batches, even when the next solver batch crashes.

## Inspect or resume safely

```bash
python scripts/run_comsol_verification_campaign.py inspect \
  --plan outputs/research/stage7_plan \
  --benchmark outputs/research/stage4 \
  --model /path/to/real/pcf_spr.mph \
  --config /path/to/real/solver.yaml \
  --out outputs/research/stage8_run
```

`inspect` does **not** start or import COMSOL. It recomputes every committed
chunk's SHA-256, validates the frozen plan and all stored sample geometries,
RI points and spectra-derived metrics. It reports total committed, remaining
and failed numerical solver samples.

To resume:

```bash
python scripts/run_comsol_verification_campaign.py run-batched \
  --plan outputs/research/stage7_plan \
  --benchmark outputs/research/stage4 \
  --model /path/to/real/pcf_spr.mph \
  --config /path/to/real/solver.yaml \
  --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json \
  --batch-size 8 --resume \
  --out outputs/research/stage8_run
```

It fails before starting COMSOL when the checkpoint source hashes, saved
batch size, model/config files, backend kind or sample metrics do not
match. All completed chunks are verified again. The command never
reconstructs a missing solver result from the surrogate.

A hard process kill may leave `.active-comsol-campaign.lock`. Verify that
**no old campaign worker is alive** before adding the explicit
`--recover-stale-lock` flag. Using this flag while a genuine worker runs
could lead to two concurrent jobs and corrupted evidence.

A crash during a batch file write may leave an orphan CSV, JSON or
`.partial` file. The tool deliberately **rejects** such uncommitted
evidence rather than silently claiming it was finished. Quarantine the
orphan bytes outside the `chunks/` directory after inspecting them and
rerun `--resume`; the batch will be solved again. Never manually modify
or rehash a committed chunk to bypass the check.

## Stage 5–6 compatibility

After all planned sweeps are committed, the workflow emits the **exact
Stage 7 `sweeps.csv` and `execution_manifest.json` contract**, plus the
new chunk ledger. The existing `seal` command works on the same run
folder:

```bash
python scripts/run_comsol_verification_campaign.py seal \
  --plan outputs/research/stage7_plan \
  --run outputs/research/stage8_run \
  --benchmark outputs/research/stage4 \
  --model /path/to/real/pcf_spr.mph \
  --config /path/to/real/solver.yaml \
  --native-solver-log /path/to/authentic_comsol_native.log \
  --refinement-archive /path/to/real-refinement-evidence \
  --out outputs/research/stage8_sealed
```

**Every real COMSOL execution session must have its native logs retained
and independently inspected.** The existing sealing interface currently
accepts a supplied native solver log and does not automatically verify that
it covers every resumed session. Do not call it a fully authenticated
scientific provenance chain without a separately reviewed multi-session
solver-log ledger.

A sealed archive must still pass Stage 5 fixed-geometry RI consistency,
mesh/spectral convergence, source authenticity review and Stage 6 claim
qualification before numerical metrics may be used in a journal paper.
No fabricated result, fake model bytes, test spectra, or imaginary DOI
may be presented as a physical experiment.

## CI and test boundary

Regression tests use deliberately fake model bytes and Lorentzian loss
arrays, explicitly labeled `test_fixture` and `software_only`. They
exercise:

- Interruption after one committed batch and resume without replaying it.
- Tampered CSV, altered model, mismatched batch size or backend, missing
  chunk manifest and orphan lock refusal.
- Explicit failed RI points in the result denominator.
- Read-only `inspect` without an installed COMSOL runtime.
- Sealing refusal for injected software-only runs.

CI success means **engineering behavior is covered**, not actual COMSOL
solver execution or independent electromagnetic verification.

Research blocker: https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42

 
## Stage 9: require a true per-session COMSOL-native log

Each uninterrupted or resumed solver invocation now receives a unique
session ID. The batch journal binds every committed chunk to that session,
and sealing requires an externally obtained **original native solver log
for every invocation**. Use bind-log and audit-logs before seal; see
[COMSOL_SESSION_PROVENANCE.md](COMSOL_SESSION_PROVENANCE.md).

The Stage 5 audit checks all copied session logs and receipt SHA-256
against the sealed source bundle. The registered native log bytes are
still operator-supplied: hashes alone **do not** prove that COMSOL executed
the claimed simulation or that the optical model is physically correct.
