# Stage 15 — Compare real COMSOL reload and reuse modes sample by sample

**Purpose:** Stage 14 introduced `run-batched --reuse-loaded-model`. Whether a long-lived loaded COMSOL model changes solver state, mode selection, resonance wavelengths, loss curves or FWHM relative to reloading the model every batch is an unanswered *physical numerical* question. Stage 15 supplies a strict **offline comparison** with two independent source-bound campaigns. This does **not** itself run COMSOL or produce new optical observations.

## 1. Freeze tolerances and inputs

Choose numerical tolerances based on an independent solver/grid-precision argument, **before looking at the two run results**. Example numbers below are placeholders to be justified, NOT scientifically validated PCF-SPR tolerances:

```bash
python scripts/audit_comsol_model_modes.py freeze \
  --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 \
  --model /real/pcf_spr.mph --config /real/comsol.yaml \
  --max-grid-nm 0.01 --max-resonance-nm 0.15 --max-fwhm-nm 0.15 \
  --max-loss-db-per-cm 0.002 --out outputs/research/stage15_protocol
```

`COMSOL_MODE_COMPARISON_PROTOCOL.json` contains all model/plan/benchmark/config SHA-256 digests, the exact frozen sample count and tolerance values. Local file creation times and hashes cannot prove external preregistration, so archive your original timestamped protocol outside the research workspace if independent provenance is required.

## 2. Execute the *same* real frozen COMSOL plan twice

First satisfy the existing mandatory Stage 10 + 11 real COMSOL pilot/report checks; use the **same model, configuration, plan and batch size** in two separate fresh directories. One is the default per-batch mode, the other adds `--reuse-loaded-model`:

```bash
python scripts/run_comsol_verification_campaign.py run-batched \
  --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 \
  --model /real/pcf_spr.mph --config /real/comsol.yaml \
  --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json \
  --ri-response-report outputs/research/stage11_ri_probe/COMSOL_RI_RESPONSE_READINESS.json \
  --batch-size 8 --out outputs/research/stage15_reload

python scripts/run_comsol_verification_campaign.py run-batched \
  --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 \
  --model /real/pcf_spr.mph --config /real/comsol.yaml \
  --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json \
  --ri-response-report outputs/research/stage11_ri_probe/COMSOL_RI_RESPONSE_READINESS.json \
  --batch-size 8 --reuse-loaded-model --out outputs/research/stage15_reuse
```

Preserve genuine externally obtained COMSOL-native logs for **every** execution session (including interruptions/resumes), using the existing `bind-log` and `audit-logs` commands. These scripts do not automatically capture or authenticate native logs.

## 3. Independently recheck every checkpoint and paired spectrum

```bash
python scripts/audit_comsol_model_modes.py audit \
  --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 \
  --model /real/pcf_spr.mph --config /real/comsol.yaml \
  --protocol outputs/research/stage15_protocol/COMSOL_MODE_COMPARISON_PROTOCOL.json \
  --reload-run outputs/research/stage15_reload --reuse-run outputs/research/stage15_reuse \
  --out outputs/research/stage15_mode_audit --strict
```

Audit checks: the two runs are **distinct** and use exactly matching frozen input SHA-256; correct *opposite* `model_session_mode` values; the expected complete sample denominator; all checkpoint/chunk/final-sweep SHA-256 and session ownership; no unfinished sessions; the original target request, geometry and RI identity for each sample. Every pair compares the full raw wavelength vector, full loss vector, independently rechecked resonance, and FWHM under explicit frozen tolerances. No resampling or interpolation hides mismatched grids; any missing/failing solve, differing row count, exceeded tolerance or altered saved spectrum is a failure.

Outputs are `COMSOL_MODE_EQUIVALENCE_ROWS.csv` (one row per frozen solver sample) and `COMSOL_MODE_EQUIVALENCE_AUDIT.json` with explicit numeric-pair pass counts, the two run digests, native log coverage and unqualified source limitations.

`--strict` exits 2 until **all numeric pairs pass and both session-native-log ledgers are complete**. Even when strict passes, `source_authenticity_verified`, `physical_correctness_verified` and `scientific_submission_ready` all remain **false**: user-supplied logs plus hashes do not authenticate genuine COMSOL execution, verify eigenmode tracking or prove Maxwell model correctness.

## 4. Genuine experiment that remains outstanding

This software cannot perform the experiment without a licensed COMSOL runtime, saved `.mph` model and independent physically grounded raw sweeps. After running both modes, repeat comparison under intentionally changed solver initialization / run order, review eigenmode identity, mesh/PML, material-dispersion and convergence, and publish only real independently reviewed evidence. Existing [issue #42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42) remains the controlling journal-submission blocker.

CI uses mock COMSOL-labeled software fixtures only, including fabricated Lorentzian loss curves and fabricated local log bytes. Those exercises test falsification, not real optics. No performance acceleration, equivalence or sensitivity claim is established by CI.