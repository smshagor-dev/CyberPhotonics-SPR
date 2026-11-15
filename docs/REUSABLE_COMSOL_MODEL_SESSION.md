# Stage 14 — One loaded COMSOL model per checkpoint execution session

Stage 8 checkpoints a frozen COMSOL campaign after each validated batch. Previously, each batch called `mph.start()` and `client.load(model.mph)` again. MPh uses one process-wide client and recommends `client.remove(model)` to release a model after use; repeatedly loading a large model may waste memory and wall-clock time. See the [MPh client API](https://mph.readthedocs.io/en/stable/api/mph.Client.html).

Stage 14 introduces a deliberately **opt-in** mode that loads a genuine `.mph` once per `run-batched` invocation while continuing to commit each numerical batch independently.

## Usage

First pass both required real solver gates described in Stage 10 and 11. Then:

```bash
python scripts/run_comsol_verification_campaign.py run-batched --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/comsol.yaml --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json --ri-response-report outputs/research/stage11_ri_probe/COMSOL_RI_RESPONSE_READINESS.json --batch-size 8 --reuse-loaded-model --out outputs/research/stage8_run
```

After a crash or catchable interruption, verify the original worker is no longer alive, handle any abandoned lock according to Stage 8 procedures, and use the **same mode**:

```bash
python scripts/run_comsol_verification_campaign.py run-batched --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/comsol.yaml --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json --ri-response-report outputs/research/stage11_ri_probe/COMSOL_RI_RESPONSE_READINESS.json --batch-size 8 --reuse-loaded-model --resume --out outputs/research/stage8_run
```

Completed SHA-bound batches are never rerun. A resumed invocation loads a **fresh model instance** and creates a distinct execution-session record; Stage 9 still requires the original COMSOL-native log for **each** such invocation. On normal completion or a catchable exception, the runner calls `client.remove(loaded_model)`. It deliberately does not call `client.clear()` or `client.disconnect()` because the MPh client is process-wide and other models may be loaded.

## Safety and compatibility

- The immutable checkpoint manifest stores `model_session_mode: reused_loaded_model`. Resume with the flag missing or newly added after starting a per-batch run is refused before model startup.
- Existing per-batch checkpoints remain byte-compatible: they do not include a new `model_session_mode` field. The legacy `run` command and default `run-batched` behavior are unchanged.
- Both modes continue to store original sample IDs, complete per-batch raw spectra and failed solves, SHA-256 chunk evidence, native session-log provenance and the same Stage 7 sealing schema.
- Failure to load the COMSOL model or license still fails closed; there is **no** fallback to synthetic results. Unit/mode/spectrum checks and source-model hashes remain mandatory.
- Reusing a loaded model does not guarantee physical solver independence between samples. A scientist must examine solver state, numerical initialization, mode tracking and potential order effects on a *real* model. An independent reload-versus-reuse equivalence experiment is still needed before claiming numerical equivalence for a paper.

## Verification evidence

```bash
pytest -q -W error tests/test_reusable_comsol_model.py tests/test_resumable_campaign.py tests/test_comsol_request_derivatives.py
```

The tests inject fake MPh clients and synthetic Lorentzian spectra. They check one load for multiple batches, `client.remove` ownership cleanup, per-batch checkpoint evidence, interruption and resumed loading without rerunning finished samples, mode mismatch rejection, refusal to treat injected runners as the real backend and legacy checkpoint compatibility.

**CI cannot benchmark actual COMSOL startup time, licensing behavior, memory use, physical model equivalence or optical correctness.** No genuine FEM or device results were generated. Journal-readiness blocker [#42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42) remains open.

## Stage 15: independently audit reload versus reuse results

Do not infer that fewer MPh model loads means the same electromagnetic
solution. Use the separate
[Stage 15 source-bound numerical comparison](COMSOL_MODEL_MODE_EQUIVALENCE.md)
on **two complete genuine COMSOL campaigns** using the same immutable
plan and model, one with legacy per-batch reloading and one with
`--reuse-loaded-model`. The audit checks every original RI sample,
raw loss/wavelength array, resonance and FWHM under an explicit frozen
tolerance protocol, while reporting native session log coverage.
Software fixtures and hashes cannot prove real physics or paper
readiness.
