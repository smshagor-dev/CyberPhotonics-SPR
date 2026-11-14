# Stage 11 — COMSOL RI-response sanity and repeated-center gate

One COMSOL spectrum at the first request's central refractive index is not sufficient evidence that the **analyte refractive-index parameter actually affects the computed SPR resonance**. A model may load correctly yet have a disconnected analyte parameter, incorrect optical expression, misconfigured study, unresolved resonance, or poor repeatability.

Stage 11 solves four points from the same frozen Stage 7 request without changing any geometric dimensions:

| Diagnostic call | Exact frozen selection |
|---|---|
| 1 — Low RI | First analyte RI of the first request |
| 2 — Center RI | Pre-registered central RI of the first request |
| 3 — High RI | Last analyte RI of the first request |
| 4 — Repeat center | Same exact geometry and RI as call 2 |

These probes **are not a full RI sweep** and cannot support manuscript sensitivity, figure of merit, statistical superiority, physical correctness, experimental sensor claims, or mesh convergence. Their purpose is to catch obvious modeling/extraction mistakes **before** many solver hours are spent.

## Real execution

First execute the Stage 10 real model pilot successfully:

```bash
python scripts/probe_comsol_model.py --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/validated_pcf_spr.mph --config /real/comsol_settings.yaml --out outputs/research/stage10_pilot --live --pilot --strict
```

Then execute the separate four-call RI response probe. Choose the numeric thresholds **before seeing these results**, based on the wavelength grid, model precision, and scientific expectations rather than trying thresholds until a study passes:

```bash
python scripts/probe_comsol_ri_response.py --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/validated_pcf_spr.mph --config /real/comsol_settings.yaml --pilot-readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json --min-shift-nm 1.0 --max-repeat-nm 0.5 --min-diagnostic-r2 0.90 --out outputs/research/stage11_ri_probe --strict
```

The numbers `1.0 nm`, `0.5 nm` and `0.90` are **example diagnostic settings only**, not universal PCF-SPR tolerances. Document and justify real thresholds using a separate scientific numerical-precision analysis. The program writes `RI_PROBE_PROTOCOL.json` with frozen request identities, thresholds and hashes **before** calling COMSOL, so result values cannot silently be compared to a different protocol afterward.

Checks include: all four successful solver calls, identical wavelength grid, at least 16 finite samples, internal resonance and FWHM recomputation, resonance away from truncated spectral boundaries, resolved peak width, no physically negative confinement loss, monotonic low/center/high resonance tracking, minimum detectable low-to-high shift (at least half a wavelength step), repeated-center resonance/FWHM agreement and a three-point *diagnostic-only* trend R².

## Rejected conditions

The gate refuses (a) unavailable real COMSOL or invalid Stage 10 single-point pilot; (b) changed frozen model/plan/config; (c) no RI response; (d) resonance jumping modes/nonmonotonic tracking; (e) insufficient peak sampling, mismatched grids or cut-off peaks; (f) unstable repeat-center results; (g) missing/failed solver calls; (h) altered raw spectra, preregistered thresholds or probe identities. Every returned report retains `scientific_submission_ready=false` and `physical_correctness_verified=false`.

## New mandatory full-campaign invocation

The Stage 7 and Stage 8 full-run commands require **both** operational gate reports:

```bash
python scripts/run_comsol_verification_campaign.py run-batched --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/validated_pcf_spr.mph --config /real/comsol_settings.yaml --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json --ri-response-report outputs/research/stage11_ri_probe/COMSOL_RI_RESPONSE_READINESS.json --batch-size 8 --out outputs/research/stage8_run
```

The CLI rechecks frozen model/plan/config hashes; the Stage 10 single-point spectrum; and the Stage 11 protocol, every stored RI spectrum, sample identity, shift and repeatability statistic. Missing reports or a software test fixture prevent the expensive production run from starting.

## Evidence produced

- `RI_PROBE_PROTOCOL.json`: pre-solver response detection thresholds and source hashes.
- `RI_PROBE_SPECTRA.csv`: the four direct solver response spectra or explicit failure records; every value remains **unqualified**.
- `COMSOL_RI_RESPONSE_READINESS.json`: input digests, protocol hash, numerical diagnostics, reasons for rejection and operational readiness only.

Native COMSOL solver logs must be independently retained from these preflight solves. The gate does not automatically capture native logs, authenticate COMSOL execution, verify PML/dispersion/mode selection, or validate scientific performance. Stage 5 real optical/numerical evidence, external cohorts, mesh convergence and qualified scientific review remain required; see [issue #42](https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42).

GitHub CI uses intentionally fictional `.mph` bytes and synthetic Lorentzian spectra solely for falsification/robustness regression tests. **CI never runs actual COMSOL and never establishes physical results.**