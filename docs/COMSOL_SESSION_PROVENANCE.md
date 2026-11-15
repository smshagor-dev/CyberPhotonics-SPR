# Stage 9 — Session-by-session native COMSOL log provenance

Stage 8 can resume interrupted FEM campaigns. Stage 9 binds **each separate invocation of the local COMSOL backend** to an immutable session ID and one externally retained original native solver log. Merely logging a Python call or retaining a single log for the final resume session is not sufficient.

**No live COMSOL execution occurs in GitHub CI.** Fixtures are synthetic and explicitly unqualified. File hashes cannot prove genuine COMSOL execution, electromagnetic correctness or independent physical performance.

## Run and inspect

Use the Stage 7 plan and Stage 8 batch runner:

```bash
python scripts/run_comsol_verification_campaign.py run-batched --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/config.yaml --readiness-report outputs/research/stage10_pilot/COMSOL_MODEL_PILOT_READINESS.json --ri-response-report outputs/research/stage11_ri_probe/COMSOL_RI_RESPONSE_READINESS.json --batch-size 8 --out outputs/research/stage8_run
python scripts/run_comsol_verification_campaign.py inspect --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/config.yaml --out outputs/research/stage8_run
```

Each invocation creates a solver_sessions/<session-id>.start.json record. Every committed batch records that ID, and each catchable interruption or normal completion creates the matching <session-id>.end.json. Inspect the start filenames or the inspect_campaign_checkpoint API session_evidence list to obtain actual session IDs.

## Bind the original native solver log for EACH session

Enable native COMSOL logging in the physical COMSOL installation before execution. Preserve the true log from every invocation, including interrupted/failed sessions. Bind each finished session:

```bash
python scripts/run_comsol_verification_campaign.py bind-log --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/config.yaml --out outputs/research/stage8_run --session-id YOUR_ACTUAL_32_HEX_SESSION_ID --native-solver-log /real/native/comsol/session-specific.log
```

This copies bytes to native_solver_logs/<session-id>.log and creates a non-overwritable JSON receipt binding the source session records, exact committed chunk manifest digests, log digest, and byte count. A log receipt **does not authenticate its physical origin**.

Then validate all sessions and their logs:

```bash
python scripts/run_comsol_verification_campaign.py audit-logs --plan outputs/research/stage7_plan --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/config.yaml --out outputs/research/stage8_run
```

The audit refuses missing/changed logs, altered receipt hashes, unknown chunk/session ownership, orphan evidence, or unclosed sessions. **One missing native solver log blocks sealing the entire checkpointed campaign.**

## Process crash and explicit closure

A hard termination may leave a session start file without an end marker. First verify the original process has terminated and handle the abandoned worker lock safely. Only then use:

```bash
python scripts/run_comsol_verification_campaign.py close-crashed-session --out outputs/research/stage8_run --session-id YOUR_ACTUAL_32_HEX_SESSION_ID
```

This marks the session manually_closed_after_crash, does not imply a successful solve, and does not waive the requirement for its original native log. Do not forge or reconstruct a lost native log.

## Seal and audit

```bash
python scripts/run_comsol_verification_campaign.py seal --plan outputs/research/stage7_plan --run outputs/research/stage8_run --benchmark outputs/research/stage4 --model /real/pcf_spr.mph --config /real/config.yaml --native-solver-log /real/native/comsol/registered-session.log --refinement-archive /real/refinement_archive --out outputs/research/stage9_sealed
python scripts/verify_inverse_comsol.py --benchmark outputs/research/stage4 --solver-archive outputs/research/stage9_sealed --refinement-archive /real/refinement_archive --out outputs/research/stage9_checked --strict
```

Checkpointed sealing now requires the complete session registry, session-specific bound native logs, all original batch hashes, and a top-level --native-solver-log matching one of those registered logs. It copies unchanged original log bytes, start/end records and receipts to session_logs/; session_log_evidence.json and verification_manifest.json hash-bind the inventory. Stage 5 rechecks the sealed inventory, rejects changed/missing files and verifies source ownership again.

All completion flags remain unqualified. Independent COMSOL model, material/PML/mesh review, actual source authenticity and external validation are **not** satisfied merely by consistent session logs or green CI.

Research blocker: https://github.com/smshagor-dev/CyberPhotonics-SPR/issues/42
A further research-integrity safeguard rejects byte-identical native log contents registered for different execution sessions. Each actual solver invocation must retain independently distinguishable original native log evidence; do not reuse one log merely to satisfy the expected session count.


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


## Stage 14: model reuse and solver invocation boundaries

A checkpoint campaign can opt into
[`--reuse-loaded-model`](REUSABLE_COMSOL_MODEL_SESSION.md). That
means one MPh-loaded COMSOL model per execution invocation, **not**
one model for the entire multi-session/resumed campaign. Each
interrupted/resumed execution still gets a unique session ID,
session-specific committed batch ownership and its own externally
retained genuine COMSOL-native solver log before sealing. The immutable
checkpoint contract binds model lifecycle mode; the default/legacy
per-batch behavior remains unchanged.
