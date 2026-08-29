# Real Validation Campaign

The Real Validation Campaign turns the release candidate into a controlled real-evidence workflow. The software can plan, preflight, capture, qualify, package, and gate the campaign, but it never fabricates COMSOL, laboratory, or device results.

## Lifecycle

```text
campaign config
  -> preflight
  -> init + hash-bound snapshot
  -> real COMSOL execution
  -> raw sensor acquisition
  -> exact-device benchmark
  -> qualification registry
  -> reviewer/submission packages
  -> full readiness
  -> stable v1.0 gate
```

Start from the tracked template:

```bash
cp configs/real_validation_campaign.example.yaml configs/real_validation_campaign.yaml
# Replace every REPLACE_* value with real paths/metadata and prepare the protocol/calibration files.
```

## Execution preflight

Before external runs, validate that the planned inputs and metadata are actually ready:

```bash
python scripts/validation_campaign.py preflight \
  --config configs/real_validation_campaign.yaml \
  --json-out outputs/real_validation/preflight.json \
  --markdown-out outputs/real_validation/PREFLIGHT.md \
  --strict
```

The preflight checks:

- COMSOL checkpoint, targets, base dataset, `.mph` model, and sweep configuration exist;
- COMSOL passes, RI span, and the odd target-centered RI sweep are valid;
- experiment instrument identity and acquisition timestamp are real, not placeholders;
- the timestamp is timezone-aware ISO-8601;
- experimental protocol and calibration records exist before acquisition;
- at least one raw-data destination is planned and writable;
- exact-device name, OS, runtime, and deployed model are present;
- LiteRT deployments use a `.tflite` model;
- registry, reviewer-package, submission-package, COMSOL, raw-data, and benchmark destinations are writable.

A passing preflight means the workflow is ready to execute externally. It is not physical evidence.

## Campaign initialization

```bash
python scripts/validation_campaign.py init \
  --config configs/real_validation_campaign.yaml \
  --out outputs/real_validation/campaign
```

Initialization writes `campaign_manifest.json`, a SHA-256-bound config snapshot, `RUNBOOK.md`, and protocol/calibration/device metadata templates. It creates no physical evidence.

## COMSOL validation

The runbook uses the existing real COMSOL closed-loop command. After the run completes, register the iteration with `scripts/register_evidence.py ... comsol`. Qualification requires `backend=comsol`, `evidence_class=comsol_physics`, model/config hash agreement, simulation output, and verification output.

## Experimental sensor validation

`run_hardware_pipeline.py` archives every acquired raw frame before preprocessing with `--raw-out-jsonl`. For a laboratory claim, use a real serial acquisition, preserve the protocol and calibration record, then register the raw archive as experimental evidence. JSONL replay remains useful for software tests but is not experimental evidence.

## Exact-device benchmark

Use the actual deployment device and enable `--benchmark-iterations`. The benchmark JSON records source, runtime, OS, optional accelerator, model hashes, raw-frame hash, and the latency/throughput statistics required by the evidence registry. It remains an unqualified candidate until explicitly registered with the exact device identity.

## Campaign status

```bash
python scripts/validation_campaign.py status \
  --campaign outputs/real_validation/campaign \
  --json-out outputs/real_validation/campaign/status.json \
  --markdown-out outputs/real_validation/campaign/STATUS.md
```

Stages are `pending`, `awaiting_qualification`, or `qualified`.

## Stable release gate

```bash
python scripts/validation_campaign.py gate \
  --campaign outputs/real_validation/campaign \
  --expected-version 1.0.0 \
  --strict
```

The stable gate requires:

- valid campaign snapshot hash;
- qualified registry containing `comsol_physics`, `experimental_sensor`, and `device_benchmark`;
- full whole-system readiness;
- project/package/citation version exactly matching the requested stable version;
- a non-prerelease package version.

Reviewer or submission JSON flags alone cannot satisfy physical readiness. The qualified evidence registry is the only physical truth source for the gate.
