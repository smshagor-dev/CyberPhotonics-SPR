# M9 Real Validation Campaign

M9 turns the release candidate into a controlled real-evidence campaign. The software can plan, capture, qualify, package, and gate the campaign, but it never fabricates COMSOL, laboratory, or device results.

## Lifecycle

```text
campaign config
  -> init + hash-bound snapshot
  -> real COMSOL execution
  -> raw sensor acquisition
  -> exact-device benchmark
  -> qualification registry
  -> reviewer/submission packages
  -> full readiness
  -> stable v1.0 gate
```

Initialize from the tracked template:

```bash
cp configs/real_validation_campaign.example.yaml configs/real_validation_campaign.yaml
# Replace every REPLACE_* value with real paths/metadata.
python scripts/validation_campaign.py init \
  --config configs/real_validation_campaign.yaml \
  --out outputs/m9/campaign
```

Initialization writes `campaign_manifest.json`, a SHA-256-bound config snapshot, `RUNBOOK.md`, and protocol/calibration/device metadata templates. It creates no physical evidence.

## M9.1 COMSOL

The runbook uses the existing real COMSOL closed-loop command. After the run completes, register the iteration with `scripts/register_evidence.py ... comsol`. Qualification requires `backend=comsol`, `evidence_class=comsol_physics`, model/config hash agreement, simulation output, and verification output.

## M9.2 Experimental sensor

`run_hardware_pipeline.py` now archives every acquired raw frame before preprocessing with `--raw-out-jsonl`. For a laboratory claim, use a real serial acquisition, preserve the protocol and calibration record, then register the raw archive as experimental evidence. JSONL replay is useful for software tests but is not experimental evidence.

## M9.3 Exact-device benchmark

Use the actual deployment device and enable `--benchmark-iterations`. The benchmark JSON records source, runtime, OS, optional accelerator, model hashes, raw-frame hash, and the latency/throughput statistics required by the evidence registry. It remains an unqualified candidate until explicitly registered with the exact device identity.

## M9.4 Status

```bash
python scripts/validation_campaign.py status \
  --campaign outputs/m9/campaign \
  --json-out outputs/m9/campaign/status.json \
  --markdown-out outputs/m9/campaign/STATUS.md
```

Stages are `pending`, `awaiting_qualification`, or `qualified`.

## M9.5 Stable release gate

```bash
python scripts/validation_campaign.py gate \
  --campaign outputs/m9/campaign \
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
