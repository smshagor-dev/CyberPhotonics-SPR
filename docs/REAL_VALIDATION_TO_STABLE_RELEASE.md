# Real Validation to Stable Release

This workflow completes the research system from real numerical physics through laboratory evidence, target-device benchmarking, manuscript Results refresh, and stable release preparation.

The software never fabricates COMSOL, experimental, or exact-device evidence. A missing physical artifact remains a blocker.

## Work sequence

### Real COMSOL Validation

Prepare the real campaign config and run preflight first:

```bash
python scripts/validation_campaign.py preflight \
  --config configs/real_validation_campaign.yaml \
  --strict
```

Initialize the hash-bound campaign:

```bash
python scripts/validation_campaign.py init \
  --config configs/real_validation_campaign.yaml \
  --out outputs/real_validation/campaign
```

Run the configured real COMSOL closed loop on a machine with COMSOL + the `comsol` extra:

```bash
python scripts/research_completion.py run-comsol \
  --campaign outputs/real_validation/campaign
```

The resulting iteration must contain a COMSOL-backed `iteration_manifest.json`, `simulation_results.csv`, and `verification.csv`. Synthetic or replay outputs cannot qualify as COMSOL physics.

### Experimental Sensor Validation

Preserve raw acquisition files, protocol, calibration, exact instrument ID, and timezone-aware acquisition time. Register raw evidence only after those artifacts are complete.

Create a measurement-analysis manifest based on:

```text
configs/experimental_measurements.example.yaml
```

Every calibrated spectrum entry must include both:

- `raw_path` — the measured source artifact;
- `path` — the calibrated `wavelength_nm,loss_db_per_cm` CSV/Parquet table.

Analyze the measured spectra:

```bash
python scripts/analyze_experimental_results.py \
  --manifest outputs/real_validation/experiment/experimental_measurements.yaml \
  --out outputs/real_validation/experiment/analysis
```

The analysis produces replicate resonance/FWHM metrics, RI means and standard deviations, OLS sensitivity/R², FOM, repeatability summaries, and 300-DPI plots. These derived values are not treated as qualified experimental evidence until their raw-file hashes are present in the qualified evidence registry.

### Exact-Device Benchmark

Run `scripts/run_hardware_pipeline.py` on the exact deployment device with the final LiteRT/TFLite model. Preserve raw frames and generate the benchmark JSON with exact device, OS, runtime, accelerator, model hashes, latency percentiles, throughput, and memory metrics.

The benchmark remains an `unqualified_candidate` until registered by:

```bash
python scripts/register_evidence.py \
  --registry outputs/real_validation/evidence/evidence_registry.json \
  device \
  --benchmark outputs/real_validation/device/benchmark.json \
  --model models/denoiser_int8.tflite \
  --device-name "EXACT DEVICE" \
  --os-name "EXACT OS" \
  --runtime LiteRT
```

### Evidence-Aware Finalization

When real artifacts exist, the completion controller can qualify every stage that satisfies the strict contracts:

```bash
python scripts/research_completion.py qualify-ready \
  --campaign outputs/real_validation/campaign
```

Then rebuild reviewer/submission evidence and the stable-release blocker view:

```bash
python scripts/research_completion.py refresh \
  --campaign outputs/real_validation/campaign
```

Or run both software-only advancement steps together:

```bash
python scripts/research_completion.py advance \
  --campaign outputs/real_validation/campaign
```

`advance` does not run laboratory acquisition or invent missing external evidence.

### Paper Results Finalization

The evidence-backed Results package is built from the validated registry plus raw-hash-linked experimental analysis:

```bash
python scripts/build_paper_results.py \
  --evidence-registry outputs/real_validation/evidence/evidence_registry.json \
  --experimental-analysis-dir outputs/real_validation/experiment/analysis \
  --out outputs/real_validation/campaign/paper_results \
  --strict
```

It generates:

- `RESULTS_EVIDENCE_SUMMARY.md`;
- `TABLE_COMSOL_EVIDENCE.csv`;
- `TABLE_EXPERIMENTAL_RESULTS.csv`;
- `TABLE_DEVICE_BENCHMARK.csv`;
- copied traceable COMSOL/device source evidence;
- bound experimental tables/figures;
- `PAPER_RESULTS_MANIFEST.json`;
- `checksums.sha256`.

Experimental numerical values are withheld unless the analysis raw hashes are a subset of the qualified experimental raw-data hashes.

### Stable Release

While the project is still a release candidate, build a promotion plan:

```bash
python scripts/prepare_stable_release.py \
  --finalization-dir outputs/real_validation/campaign/finalization \
  --paper-results-dir outputs/real_validation/campaign/paper_results \
  --target-version 1.0.0 \
  --json-out outputs/real_validation/stable-release-plan.json \
  --markdown-out outputs/real_validation/stable-release-plan.md \
  --strict
```

The promotion gate accepts an evidence-complete release candidate only when the finalization package has full physical readiness and its sole remaining finalization blocker is the prerelease version itself.

Only after that gate passes may version metadata be updated:

```bash
python scripts/prepare_stable_release.py \
  --finalization-dir outputs/real_validation/campaign/finalization \
  --paper-results-dir outputs/real_validation/campaign/paper_results \
  --target-version 1.0.0 \
  --apply \
  --strict
```

`--apply` updates package/CITATION version metadata and writes `STABLE_RELEASE_EVIDENCE.json`. It does not create a Git tag, GitHub Release, DOI, or journal submission.

After promotion, rebuild finalization/results at the stable version, run the full test/release suite, commit the certificate and version changes, and only then create `v1.0.0`. Stable tags are rejected by the release-validation workflow if the promotion certificate is missing or mismatched.

## One status command

At any time:

```bash
python scripts/research_completion.py status \
  --campaign outputs/real_validation/campaign \
  --json-out outputs/real_validation/completion-status.json \
  --markdown-out outputs/real_validation/completion-status.md
```

The report uses the work names above and shows the next command for every incomplete stage.

## Scientific boundary

Packaging integrity and automation are not substitutes for scientific validation. COMSOL setup, mesh/boundary conditions, laboratory protocol, calibration, analyte preparation, fabrication, raw measurements, and exact-device conditions remain subject to scientific inspection and peer review.
