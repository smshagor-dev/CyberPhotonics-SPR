# CyberPhotonics-SPR

A publication-oriented cyber-physical photonics research framework for **PCF-SPR simulation, physics-informed inverse design, active learning, calibrated multi-objective optimization, COMSOL closed-loop verification, real-time edge sensing, and research-grade reproducibility**.

The repository is organized around three linked pipelines:

- **A — Physics & data:** COMSOL sweeps through `mph`, explicit unit validation, fixed-geometry refractive-index sweeps, spectral metric extraction, synthetic software-validation data, and accepted-only closed-loop dataset augmentation.
- **B — AI inverse design:** a conditioned PyTorch forward surrogate, tandem inverse generator, fabrication constraints, ONNX export, deep forward ensembles, conformal uncertainty, OOD detection, Pareto candidate selection, and physics re-verification.
- **C — Edge & sensor runtime:** 1D-CNN denoising, RI/resonance inference, validated full-INT8 LiteRT deployment, wavelength/dark/reference calibration, JSONL or serial acquisition, spectral OOD, prediction intervals, and latency/memory benchmarking.

> **Evidence rule:** synthetic results validate software and methodology only. Physical performance claims require verified COMSOL and/or experimental sensor data. Runtime benchmarks for Raspberry Pi, Jetson, or other hardware must be measured on the actual device.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[edge,onnx,xai,comsol,dev]"
```

For serial sensor acquisition:

```powershell
pip install -e ".[edge,hardware]"
```

Python 3.10–3.13 is supported. COMSOL automation additionally requires a licensed COMSOL installation compatible with `mph`.

## 1. Generate scientifically valid synthetic data

Each base geometry receives a fixed RI sweep; sensitivity is never estimated across unrelated random geometries.

```powershell
python scripts/generate_synthetic_dataset.py `
  --samples 100 `
  --out data/processed/synthetic.parquet
```

`--samples 100` means 100 base geometries. With the default five-point RI sweep this produces 500 rows. Dataset writes include provenance metadata and SHA-256 hashes.

## 2. Train the conditioned tandem inverse model

```powershell
python -m sprpcf.ml.train_tandem `
  --data data/processed/synthetic.parquet `
  --epochs 25 `
  --out models/tandem.pt `
  --onnx-out models/inverse_pcf_spr.onnx `
  --seed 7
```

The forward surrogate input is:

```text
pitch_um
d_over_lambda
metal_thickness_nm
channel_radius_um
analyte_ri
```

The inverse generator receives target `[sensitivity, FOM, lambda_res]` plus analyte RI and returns the four fabrication variables. Train/validation splitting is grouped by base geometry to prevent RI-sweep leakage.

The exported ONNX inverse interface uses physical units:

```text
inputs:  target_metrics [sensitivity, FOM, lambda_res], analyte_ri
output:  geometry [pitch_um, d_over_lambda, metal_thickness_nm, channel_radius_um]
```

## 3. Fabrication constraints

Current supported envelope:

```text
0.20 <= d_over_lambda <= 0.90
0.8 um <= pitch_um <= 4.0 um
15 nm <= metal_thickness_nm <= 80 nm
0.20 um <= channel_radius_um <= 1.50 um
air-hole diameter < pitch
```

Constraints are validated at simulation boundaries, penalized during inverse training, and enforced on exported physical geometry. Validation reports retain raw pre-projection violations so projection cannot hide a weak inverse model.

## 4. Explainability

```powershell
python -m sprpcf.ml.explainability `
  --checkpoint models/tandem.pt `
  --data data/processed/synthetic.parquet `
  --out outputs/feature_attribution.csv `
  --heatmap outputs/feature_attribution.png
```

Integrated Gradients and optional SHAP operate in the same standardized five-feature space used by the trained forward surrogate, including analyte RI.

## 5. Active learning

Candidate tables contain:

```text
sensitivity_nm_per_riu
fom_per_riu
lambda_res_nm
analyte_ri
```

Select uncertain candidates:

```powershell
python -m sprpcf.ml.active_learning `
  --checkpoint models/tandem.pt `
  --candidates data/processed/candidates.csv `
  --threshold 0.05 `
  --out outputs/uncertain_candidates.csv
```

Run only selected generated geometries in COMSOL:

```powershell
python -m sprpcf.ml.active_learning `
  --checkpoint models/tandem.pt `
  --candidates data/processed/candidates.csv `
  --threshold 0.05 `
  --out outputs/uncertain_candidates.csv `
  --comsol-model path\to\pcf_spr.mph `
  --comsol-config sweep.example.yaml `
  --comsol-out data/raw/active_learning_comsol.parquet
```

## 6. COMSOL sweep

```powershell
python -m sprpcf.simulation.comsol_sweep `
  --model path\to\pcf_spr.mph `
  --config sweep.example.yaml `
  --out data/raw/comsol_sweep.parquet
```

Sensitivity is calculated only within identical geometry groups while analyte RI varies. Duplicate RI values in a fixed-geometry sweep are rejected.

Unit contract example:

```yaml
wavelength_scale_to_nm: 1.0
loss_scale_to_db_per_cm: 1.0
expected_wavelength_nm: [400.0, 1000.0]
```

If COMSOL returns wavelength in meters, use `wavelength_scale_to_nm: 1.0e9`. Implausible wavelength ranges fail fast.

## 7. Train and validate edge models

```powershell
python -m sprpcf.edge.train_denoiser `
  --data data/processed/synthetic.parquet `
  --epochs 20 `
  --batch-size 64 `
  --device auto `
  --quantize `
  --seed 7
```

Artifacts:

```text
models/edge_denoiser.keras
models/edge_ri_predictor.keras
models/edge_denoiser_quantized.tflite
models/edge_ri_predictor_quantized.tflite
```

Full-INT8 evaluation reports denoising PSNR/SSIM, RI and resonance MAE/R², float-vs-INT8 deltas, latency, and artifact sizes. The runtime uses LiteRT rather than deprecated `tf.lite.Interpreter`.

## 8. Replay the stored-spectrum stream

```powershell
python -m sprpcf.edge.realtime_feed `
  --data data/processed/synthetic.parquet `
  --model models/edge_denoiser.keras `
  --ri-model models/edge_ri_predictor.keras
```

This is explicitly a **stored-spectrum replay**, not a hardware acquisition driver.

Quantized benchmark:

```powershell
python main.py simulate-stream `
  --data data/processed/synthetic.parquet `
  --tflite-dir models `
  --duration-sec 10
```

## 9. Generate the scientific validation pack

```powershell
python scripts/run_validation_pack.py `
  --data data/processed/synthetic.parquet `
  --checkpoint models/tandem.pt `
  --out outputs/validation `
  --bootstrap-resamples 5000 `
  --mc-samples 64 `
  --seed 7
```

Physics-loss ablation:

```powershell
python scripts/run_ablation_study.py `
  --data data/processed/synthetic.parquet `
  --out outputs/ablation `
  --seeds 7,17,29 `
  --epochs 50 `
  --device cpu
```

The validation pack exports fitted fixed-geometry sensitivity/FOM, linearity, bootstrap intervals, Ridge-vs-neural baselines, inverse target satisfaction, raw/post-projection fabrication validity, uncertainty, provenance, CSV/JSON evidence, Markdown reporting, and 300-dpi plots. See `docs/SCIENTIFIC_VALIDATION.md`.

## 10. Run the physics-validated closed loop

```powershell
python scripts/run_comsol_closed_loop.py `
  --checkpoint models/tandem.pt `
  --targets data/processed/design_targets.csv `
  --base-data data/processed/training.parquet `
  --backend comsol `
  --comsol-model path\to\pcf_spr.mph `
  --comsol-config sweep.example.yaml `
  --out outputs/closed_loop/iteration_001 `
  --retrain
```

Each proposed geometry receives an odd, target-centered RI sweep. Independent sensitivity/FOM/resonance/linearity gates decide whether physics rows enter the augmented training dataset. Iteration manifests hash source data, model/config inputs, and outputs. See `docs/COMSOL_CLOSED_LOOP.md`.

## 11. Calibrated multi-objective AI design

Build a forward-surrogate ensemble:

```powershell
python -m sprpcf.ml.ensemble `
  --checkpoint models/tandem.pt `
  --data data/processed/training.parquet `
  --out models/tandem_ensemble.pt `
  --members 5 `
  --epochs 50 `
  --device auto
```

Generate and rank a latent candidate population:

```powershell
python scripts/run_multiobjective_design.py `
  --checkpoint models/tandem_ensemble.pt `
  --targets data/processed/design_targets.csv `
  --reference-data data/processed/training.parquet `
  --out outputs/multiobjective `
  --candidates-per-target 128 `
  --confidence 0.95
```

Or send Pareto-selected designs directly into the physics loop:

```powershell
python scripts/run_advanced_closed_loop.py `
  --checkpoint models/tandem_ensemble.pt `
  --targets data/processed/design_targets.csv `
  --base-data data/processed/training.parquet `
  --backend comsol `
  --comsol-model path\to\pcf_spr.mph `
  --comsol-config sweep.example.yaml `
  --out outputs/advanced_closed_loop/iteration_001 `
  --candidates-per-target 128 `
  --confidence 0.95 `
  --retrain
```

The selector combines held-out conformal residual calibration, deep-ensemble disagreement, Mahalanobis OOD scoring, raw-vs-projected fabrication distance, separate target-error objectives, and Pareto non-dominated ranking. Confidence is a ranking aid, not a physical-success probability. See `docs/ADVANCED_AI_DESIGN.md`.

## 12. Calibrated real-sensor runtime

The hardware layer accepts a device-independent newline-delimited JSON protocol over file replay or serial transport. It supports raw pixel/intensity spectrometers as well as already-calibrated wavelength/loss frames.

Create an edge calibration bundle from labeled held-out data:

```powershell
python scripts/calibrate_edge_runtime.py `
  --data data/processed/experimental.parquet `
  --denoiser models/edge_denoiser_quantized.tflite `
  --predictor models/edge_ri_predictor_quantized.tflite `
  --runtime litert `
  --coverage 0.95 `
  --ood-coverage 0.99 `
  --out models/edge_calibration.json
```

Replay a captured sensor stream:

```powershell
python scripts/run_hardware_pipeline.py `
  --source jsonl `
  --input-jsonl data/raw/sensor_capture.jsonl `
  --grid-data data/processed/training.parquet `
  --denoiser models/edge_denoiser_quantized.tflite `
  --predictor models/edge_ri_predictor_quantized.tflite `
  --calibration models/edge_calibration.json `
  --frames 100 `
  --benchmark-iterations 500
```

Run a serial sensor:

```powershell
python scripts/run_hardware_pipeline.py `
  --source serial `
  --serial-port COM5 `
  --baudrate 115200 `
  --grid-data data/processed/training.parquet `
  --denoiser models/edge_denoiser_quantized.tflite `
  --predictor models/edge_ri_predictor_quantized.tflite `
  --calibration models/edge_calibration.json `
  --frames 100
```

For pixel/intensity devices, add the wavelength polynomial plus dark/reference arrays. The runtime performs wavelength calibration, dark/reference loss conversion, non-extrapolating resampling to the exact model grid, per-frame normalization, denoising, RI/resonance inference, spectral OOD scoring, held-out prediction intervals, and physical resonance extraction.

Benchmark output includes P50/P95/P99/mean end-to-end latency, throughput, peak Python heap, and process max RSS where supported. See `docs/HARDWARE_RUNTIME.md` for the sensor protocol and measurement contract.

## 13. Research reproducibility and release engineering

Validate citation/version consistency, required research metadata, and repository release hygiene:

```powershell
python scripts/verify_release.py
```

Capture a hash-bound experiment/environment bundle for every reportable run:

```powershell
python scripts/create_reproducibility_bundle.py `
  --out outputs/repro/experiment_001 `
  --name experiment_001 `
  --seed 7 `
  --data data/processed/training.parquet `
  --checkpoint models/tandem.pt `
  --config configs/experiment.yaml
```

Each bundle records Git state, seed, configuration, exact installed Python-package snapshot, portable artifact paths, and SHA-256 hashes without copying large datasets or models. Generated files include `manifest.json`, `environment.json`, `environment.lock.txt`, `checksums.sha256`, and `REPRODUCE.md`.

The repository also includes:

- `CITATION.cff` with release-version validation.
- `MODEL_CARD.md` with model scope, intended use, and limitations.
- `DATASET_CARD.md` separating synthetic, COMSOL, and experimental evidence.
- `Dockerfile` and `.devcontainer/devcontainer.json` for a reproducible Python 3.11 CPU research environment.
- `.github/workflows/release-validation.yml` for tag-time package/evidence validation.

The release-validation workflow builds wheel/source distributions and evidence artifacts but does **not** automatically publish to PyPI or create a public GitHub release. See `docs/REPRODUCIBILITY.md` for the full release and DOI/Zenodo-ready protocol.

## Metric definitions

Wavelength sensitivity for a fixed geometry:

```text
S_lambda = Delta lambda_res / Delta n_analyte  [nm/RIU]
```

Figure of merit:

```text
FOM = abs(S_lambda) / FWHM
```

FWHM crossings are linearly interpolated instead of rounded to wavelength-grid samples.

## Reproducibility and validation

- Synthetic generation, PyTorch/TensorFlow training, ensemble training, candidate generation, calibration splits, and grouped dataset splits accept deterministic seeds where applicable.
- Dataset writes and calibration/closed-loop manifests preserve artifact provenance and SHA-256 hashes.
- Every reportable experiment can emit a portable manifest, exact package snapshot, and artifact checksum bundle.
- Release versions are checked for consistency across `pyproject.toml`, package `__version__`, and `CITATION.cff`.
- Validation splits are grouped by base geometry to prevent RI-sweep leakage.
- Scientific validation separates software/synthetic evidence from COMSOL and experimental evidence.
- Multi-objective design records Pareto rank, calibrated residual intervals, ensemble disagreement, OOD score, fabrication projection distance, and target-satisfaction ranking.
- Hardware calibration uses held-out labeled spectra; synthetic calibration must not be reported as experimental coverage.
- Hardware runtime rejects wavelength extrapolation and exposes measured resonance independently of the neural RI predictor.
- CI runs fatal Ruff checks, bytecode compilation, release-metadata validation, and tests on every push/PR; Python warnings are errors.
- CI covers Python 3.10–3.13 plus the TensorFlow/LiteRT INT8 edge/full-pipeline gate.
- Tag-time release validation builds package/evidence artifacts without auto-publishing them.

Run locally:

```powershell
ruff check .
python scripts/verify_release.py
pytest -q -W error
```

## Repository layout

```text
data/raw/          Raw COMSOL/experimental/sensor-capture data (not committed)
data/processed/    Training- and calibration-ready datasets (not committed)
models/            PyTorch/Keras/ONNX/TFLite/calibration artifacts (not committed)
outputs/           Metrics, plots, reports, candidates, closed-loop and hardware evidence
src/sprpcf/        Core Python package
scripts/           Dataset, validation, optimization, closed-loop, hardware and release runners
tests/             Scientific, ML, deployment, hardware, reproducibility and integration tests
docs/              Scientific, COMSOL, advanced-AI, hardware and reproducibility protocols
CITATION.cff       Software citation metadata
MODEL_CARD.md      Model scope, evaluation contract and limitations
DATASET_CARD.md    Dataset provenance, split policy and evidence boundaries
```
