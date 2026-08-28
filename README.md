# CyberPhotonics-SPR

Research framework for photonic crystal fiber surface plasmon resonance (PCF-SPR) simulation, conditioned ML inverse design, active learning, edge spectral inference, scientific validation, COMSOL closed-loop verification, and calibrated multi-objective design.

The framework has three linked pipelines:

- **A — Physics/data:** COMSOL sweeps through `mph`, spectral metric extraction, explicit unit validation, synthetic fixed-geometry RI sweeps when COMSOL is unavailable, and closed-loop dataset augmentation from accepted physics runs.
- **B — ML inverse design:** PyTorch forward surrogate conditioned on sensor geometry **and analyte RI**, a tandem inverse generator with fabrication penalties, bounded ONNX output, optional forward ensembles, conformal uncertainty, OOD detection, and Pareto candidate selection.
- **C — Edge inference:** 1D-CNN denoising and RI/resonance prediction with validated full-INT8 TFLite/LiteRT export and latency/accuracy reporting.

Synthetic data is for pipeline validation only. Physical research conclusions should be based on verified COMSOL/experimental data.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[edge,onnx,xai,comsol,dev]"
```

Python 3.10–3.13 is supported by the package metadata. COMSOL automation additionally requires a licensed COMSOL installation compatible with `mph`.

## 1. Generate scientifically valid synthetic data

Synthetic sensitivity is no longer estimated across unrelated random geometries. Each base geometry receives a fixed RI sweep, and sensitivity is calculated only inside that geometry group.

```powershell
python scripts/generate_synthetic_dataset.py --samples 100 --out data/processed/synthetic.parquet
```

`--samples` means **base geometries**. The default five-point RI sweep produces 500 rows. A provenance sidecar is written beside every dataset with row/column metadata and a SHA-256 content hash.

## 2. Train conditioned tandem inverse design

```powershell
python -m sprpcf.ml.train_tandem `
  --data data/processed/synthetic.parquet `
  --epochs 25 `
  --out models/tandem.pt `
  --onnx-out models/inverse_pcf_spr.onnx `
  --seed 7
```

The forward surrogate uses:

```text
pitch_um
d_over_lambda
metal_thickness_nm
channel_radius_um
analyte_ri
```

The inverse model receives target sensing metrics plus analyte RI and generates the four fabrication-design variables. Train/validation splitting is grouped by base geometry, so different RI points from one geometry cannot leak into both splits.

The exported inverse ONNX interface uses physical units:

```text
inputs:  target_metrics [sensitivity, FOM, lambda_res], analyte_ri
output:  geometry [pitch_um, d_over_lambda, metal_thickness_nm, channel_radius_um]
```

Output geometry is projected to the supported fabrication envelope.

## 3. Fabrication constraints

The current supported envelope is:

```text
0.20 <= d_over_lambda <= 0.90
0.8 um <= pitch_um <= 4.0 um
15 nm <= metal_thickness_nm <= 80 nm
0.20 um <= channel_radius_um <= 1.50 um
air-hole diameter < pitch
```

These constraints are validated for COMSOL inputs, penalized during inverse training, and enforced on physical ONNX output.

## 4. Explainability

```powershell
python -m sprpcf.ml.explainability `
  --checkpoint models/tandem.pt `
  --data data/processed/synthetic.parquet `
  --out outputs/feature_attribution.csv `
  --heatmap outputs/feature_attribution.png
```

Integrated Gradients and optional SHAP operate in the same standardized five-feature input space used to train the forward model. Analyte RI is included as a condition feature instead of being silently omitted.

## 5. Active learning

Candidate files must contain:

```text
sensitivity_nm_per_riu
fom_per_riu
lambda_res_nm
analyte_ri
```

Select uncertain candidates with trained MC dropout:

```powershell
python -m sprpcf.ml.active_learning `
  --checkpoint models/tandem.pt `
  --candidates data/processed/candidates.csv `
  --threshold 0.05 `
  --out outputs/uncertain_candidates.csv
```

To close the loop and run **only the selected generated geometries** in COMSOL:

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

Sensitivity is calculated only for rows sharing identical geometry while analyte RI changes. Duplicate RI values inside one fixed-geometry sweep are rejected rather than producing invalid gradients.

The YAML file includes an explicit unit contract:

```yaml
wavelength_scale_to_nm: 1.0
loss_scale_to_db_per_cm: 1.0
expected_wavelength_nm: [400.0, 1000.0]
```

If COMSOL returns wavelength in meters, set `wavelength_scale_to_nm: 1.0e9`. Implausible wavelength ranges fail fast instead of silently contaminating research results.

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

When `--quantize` is enabled, the framework evaluates the actual INT8 models and reports denoising PSNR/SSIM, RI and resonance MAE/R², float-vs-INT8 error deltas, P50/P95 latency, and model sizes.

Standalone full-INT8 export requires calibration data and will fail if it is omitted:

```powershell
python -m sprpcf.edge.export_tflite `
  --model models/edge_denoiser.keras `
  --out models/edge_denoiser_quantized.tflite `
  --quantization int8 `
  --calibration-data data/processed/synthetic.parquet
```

## 8. Replay a simulated sensor stream

```powershell
python -m sprpcf.edge.realtime_feed `
  --data data/processed/synthetic.parquet `
  --model models/edge_denoiser.keras `
  --ri-model models/edge_ri_predictor.keras
```

This command is explicitly a **stored-spectrum sensor replay**, not a hardware acquisition driver. Each noisy measured spectrum is normalized from its own observed statistics before inference.

Quantized benchmark:

```powershell
python main.py simulate-stream `
  --data data/processed/synthetic.parquet `
  --tflite-dir models `
  --duration-sec 10
```

## 9. Generate the scientific validation pack

The research validation layer independently fits each fixed-geometry RI sweep, reports bootstrap confidence intervals, compares the neural forward surrogate with a leakage-resistant Ridge baseline, measures inverse target satisfaction, separates raw versus post-projection fabrication validity, and records MC-dropout uncertainty plus dataset/checkpoint SHA-256 provenance.

```powershell
python scripts/run_validation_pack.py `
  --data data/processed/synthetic.parquet `
  --checkpoint models/tandem.pt `
  --out outputs/validation `
  --bootstrap-resamples 5000 `
  --mc-samples 64 `
  --seed 7
```

For the physics-loss ablation across deterministic seeds:

```powershell
python scripts/run_ablation_study.py `
  --data data/processed/synthetic.parquet `
  --out outputs/ablation `
  --seeds 7,17,29 `
  --epochs 50 `
  --device cpu
```

The validation pack exports CSV/JSON evidence, provenance, a Markdown report, and 300-dpi publication plots. See `docs/SCIENTIFIC_VALIDATION.md` for the full evidence protocol.

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

Each generated geometry is evaluated over an odd target-centered fixed-geometry RI sweep. Sensitivity/FOM/resonance/linearity gates decide whether physics rows are accepted into the augmented training dataset. The iteration manifest hashes the source dataset, checkpoint, COMSOL model/config, and generated artifacts.

## 11. Calibrated multi-objective AI design

Upgrade an existing tandem checkpoint with independently initialized forward-surrogate ensemble members:

```powershell
python -m sprpcf.ml.ensemble `
  --checkpoint models/tandem.pt `
  --data data/processed/training.parquet `
  --out models/tandem_ensemble.pt `
  --members 5 `
  --epochs 50 `
  --device auto
```

Generate a latent candidate pool for each target and rank it by separate sensitivity, FOM, resonance, manufacturability and OOD objectives:

```powershell
python scripts/run_multiobjective_design.py `
  --checkpoint models/tandem_ensemble.pt `
  --targets data/processed/design_targets.csv `
  --reference-data data/processed/training.parquet `
  --out outputs/multiobjective `
  --candidates-per-target 128 `
  --confidence 0.95
```

Or run the Pareto-selected designs directly through the Phase-2 physics loop:

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

The advanced selector combines held-out conformal residual calibration, optional deep-ensemble disagreement, Mahalanobis OOD scoring, raw-vs-projected fabrication distance, and Pareto non-dominated ranking. Its confidence score is a ranking aid—not a probability of physical success—and never bypasses COMSOL acceptance gates. See `docs/ADVANCED_AI_DESIGN.md`.

## Metric definitions

Wavelength sensitivity for a fixed geometry is:

```text
S_lambda = Delta lambda_res / Delta n_analyte  [nm/RIU]
```

Figure of merit is calculated from sensitivity magnitude:

```text
FOM = abs(S_lambda) / FWHM
```

FWHM crossings are linearly interpolated rather than rounded to wavelength-grid samples.

## Reproducibility and validation

- Synthetic generation, PyTorch training, TensorFlow training, ensemble training, candidate generation, and grouped splits accept deterministic seeds.
- Dataset writes produce `.meta.json` provenance sidecars with SHA-256 hashes.
- Validation splits are grouped by base geometry to prevent RI-sweep leakage.
- Scientific validation reports fitted fixed-geometry sensitivity/FOM, bootstrap CIs, model baselines, target satisfaction, raw/post-projection constraint rates, uncertainty, and artifact hashes.
- Advanced design records Pareto rank, calibrated residual intervals, ensemble disagreement, OOD score, fabrication projection distance, and confidence ranking for every candidate.
- CI runs fatal Ruff checks, bytecode compilation, and the test suite on every push/PR; Python warnings are treated as errors.
- Tests cover grouped sensitivity, duplicate-RI rejection, COMSOL unit validation, fabrication bounds, conditioned XAI, active-learning handoff, ONNX interface, group leakage, INT8 deployment, scientific validation, COMSOL closed-loop validation, and multi-objective design.

Run locally:

```powershell
ruff check .
pytest -q
```

## Repository layout

```text
data/raw/          Raw COMSOL/experimental data (not committed)
data/processed/    Training-ready datasets (not committed)
models/            Trained PyTorch/Keras/ONNX/TFLite artifacts (not committed)
outputs/           Metrics, plots, reports, candidates, and closed-loop artifacts
src/sprpcf/        Python package
scripts/           Dataset, validation, optimization, and closed-loop runners
tests/             Scientific, ML, deployment, and integration tests
```
