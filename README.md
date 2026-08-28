# ML-Driven Inverse Design and Edge Processing for PCF-SPR Sensors

Research framework for PCF-SPR simulation, inverse design, active learning, explainability, and edge spectral processing.

## Correctness changes in v0.2

Sensitivity and FOM are computed only across analyte-RI sweeps at fixed geometry. The forward surrogate uses pitch, d/Λ, metal thickness, channel radius, and analyte RI. The inverse network produces geometry while conditioning on analyte RI. Train/validation splitting keeps RI points from the same synthetic geometry together.

Active-learning uncertainty uses dropout that is present during training, and selected candidates can be sent directly to COMSOL. Fabrication bounds cover pitch, d/Λ, metal thickness, channel radius, and overlap.

COMSOL expressions support `wavelength_scale_to_nm` and `loss_scale_to_db_per_cm` in YAML. Configure them if model expressions are not already in nm and dB/cm.

## Install
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[io,onnx,edge,comsol,xai,dev]"
```

## Synthetic validation
```powershell
python scripts/generate_synthetic_dataset.py --samples 500 --out data/processed/synthetic.parquet
```
Synthetic data validates the software pipeline; it is not a substitute for FEM evidence.

## Inverse design
```powershell
python -m sprpcf.ml.train_tandem --data data/processed/synthetic.parquet --epochs 25 --out models/tandem.pt
```
The ONNX inverse interface accepts both `target_metrics` and `analyte_ri`.

## Edge models
```powershell
python -m sprpcf.edge.train_denoiser --data data/processed/synthetic.parquet --epochs 20 --batch-size 64 --device auto --quantize
```
Outputs are `edge_denoiser.keras`, `edge_ri_predictor.keras` and their quantized TFLite counterparts. Quantized evaluation reports post-INT8 accuracy, P50/P95 denoiser latency, and model sizes.

Dataset replay (not a hardware driver):
```powershell
python -m sprpcf.edge.realtime_feed --data data/processed/synthetic.parquet --model models/edge_denoiser.keras --ri-model models/edge_ri_predictor.keras
```

Full INT8 manual export requires calibration data:
```powershell
python -m sprpcf.edge.export_tflite --model models/edge_denoiser.keras --out models/denoiser.tflite --quantization int8 --calibration-data data/processed/synthetic.parquet
```

## COMSOL
```powershell
python -m sprpcf.simulation.comsol_sweep --model path\to\pcf_spr.mph --config sweep.example.yaml --out data/raw/comsol_sweep.parquet
```
Sensitivity is `Delta(lambda_res) / Delta(n_analyte)` at identical geometry. FOM is sensitivity divided by FWHM.

## Fabrication bounds
- `0.20 <= d_over_lambda <= 0.90`
- `0.8 um <= pitch_um <= 4.0 um`
- `15 nm <= metal_thickness_nm <= 80 nm`
- `0.10 um <= channel_radius_um <= 2.0 um`
- air-hole diameter must not exceed pitch.

## Reproducibility
Training exposes deterministic seeds and checkpoints store schema/scaler metadata. GitHub Actions runs syntax, lint, packaging, and core tests on Python 3.10-3.12. For publishable claims, archive the exact COMSOL model, solver/version settings, raw FEM data, checkpoint, environment lock, and benchmark hardware with the manuscript artifacts.
