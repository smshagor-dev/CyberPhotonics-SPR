# ML-Driven Inverse Design and Edge Processing for PCF-SPR Sensors

This repository implements an end-to-end research framework for photonic crystal fiber surface plasmon resonance (PCF-SPR) sensor design:

- Pipeline A: COMSOL-driven simulation sweeps and spectral metric extraction.
- Pipeline B: PyTorch tandem neural network for inverse design under fabrication constraints.
- Pipeline C: lightweight 1D-CNN autoencoder for spectral denoising and real-time refractive-index prediction.

The code is modular so it can run with real COMSOL models through `mph`, or with synthetic spectra for model and pipeline validation when COMSOL is unavailable.

## Quick Start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .
```

Generate a small synthetic dataset:

```powershell
python scripts/generate_synthetic_dataset.py --samples 500 --out data/processed/synthetic.parquet
```

Train the forward and inverse tandem model:

```powershell
python -m sprpcf.ml.train_tandem --data data/processed/synthetic.parquet --epochs 25 --out models/tandem.pt
```

Train the edge denoising model:

```powershell
python -m sprpcf.edge.train_denoiser --data data/processed/synthetic.parquet --epochs 20 --batch-size 64 --device auto --quantize
```

This writes `models/edge_denoiser_quantized.tflite` and `models/edge_ri_predictor_quantized.tflite` when `--quantize` is enabled.

Simulate a real-time noisy sensor stream:

```powershell
python -m sprpcf.edge.realtime_feed --data data/processed/synthetic.parquet --model models/denoiser.keras --ri-model models/ri_predictor.keras
```

Run active-learning candidate acquisition:

```powershell
python -m sprpcf.ml.active_learning --help
```

Generate feature attribution matrices:

```powershell
python -m sprpcf.ml.explainability --checkpoint models/tandem.pt --data data/processed/synthetic.parquet --out outputs/feature_attribution.csv --heatmap outputs/feature_attribution.png
```

The explainability engine supports built-in Integrated Gradients and optional SHAP if `shap` is installed.

Run a COMSOL sweep when COMSOL Multiphysics and a compatible `.mph` model are available:

```powershell
python -m sprpcf.simulation.comsol_sweep --model path\to\pcf_spr.mph --config sweep.yaml --out data/raw/comsol_sweep.parquet
```

## Physical Model Notes

The SPR resonance wavelength is estimated from the confinement-loss peak. Sensitivity is computed as:

```text
S_lambda = Delta lambda_res / Delta n_analyte  [nm/RIU]
```

The figure of merit is:

```text
FOM = S_lambda / FWHM
```

The inverse-design loss constrains geometry to fabrication-safe regions:

- `0.20 <= d_over_lambda <= 0.90`
- `0.8 um <= pitch_um <= 4.0 um`
- `15 nm <= metal_thickness_nm <= 80 nm`
- adjacent air holes do not overlap in the simplified lattice constraint `d < pitch`.

These bounds can be adjusted in `src/sprpcf/ml/losses.py` and the sweep configuration.

## Suggested Project Layout

```text
data/raw/          Raw COMSOL outputs
data/processed/    Training-ready parquet/csv datasets
models/            Trained PyTorch and TensorFlow artifacts
outputs/           Plots, reports, and exported edge models
src/sprpcf/         Python package
```
