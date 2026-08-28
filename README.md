# CyberPhotonics-SPR

[![CI](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/workflows/ci.yml/badge.svg)](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

**CyberPhotonics-SPR** is a research and engineering framework for photonic crystal fiber surface plasmon resonance (PCF-SPR) sensor design. It combines numerical photonics, physics-informed inverse design, uncertainty-aware multi-objective optimization, COMSOL verification, edge inference, hardware-facing spectrum processing, and reproducible scientific reporting in one codebase.

The repository is written for two audiences at the same time:

- **Researchers** who need traceable equations, scientifically valid refractive-index sweeps, independent validation metrics, uncertainty estimates, reproducible experiments, and a strict separation between synthetic, simulated, and experimental evidence.
- **Developers** who need a maintainable Python package, deterministic command-line workflows, tested scientific utilities, ONNX/LiteRT deployment, CI, modular interfaces, and machine-readable artifacts.

The mathematical calculation path is part of the software contract. Resonance wavelength, FWHM, sensitivity, FOM, linearity, fabrication penalties, uncertainty intervals, OOD scores, and candidate ranking are calculated explicitly rather than hidden behind dashboard summaries.

> **Evidence policy**  
> Synthetic data validate software and methodology. COMSOL results are numerical-physics evidence only when produced from a verified model and configuration. Experimental performance claims require measured sensor data. Surrogate predictions are never presented as experimental measurements.

---

## 1. Research problem

A PCF-SPR sensor is governed by the coupled relationship between geometry, material dispersion, analyte refractive index, and wavelength-dependent plasmonic loss.

The forward problem is written as:

```math
\mathbf{m}=f(\mathbf{g},n_a)
```

where the physical geometry is

```math
\mathbf{g}=\left[\Lambda,\frac{d}{\Lambda},t_{Au},r_c\right]
```

and the sensing metrics are

```math
\mathbf{m}=\left[S_\lambda,\mathrm{FOM},\lambda_{res}\right].
```

The inverse problem is more difficult because several geometries can produce similar optical responses. The practical task is therefore not simply to return one geometry. A useful inverse-design system must generate fabrication-valid candidates, quantify model uncertainty, avoid unsupported extrapolation, and re-check the selected design with numerical physics.

The implemented research flow is:

```text
Design target
    ↓
Conditioned inverse generator
    ↓
Candidate population
    ↓
Fabrication projection + uncertainty + OOD analysis
    ↓
Pareto selection
    ↓
Exact selected geometry
    ↓
Fixed-geometry RI sweep
    ↓
COMSOL or software-validation spectrum
    ↓
Independent resonance / FWHM / sensitivity / FOM / R² calculation
    ↓
Accept or reject
    ↓
Dataset augmentation / retraining / edge deployment / evidence report
```

A selected dashboard or Pareto candidate keeps the same geometry identity when it is passed to physics verification. The interface does not display one geometry and silently simulate another.

---

## 2. Scientific variables and units

| Symbol | Repository field | Meaning | Unit |
|---|---|---|---|
| `Λ` | `pitch_um` | PCF pitch | µm |
| `d/Λ` | `d_over_lambda` | normalized air-hole diameter | dimensionless |
| `d` | derived | air-hole diameter | µm |
| `t_Au` | `metal_thickness_nm` | gold layer thickness | nm |
| `r_c` | `channel_radius_um` | sensing-channel radius | µm |
| `n_a` | `analyte_ri` | analyte refractive index | RIU |
| `λ_res` | `lambda_res_nm` | resonance wavelength | nm |
| `FWHM` | `fwhm_nm` | full width at half maximum | nm |
| `S_λ` | `sensitivity_nm_per_riu` | wavelength sensitivity | nm/RIU |
| `FOM` | `fom_per_riu` | figure of merit | RIU⁻¹ |

Current fabrication envelope:

```math
0.8 \le \Lambda \le 4.0\;\mu\mathrm{m}
```

```math
0.20 \le \frac{d}{\Lambda} \le 0.90
```

```math
15 \le t_{Au} \le 80\;\mathrm{nm}
```

```math
0.20 \le r_c \le 1.50\;\mu\mathrm{m}
```

The air-hole diameter is

```math
d=\Lambda\left(\frac{d}{\Lambda}\right)
```

and the non-overlap condition is

```math
d<\Lambda.
```

These limits are checked in physical units at simulation boundaries and are also used by differentiable fabrication penalties during inverse-model training.

---

# 3. Mathematical and computational formulation

This section describes the equations used by the implementation. The equations are intentionally tied to the code path so that a reported number can be traced back to a calculation, configuration, dataset, and commit.

## 3.1 Silica dispersion: Sellmeier equation

The fused-silica refractive index is calculated with a three-term Sellmeier relation:

```math
n_{\mathrm{SiO_2}}^2(\lambda)
=
1+
\sum_{j=1}^{3}
\frac{B_j\lambda^2}{\lambda^2-C_j}.
```

The implementation uses

```math
B=[0.6961663,\;0.4079426,\;0.8974794]
```

and

```math
C=[0.0684043^2,\;0.1162414^2,\;9.896161^2]\;\mu\mathrm{m}^2.
```

Wavelength is expressed in micrometres in the dispersion functions. NumPy and differentiable PyTorch implementations use the same coefficients.

## 3.2 Gold dispersion: Drude-Lorentz model

Photon energy is obtained from

```math
E=\frac{hc}{\lambda},
```

with

```math
hc=1.239841984\;\mathrm{eV\,\mu m}.
```

The compact gold model used for physical regularization follows

```math
\varepsilon_{Au}(E)
=
\varepsilon_{\infty}
-
\frac{f_D\omega_p^2}{E(E+i\gamma_D)}
+
\sum_j
\frac{f_j\omega_p^2}{E_j^2-E^2-i\gamma_jE}.
```

This model is a regularization aid. It is not claimed to replace measured optical constants in a high-fidelity electromagnetic study.

## 3.3 Resonance wavelength

Given wavelengths `λ_i` and confinement-loss values `L_i`, the strongest represented peak is identified first. A three-point quadratic fit is then used around the discrete maximum to reduce wavelength-grid bias.

For local coordinates `x = λ - λ_i`:

```math
L(x)=ax^2+bx+c.
```

When the fitted parabola opens downward and its vertex remains inside the three-sample interval, the sub-grid resonance offset is

```math
x^*=-\frac{b}{2a}.
```

The refined resonance wavelength is therefore

```math
\lambda_{res}=\lambda_i+x^*.
```

The refinement stays inside the neighboring samples; it does not extrapolate a resonance outside the available local spectrum.

## 3.4 Full width at half maximum

The minimum loss in the spectrum is used as the local baseline:

```math
L_{base}=\min_i L_i.
```

The half-height level is

```math
L_{1/2}
=
L_{base}
+
\frac{L_{peak}-L_{base}}{2}.
```

For two neighboring samples `(λ_0, L_0)` and `(λ_1, L_1)` that bracket a half-height crossing, linear interpolation gives

```math
\lambda_{cross}
=
\lambda_0
+
\frac{L_{1/2}-L_0}{L_1-L_0}
(\lambda_1-\lambda_0).
```

After the left and right crossings are found,

```math
\mathrm{FWHM}
=
\lambda_{right}-\lambda_{left}.
```

Interpolating the crossings avoids rounding FWHM to the wavelength-grid spacing.

## 3.5 Wavelength sensitivity

Sensitivity is calculated only while the physical geometry is fixed and analyte refractive index changes. Calculating a derivative across unrelated random geometries is scientifically invalid and is explicitly avoided.

For neighboring RI points:

```math
S_\lambda
\approx
\frac{\Delta\lambda_{res}}{\Delta n_a}
\quad [\mathrm{nm/RIU}].
```

The dataset-generation path uses finite-difference gradients inside each fixed-geometry RI sweep.

For independent validation, a straight line is fitted across the complete RI sweep:

```math
\lambda_{res}=S_{fit}n_a+b.
```

The least-squares slope is

```math
S_{fit}
=
\frac{
\sum_i(n_i-\bar n)(\lambda_i-\bar\lambda)
}{
\sum_i(n_i-\bar n)^2
}.
```

This fitted sensitivity is intentionally independent from the local finite-difference labels used during learning.

## 3.6 RI linearity

The RI-to-resonance linearity is reported with the coefficient of determination:

```math
R^2
=
1-
\frac{
\sum_i(\lambda_i-\hat\lambda_i)^2
}{
\sum_i(\lambda_i-\bar\lambda)^2
}.
```

Closed-loop verification can reject a design when the RI response falls below a configured `R²` threshold, even if individual target errors appear acceptable.

## 3.7 Figure of merit

The wavelength-domain figure of merit is

```math
\mathrm{FOM}
=
\frac{|S_\lambda|}{\mathrm{FWHM}}
\quad [\mathrm{RIU}^{-1}].
```

High sensitivity alone is therefore not enough. A broad resonance lowers FOM.

## 3.8 Worked calculation example

The values below are **illustrative only** and are not reported project results.

Assume one fixed geometry is evaluated at three analyte refractive indices:

| `n_a` | `λ_res` |
|---:|---:|
| 1.33 | 620 nm |
| 1.35 | 636 nm |
| 1.37 | 652 nm |

Using the first two points:

```math
S_\lambda
=
\frac{636-620}{1.35-1.33}
=
\frac{16}{0.02}
=800\;\mathrm{nm/RIU}.
```

Because all three points are exactly linear in this example:

```math
S_{fit}=800\;\mathrm{nm/RIU},
\qquad
R^2=1.0.
```

If the local resonance baseline is `2 dB/cm` and the peak is `10 dB/cm`, then

```math
L_{1/2}
=2+\frac{10-2}{2}
=6\;\mathrm{dB/cm}.
```

If interpolation places the half-height crossings at `616 nm` and `656 nm`:

```math
\mathrm{FWHM}=656-616=40\;\mathrm{nm}.
```

The resulting FOM is

```math
\mathrm{FOM}
=
\frac{800}{40}
=20\;\mathrm{RIU}^{-1}.
```

The validation chain is therefore:

```text
spectrum
→ resonance wavelength
→ half-height crossings
→ FWHM
→ fixed-geometry RI sensitivity
→ FOM
→ RI linearity
→ acceptance or rejection
```

---

# 4. Physics-informed tandem inverse design

## 4.1 Standardization

Geometry, analyte condition, and target metrics are standardized independently:

```math
\tilde{x}=\frac{x-\mu_x}{\sigma_x}.
```

Scaler means and scales are stored in the checkpoint and reused by the physical-unit ONNX export path.

The conditioned forward surrogate receives:

```text
pitch_um
d_over_lambda
metal_thickness_nm
channel_radius_um
analyte_ri
```

and predicts:

```text
sensitivity_nm_per_riu
fom_per_riu
lambda_res_nm
```

The inverse generator receives desired sensing metrics, analyte RI, and a latent vector and proposes the four geometry variables.

## 4.2 Forward-model loss

The forward model is trained in standardized metric space with mean-squared error:

```math
\mathcal{L}_{forward}
=
\frac{1}{N}
\sum_{i=1}^{N}
\left\|
f(\tilde{\mathbf{g}}_i,\tilde n_i)-\tilde{\mathbf{m}}_i
\right\|_2^2.
```

## 4.3 Tandem inverse loss

The inverse generator is optimized through a frozen forward surrogate. In simplified form:

```math
\mathcal{L}_{inverse}
=
\mathcal{L}_{target}
+
\alpha\mathcal{L}_{overlap}
+
\beta\mathcal{L}_{bounds}
+
\gamma\mathcal{L}_{dispersion}.
```

Target satisfaction is

```math
\mathcal{L}_{target}
=
\mathrm{MSE}\left(
f(g(\mathbf{m}^*,n_a,\mathbf{z}),n_a),
\mathbf{m}^*
\right).
```

For air-hole overlap:

```math
\mathcal{L}_{overlap}
=
\operatorname{mean}\left[
\operatorname{ReLU}\left(
\frac{d-\Lambda}{\max(\Lambda,\epsilon)}
\right)^2
\right].
```

For any physical variable `x` with fabrication limits `l` and `u`, the normalized boundary penalty is

```math
\mathcal{L}_{bound}(x)
=
\operatorname{mean}\left[
\operatorname{ReLU}\left(\frac{l-x}{u-l}\right)^2
+
\operatorname{ReLU}\left(\frac{x-u}{u-l}\right)^2
\right].
```

The total boundary penalty is the sum over pitch, normalized air-hole diameter, metal thickness, and channel radius.

Raw inverse outputs and projected outputs are evaluated separately. Projection is not allowed to hide poor unconstrained generator behavior in the validation report.

---

# 5. Uncertainty, calibration, OOD detection, and Pareto design

The inverse-design problem is one-to-many, so a single predicted geometry is not treated as sufficient evidence. The advanced design path combines candidate generation, forward ensembles, held-out residual calibration, OOD analysis, fabrication projection distance, and Pareto ranking.

## 5.1 Ensemble prediction

For `K` forward-surrogate members:

```math
\bar{\mathbf{m}}
=
\frac{1}{K}
\sum_{k=1}^{K}\mathbf{m}_k.
```

The sample standard deviation across ensemble members is retained as a model-disagreement signal.

## 5.2 Held-out residual calibration

For calibration row `i` and metric `j`, the absolute residual is

```math
r_{ij}=|\hat m_{ij}-m_{ij}|.
```

A finite-sample conformal quantile is estimated independently for each metric. Candidate intervals combine the calibrated residual radius with ensemble disagreement. These intervals describe held-out model error behavior; they do not replace physics verification.

## 5.3 Mahalanobis OOD score

Let standardized model input `x` have training center `μ` and regularized covariance matrix `Σ`. The Mahalanobis distance is

```math
D_M(\mathbf{x})
=
\sqrt{
(\mathbf{x}-\boldsymbol{\mu})^T
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu})
}.
```

The normalized OOD score is

```math
\mathrm{OOD}(\mathbf{x})
=
\frac{D_M(\mathbf{x})}{D_{threshold}}.
```

Interpretation:

```text
OOD <= 1.0  → inside the calibrated reference threshold
OOD > 1.0   → outside the calibrated reference threshold
```

This is a model-domain diagnostic, not proof of physical validity.

## 5.4 Pareto objectives and ranking

For target metric `j`, normalized error is

```math
e_j
=
\frac{|\hat m_j-m_j^*|}{s_j},
```

where `s_j` is the stored physical metric scale.

Pareto ranking minimizes the separate target errors together with fabrication projection distance and OOD score. Candidate `A` dominates candidate `B` only when `A` is no worse in every objective and strictly better in at least one objective.

The practical composite ranking score used by the implementation is

```math
C
=
\operatorname{mean}(e_j)
+0.25p
+0.20\max(o-1,0)
+0.10(1-c),
```

where:

- `p` is normalized fabrication projection distance,
- `o` is normalized OOD score,
- `c` is target-interval coverage fraction.

The reported confidence-ranking value is

```math
R
=
\frac{
\exp[-\operatorname{clip}(C,0,50)]
}{
1+\max(o-1,0)
}.
```

`R` is a ranking aid. It is **not** a calibrated probability that a fabricated sensor will succeed.

---

# 6. Scientifically valid dataset construction

The most important data rule is:

> **Sensitivity is never calculated across unrelated random geometries.**

Each base geometry is evaluated at multiple analyte RI values. The geometry remains fixed while `n_a` changes. The same physical geometry group is also kept together during train/validation splitting to prevent RI-sweep leakage.

Generate synthetic software-validation data:

```powershell
python scripts/generate_synthetic_dataset.py `
  --samples 100 `
  --out data/processed/synthetic.parquet
```

With the default five-point RI sweep, 100 base geometries generate 500 spectrum rows.

Synthetic spectra are suitable for software tests, metric-extraction checks, model training, deployment validation, and orchestration tests. They are not experimental PCF-SPR measurements.

---

# 7. Installation

## 7.1 Base environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Supported Python versions: **3.10, 3.11, 3.12, and 3.13**.

## 7.2 Optional capability groups

```powershell
# Development and tests
pip install -e ".[dev]"

# ONNX export/runtime
pip install -e ".[onnx]"

# TensorFlow + LiteRT edge path
pip install -e ".[edge]"

# Serial sensor acquisition
pip install -e ".[edge,hardware]"

# COMSOL automation
pip install -e ".[comsol]"

# SHAP explainability
pip install -e ".[xai]"

# Interactive research dashboard
pip install -e ".[dashboard]"

# Full research/development environment
pip install -e ".[dev,onnx,edge,hardware,comsol,xai,dashboard]"
```

COMSOL automation additionally requires a licensed COMSOL installation compatible with the Python `mph` package.

---

# 8. Developer workflow

## 8.1 Train the tandem model

```powershell
python -m sprpcf.ml.train_tandem `
  --data data/processed/synthetic.parquet `
  --epochs 50 `
  --out models/tandem.pt `
  --onnx-out models/inverse_pcf_spr.onnx `
  --seed 7
```

The checkpoint stores model states, feature ordering, physical validation metrics, scaler parameters, and the experiment seed.

## 8.2 Explain the forward surrogate

```powershell
python -m sprpcf.ml.explainability `
  --checkpoint models/tandem.pt `
  --data data/processed/synthetic.parquet `
  --out outputs/feature_attribution.csv `
  --heatmap outputs/feature_attribution.png
```

Integrated Gradients and optional SHAP use the same standardized five-variable input space as the trained forward surrogate.

## 8.3 Build a forward ensemble

```powershell
python -m sprpcf.ml.ensemble `
  --checkpoint models/tandem.pt `
  --data data/processed/training.parquet `
  --out models/tandem_ensemble.pt `
  --members 5 `
  --epochs 50 `
  --device auto
```

## 8.4 Generate Pareto-ranked designs

```powershell
python scripts/run_multiobjective_design.py `
  --checkpoint models/tandem_ensemble.pt `
  --targets data/processed/design_targets.csv `
  --reference-data data/processed/training.parquet `
  --out outputs/multiobjective `
  --candidates-per-target 128 `
  --confidence 0.95
```

See [`docs/ADVANCED_AI_DESIGN.md`](docs/ADVANCED_AI_DESIGN.md).

---

# 9. COMSOL physics workflow

Run a configured COMSOL sweep:

```powershell
python -m sprpcf.simulation.comsol_sweep `
  --model path\to\pcf_spr.mph `
  --config sweep.example.yaml `
  --out data/raw/comsol_sweep.parquet
```

Example unit contract:

```yaml
wavelength_scale_to_nm: 1.0
loss_scale_to_db_per_cm: 1.0
expected_wavelength_nm: [400.0, 1000.0]
```

If COMSOL returns wavelength in metres:

```yaml
wavelength_scale_to_nm: 1.0e9
```

Implausible wavelength ranges fail fast instead of being silently accepted.

## 9.1 Closed-loop verification

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

For each proposed design, the closed loop creates an odd, target-centered RI sweep. It independently calculates fitted sensitivity, mean FWHM, fitted FOM, target resonance error, and RI linearity. A failed RI point can reject the design rather than allowing an incomplete sweep to appear valid.

Only accepted physics rows are eligible for dataset augmentation.

See [`docs/COMSOL_CLOSED_LOOP.md`](docs/COMSOL_CLOSED_LOOP.md).

## 9.2 Pareto → COMSOL verification

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

---

# 10. Scientific validation pack

Generate reviewer-facing evidence:

```powershell
python scripts/run_validation_pack.py `
  --data data/processed/synthetic.parquet `
  --checkpoint models/tandem.pt `
  --out outputs/validation `
  --bootstrap-resamples 5000 `
  --mc-samples 64 `
  --seed 7
```

Run the physics-loss ablation:

```powershell
python scripts/run_ablation_study.py `
  --data data/processed/synthetic.parquet `
  --out outputs/ablation `
  --seeds 7,17,29 `
  --epochs 50 `
  --device cpu
```

The validation pack can report:

- fixed-geometry RI sensitivity,
- resonance FWHM and FOM,
- RI linearity `R²`,
- percentile-bootstrap confidence intervals,
- Ridge baseline versus neural surrogate,
- inverse target satisfaction,
- raw and post-projection fabrication validity,
- repeated-seed ablation,
- uncertainty diagnostics,
- provenance JSON,
- CSV evidence tables,
- manuscript-ready plots.

For values `x_1, …, x_N`, the bootstrap mean interval is produced by sampling with replacement, recalculating the mean for each resample, and taking the configured percentile limits. The random seed is explicit.

See [`docs/SCIENTIFIC_VALIDATION.md`](docs/SCIENTIFIC_VALIDATION.md).

---

# 11. Edge and hardware-facing inference

## 11.1 Train and quantize edge models

```powershell
python -m sprpcf.edge.train_denoiser `
  --data data/processed/synthetic.parquet `
  --epochs 20 `
  --batch-size 64 `
  --device auto `
  --quantize `
  --seed 7
```

Typical artifacts:

```text
models/edge_denoiser.keras
models/edge_ri_predictor.keras
models/edge_denoiser_quantized.tflite
models/edge_ri_predictor_quantized.tflite
```

The quantized deployment path uses LiteRT and validates full integer quantization.

## 11.2 Calibrate the runtime

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

Experimental calibration should use held-out labeled experimental spectra. Synthetic calibration must not be described as measured experimental coverage.

## 11.3 Replay a captured sensor stream

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

## 11.4 Serial spectrometer adapter

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

The preprocessing path supports wavelength calibration, dark/reference correction, non-extrapolating resampling, normalization, denoising, RI/resonance inference, spectral OOD evaluation, and latency/memory benchmarking.

Runtime reports can include mean latency, P50, P95, P99, throughput, peak Python heap, and process maximum RSS where supported.

See [`docs/HARDWARE_RUNTIME.md`](docs/HARDWARE_RUNTIME.md).

---

# 12. Research dashboard

The Streamlit dashboard is an interface to the existing research functions, not a second calculation engine.

Install and run:

```powershell
pip install -e ".[dashboard]"
streamlit run src/sprpcf/dashboard/app.py
```

With COMSOL support:

```powershell
pip install -e ".[dashboard,comsol]"
streamlit run src/sprpcf/dashboard/app.py
```

Dashboard flow:

```text
Target metrics
→ Pareto-ranked inverse designs
→ selected geometry
→ 3D PCF schematic
→ fabrication / confidence / OOD diagnostics
→ exact-candidate physics verification
→ RI-dependent spectra
→ sensitivity / FOM / linearity
→ scientific-validation evidence
→ edge benchmark evidence
→ explainability summary
→ SHA-256-bound Markdown report
```

The 3D PCF view is a schematic communication aid, not a COMSOL finite-element mesh.

See [`docs/DASHBOARD.md`](docs/DASHBOARD.md).

---

# 13. Evidence hierarchy

| Evidence source | Valid use | Must not be claimed as |
|---|---|---|
| Synthetic spectra | software tests, algorithm checks, CI, pipeline validation | experimental sensor performance |
| Surrogate prediction | candidate screening and inverse-design ranking | independent physics proof |
| COMSOL simulation | numerical-physics verification when model/configuration are valid | laboratory measurement |
| Recorded sensor data | hardware pipeline and experimental evaluation | evidence beyond the measurement protocol |
| Real-time device benchmark | latency/memory measured on that exact device | performance on untested hardware |

This hierarchy is reflected in manifests and reports so that manuscript evidence remains traceable to its actual source.

---

# 14. Reproducibility

Validate release metadata:

```powershell
python scripts/verify_release.py
```

Create a hash-bound experiment bundle:

```powershell
python scripts/create_reproducibility_bundle.py `
  --out outputs/repro/experiment_001 `
  --name experiment_001 `
  --seed 7 `
  --data data/processed/training.parquet `
  --checkpoint models/tandem.pt `
  --config configs/experiment.yaml
```

A reproducibility bundle records Git state, seed, experiment configuration, installed Python packages, artifact paths, and SHA-256 checksums.

Typical outputs:

```text
manifest.json
environment.json
environment.lock.txt
checksums.sha256
REPRODUCE.md
```

The repository also provides:

- `CITATION.cff`
- `MODEL_CARD.md`
- `DATASET_CARD.md`
- `Dockerfile`
- `.devcontainer/devcontainer.json`
- tag-time release validation

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

---

# 15. Testing and continuous integration

Local quality gate:

```powershell
ruff check .
python -m compileall -q src main.py scripts
python scripts/verify_release.py
pytest -q -W error
```

GitHub Actions validates:

- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13
- package imports
- release metadata
- wheel/source build
- core scientific and ML tests with warnings treated as errors
- edge/full-pipeline tests
- dashboard dependency installation and app import
- dashboard tests with warnings treated as errors

The CI policy is intentionally strict because numerical research software can return plausible-looking outputs even when an upstream warning indicates a broken execution path.

---

# 16. Repository structure

```text
CyberPhotonics-SPR/
├── src/sprpcf/
│   ├── simulation/     # spectra, metrics, COMSOL, dispersion, schemas
│   ├── ml/             # tandem model, ensemble, Pareto design, XAI, ONNX
│   ├── validation/     # scientific validation, ablation, closed loop
│   ├── edge/           # denoising, quantization, LiteRT, hardware runtime
│   ├── dashboard/      # Streamlit research interface and report helpers
│   └── utils/          # reproducibility and release utilities
├── scripts/            # reproducible command-line research workflows
├── tests/              # unit, integration, edge, dashboard, scientific tests
├── docs/               # detailed research and engineering protocols
├── data/raw/           # raw COMSOL / sensor data; normally not committed
├── data/processed/     # model-ready data; normally not committed
├── models/             # checkpoints and deployment models; normally not committed
├── outputs/            # evidence tables, figures, reports, benchmarks
├── CITATION.cff
├── MODEL_CARD.md
├── DATASET_CARD.md
├── Dockerfile
└── pyproject.toml
```

---

# 17. Design principles

### Scientific traceability
Every reported quantity should be reproducible from source data, equations, configuration, code, and artifact hashes.

### Fixed-geometry sensing calculations
RI sensitivity is calculated only while physical geometry remains fixed.

### Independent verification
Learned labels are not the sole validation source. The validation layer independently fits RI sensitivity and recalculates FOM and linearity.

### Explicit physical units
Geometry, wavelength, loss, RI, and deployment interfaces use documented units. COMSOL scaling is configuration-controlled.

### Constraint transparency
Both raw inverse-generator validity and post-projection validity are retained.

### Uncertainty before expensive physics
Ensemble disagreement, calibration intervals, OOD distance, and Pareto ranking help prioritize which designs deserve COMSOL evaluation.

### No evidence inflation
Synthetic, surrogate, COMSOL, and experimental results are labeled according to what they actually demonstrate.

### Reproducible engineering
Seeds, hashes, model metadata, package snapshots, CI, and evidence bundles are part of the research workflow.

---

# 18. Known limitations

- A neural surrogate is only as reliable as the quality and coverage of its reference data.
- Mahalanobis OOD scoring is a practical statistical diagnostic, not proof that every in-domain prediction is correct.
- Conformal intervals describe calibration-set residual behavior and should be recalibrated when the data distribution changes.
- The compact gold dispersion regularizer is not a substitute for measured optical constants in high-fidelity electromagnetic analysis.
- COMSOL automation cannot establish physical correctness if the underlying `.mph` model, boundary conditions, mesh, materials, or units are wrong.
- Hardware latency must be measured on the target device before making device-specific performance claims.
- Experimental claims require actual measured sensor data and an appropriate experimental protocol.

---

# 19. Citation and research use

If this repository contributes to academic work, use [`CITATION.cff`](CITATION.cff) and record the exact commit used for the reported experiment.

For manuscript-quality work, retain at minimum:

```text
Git commit
experiment seed
input dataset hash
model/checkpoint hash
COMSOL model/configuration hash when applicable
calibration configuration
validation thresholds
software environment snapshot
result/evidence hashes
```

This makes a reported number traceable to the exact computational state that produced it.

---

# 20. License

CyberPhotonics-SPR is distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for the complete terms.
