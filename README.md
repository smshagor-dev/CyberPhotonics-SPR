# CyberPhotonics-SPR

[![CI](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/workflows/ci.yml/badge.svg)](https://github.com/smshagor-dev/CyberPhotonics-SPR/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10--3.13-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-blue)

**CyberPhotonics-SPR** is a research and engineering framework for photonic crystal fiber surface plasmon resonance (PCF-SPR) sensor design. It connects numerical photonics, physics-informed machine learning, constrained inverse design, COMSOL verification, uncertainty-aware multi-objective optimization, edge inference, hardware-facing spectrum processing, and reproducible scientific reporting in one codebase.

The project is written for two audiences at the same time:

- **Researchers** who need traceable equations, scientifically valid refractive-index sweeps, independent validation metrics, uncertainty estimates, reproducible experiments, and a clear distinction between synthetic, simulated, and experimental evidence.
- **Developers** who need a maintainable Python package, tested command-line workflows, deterministic pipelines, ONNX/LiteRT deployment, CI, modular interfaces, and machine-readable artifacts.

The mathematical calculation path is treated as part of the software contract. Sensitivity, FWHM, FOM, linearity, fabrication constraints, calibration intervals, OOD scores, and design ranking are calculated explicitly and are not hidden behind dashboard summaries.

> **Evidence policy**  
> Synthetic data are used to validate software and methodology. COMSOL results are numerical-physics evidence only when produced from a verified model and configuration. Experimental performance claims require measured sensor data. The repository does not convert synthetic or surrogate predictions into experimental claims.

---

## 1. Research problem

A PCF-SPR sensor is governed by a coupled relationship between geometry, material dispersion, analyte refractive index, and the wavelength-dependent plasmonic loss response. In forward form, the problem can be written as

$$
\mathbf{m}=f(\mathbf{g},n_a),
$$

where

$$
\mathbf{g}=
\left[
\Lambda,
\frac{d}{\Lambda},
t_{Au},
r_c
\right]
$$

is the physical geometry and

$$
\mathbf{m}=
\left[
S_\lambda,
\mathrm{FOM},
\lambda_{res}
\right]
$$

contains the sensing metrics.

The inverse-design problem is harder because several geometries may produce similar optical responses. The practical objective is therefore not simply to predict one geometry, but to find fabrication-valid candidates that satisfy a sensing target, remain inside the calibrated model domain, and can be re-verified by numerical physics.

CyberPhotonics-SPR implements this as

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
Independent sensitivity / FWHM / FOM / R² calculation
    ↓
Accept or reject
    ↓
Dataset augmentation / retraining / edge deployment / evidence report
```

A selected dashboard or Pareto candidate retains the same geometry identity when it is passed to the physics-verification stage. The interface does not display one design and silently simulate another.

---

## 2. Core scientific variables and units

| Symbol | Repository field | Meaning | Unit |
|---|---|---|---|
| $\Lambda$ | `pitch_um` | PCF pitch | µm |
| $d/\Lambda$ | `d_over_lambda` | normalized air-hole diameter | dimensionless |
| $d$ | derived | air-hole diameter, $d=\Lambda(d/\Lambda)$ | µm |
| $t_{Au}$ | `metal_thickness_nm` | gold layer thickness | nm |
| $r_c$ | `channel_radius_um` | sensing-channel radius | µm |
| $n_a$ | `analyte_ri` | analyte refractive index | RIU |
| $\lambda_{res}$ | `lambda_res_nm` | resonance wavelength | nm |
| $\mathrm{FWHM}$ | `fwhm_nm` | resonance full width at half maximum | nm |
| $S_\lambda$ | `sensitivity_nm_per_riu` | wavelength sensitivity | nm/RIU |
| FOM | `fom_per_riu` | figure of merit | RIU$^{-1}$ |

Current fabrication envelope:

$$
0.8 \le \Lambda \le 4.0\;\mu m
$$

$$
0.20 \le d/\Lambda \le 0.90
$$

$$
15 \le t_{Au} \le 80\;nm
$$

$$
0.20 \le r_c \le 1.50\;\mu m
$$

with the geometric non-overlap condition

$$
d=\Lambda\left(\frac{d}{\Lambda}\right)<\Lambda.
$$

These constraints are checked in physical units at simulation boundaries and are also used as differentiable penalties during inverse-model training.

---

# 3. Mathematical and computational formulation

This section describes the calculations used by the implementation. For publication work, these equations should be read together with the experiment configuration and generated provenance files.

## 3.1 Silica dispersion: Sellmeier model

The fused-silica refractive index is calculated with the three-term Sellmeier equation

$$
n_{SiO_2}^2(\lambda)
=
1+
\sum_{j=1}^{3}
\frac{B_j\lambda^2}{\lambda^2-C_j},
$$

where wavelength is expressed in µm. The implementation uses

$$
B=[0.6961663,\;0.4079426,\;0.8974794]
$$

and

$$
C=[0.0684043^2,\;0.1162414^2,\;9.896161^2]\;\mu m^2.
$$

The same relation is available in NumPy and differentiable PyTorch form so that the physical regularizer and the simulation-side calculations use a consistent definition.

## 3.2 Gold dispersion: Drude-Lorentz model

For physics regularization, gold is represented by a compact Drude-Lorentz permittivity model. Photon energy is first obtained from

$$
E=\frac{hc}{\lambda},
$$

with

$$
hc=1.239841984\;\mathrm{eV\,\mu m}.
$$

The complex permittivity follows the form

$$
\varepsilon_{Au}(E)
=
\varepsilon_\infty
-
\frac{f_D\omega_p^2}{E(E+i\gamma_D)}
+
\sum_j
\frac{f_j\omega_p^2}{E_j^2-E^2-i\gamma_jE}.
$$

This model is used as a physical regularization aid. It is **not** presented as a replacement for experimentally measured optical constants in a high-fidelity COMSOL study.

## 3.3 Resonance wavelength

Given a wavelength array $\lambda_i$ and confinement-loss values $L_i$, the strongest physically represented peak is selected. A three-point quadratic fit is then used around the discrete peak to reduce wavelength-grid bias.

For local coordinates $x=\lambda-\lambda_i$,

$$
L(x)=ax^2+bx+c.
$$

When $a<0$ and the parabola vertex remains inside the three-sample interval, the refined resonance offset is

$$
x^*=-\frac{b}{2a},
$$

therefore

$$
\lambda_{res}=\lambda_i+x^*.
$$

The method refines the peak inside measured/simulated neighboring samples; it does not extrapolate a resonance outside the available local spectrum.

## 3.4 Full width at half maximum

The implementation uses the spectrum minimum as the local baseline,

$$
L_{base}=\min_i L_i,
$$

and defines the half-height level as

$$
L_{1/2}
=
L_{base}
+
\frac{L_{peak}-L_{base}}{2}.
$$

Left and right half-height crossings are linearly interpolated. For two samples $(\lambda_0,L_0)$ and $(\lambda_1,L_1)$ surrounding a crossing,

$$
\lambda_{cross}
=
\lambda_0
+
\frac{L_{1/2}-L_0}{L_1-L_0}
(\lambda_1-\lambda_0).
$$

The final width is

$$
\mathrm{FWHM}
=
\lambda_{right}-\lambda_{left}.
$$

Using interpolated crossings avoids quantizing FWHM to the wavelength-grid spacing.

## 3.5 Wavelength sensitivity

Sensitivity must be calculated while **geometry is fixed** and only analyte refractive index changes. Mixing unrelated geometries in one derivative is scientifically invalid and is explicitly avoided by this project.

For neighboring RI points,

$$
S_\lambda
\approx
\frac{\Delta\lambda_{res}}{\Delta n_a}
\quad [\mathrm{nm/RIU}].
$$

The dataset-generation path uses local finite-difference gradients within each fixed-geometry RI sweep.

For independent scientific validation, the repository also fits a straight line across the full RI sweep,

$$
\lambda_{res}=S_{fit}n_a+b,
$$

where the least-squares sensitivity is

$$
S_{fit}
=
\frac{
\sum_i(n_i-\bar n)(\lambda_i-\bar\lambda)
}{
\sum_i(n_i-\bar n)^2
}.
$$

This fitted sensitivity is intentionally independent from the local finite-difference labels used during learning.

## 3.6 RI linearity

The linearity of the RI-to-resonance relationship is reported as

$$
R^2
=
1-
\frac{
\sum_i(\lambda_i-\hat\lambda_i)^2
}{
\sum_i(\lambda_i-\bar\lambda)^2
}.
$$

Closed-loop validation can reject a design when the RI response falls below a specified $R^2$ threshold, even if individual target errors appear acceptable.

## 3.7 Figure of merit

The wavelength-domain figure of merit is

$$
\mathrm{FOM}
=
\frac{|S_\lambda|}{\mathrm{FWHM}}
\quad [\mathrm{RIU}^{-1}].
$$

A large sensitivity alone is therefore not sufficient; broad resonances reduce FOM.

## 3.8 Worked calculation example

The following values are **illustrative only** and are not reported project results.

Assume one fixed geometry is simulated for three analyte refractive indices:

| $n_a$ | $\lambda_{res}$ |
|---:|---:|
| 1.33 | 620 nm |
| 1.35 | 636 nm |
| 1.37 | 652 nm |

Using the first two points,

$$
S_\lambda
=
\frac{636-620}{1.35-1.33}
=
\frac{16}{0.02}
=800\;\mathrm{nm/RIU}.
$$

The same slope is obtained from all three points because they are exactly linear, so

$$
S_{fit}=800\;\mathrm{nm/RIU},
\qquad
R^2=1.0.
$$

Suppose the resonance baseline is $2\;\mathrm{dB/cm}$ and the peak is $10\;\mathrm{dB/cm}$. Then

$$
L_{1/2}
=2+\frac{10-2}{2}
=6\;\mathrm{dB/cm}.
$$

If interpolation gives half-height crossings at 616 nm and 656 nm,

$$
\mathrm{FWHM}=656-616=40\;nm.
$$

The corresponding figure of merit is

$$
\mathrm{FOM}
=
\frac{800}{40}
=20\;\mathrm{RIU}^{-1}.
$$

This is the same calculation chain used conceptually by the validation pipeline: spectrum → resonance → FWHM → fixed-geometry sensitivity → FOM → linearity/acceptance.

---

# 4. Physics-informed tandem inverse design

## 4.1 Standardization

Geometry, analyte condition, and target metrics are standardized independently:

$$
\tilde x=\frac{x-\mu_x}{\sigma_x}.
$$

Scaler means and scales are stored in the model checkpoint and are also embedded into the physical-unit ONNX export path.

The conditioned forward surrogate receives

```text
pitch_um
d_over_lambda
metal_thickness_nm
channel_radius_um
analyte_ri
```

and predicts

```text
sensitivity_nm_per_riu
fom_per_riu
lambda_res_nm
```

The inverse generator receives the desired sensing metrics, analyte RI, and a latent vector, and proposes the four geometry variables.

## 4.2 Forward-model loss

The forward model is trained in standardized metric space with mean-squared error,

$$
\mathcal L_{forward}
=
\frac{1}{N}
\sum_{i=1}^{N}
\|f(\tilde{\mathbf g}_i,\tilde n_i)-\tilde{\mathbf m}_i\|_2^2.
$$

## 4.3 Tandem inverse loss

The inverse generator is optimized through a frozen forward surrogate. In simplified form,

$$
\mathcal L_{inverse}
=
\mathcal L_{target}
+
\alpha\mathcal L_{overlap}
+
\beta\mathcal L_{bounds}
+
\gamma\mathcal L_{dispersion}.
$$

Target satisfaction is

$$
\mathcal L_{target}
=
\mathrm{MSE}
\left(
 f(g(\mathbf m^*,n_a,\mathbf z),n_a),
 \mathbf m^*
\right).
$$

For air-hole overlap,

$$
\mathcal L_{overlap}
=
\operatorname{mean}
\left[
\operatorname{ReLU}
\left(
\frac{d-\Lambda}{\max(\Lambda,\epsilon)}
\right)^2
\right].
$$

For a variable $x$ with lower and upper fabrication limits $l$ and $u$, the normalized boundary penalty is

$$
\mathcal L_{bound}(x)
=
\operatorname{mean}
\left[
\operatorname{ReLU}\left(\frac{l-x}{u-l}\right)^2
+
\operatorname{ReLU}\left(\frac{x-u}{u-l}\right)^2
\right].
$$

The total boundary loss is the sum over pitch, normalized hole diameter, metal thickness, and channel radius.

Raw inverse outputs and post-projection outputs are evaluated separately. A physically projected design is therefore not allowed to hide poor unconstrained generator behavior in the validation report.

---

# 5. Uncertainty, calibration, OOD detection, and Pareto design

A single inverse prediction is not treated as sufficient evidence for an ill-posed design problem. The advanced design pipeline can combine multiple forward-surrogate members, held-out conformal residual calibration, Mahalanobis OOD scoring, fabrication projection distance, and multi-objective ranking.

## 5.1 Ensemble prediction

For $K$ forward models,

$$
\bar{\mathbf m}
=
\frac{1}{K}\sum_{k=1}^{K}\mathbf m_k,
$$

with ensemble sample standard deviation used as a model-disagreement signal.

## 5.2 Conformal residual interval

On a held-out calibration set, absolute residuals are calculated as

$$
r_{ij}=|\hat m_{ij}-m_{ij}|.
$$

A finite-sample conformal quantile is selected for each metric. Candidate intervals use the calibrated residual half-width together with ensemble disagreement. These intervals describe held-out model error behavior; they are not a substitute for physics verification.

## 5.3 Mahalanobis OOD score

The standardized training input is modeled by center $\boldsymbol\mu$ and regularized covariance $\Sigma$. Distance is

$$
D_M(\mathbf x)
=
\sqrt{
(\mathbf x-\boldsymbol\mu)^T
\Sigma^{-1}
(\mathbf x-\boldsymbol\mu)
}.
$$

The reported normalized OOD score is

$$
\mathrm{OOD}(\mathbf x)
=
\frac{D_M(\mathbf x)}{D_{threshold}}.
$$

Therefore

```text
OOD <= 1.0  → inside the calibrated reference threshold
OOD > 1.0   → outside that threshold
```

This is a model-domain diagnostic, not a statement that a physical device is valid or invalid.

## 5.4 Pareto objectives

For each target metric $j$, normalized target error is

$$
e_j
=
\frac{|\hat m_j-m_j^*|}{s_j},
$$

where $s_j$ is the stored physical metric scale.

The Pareto ranking minimizes separate target errors together with fabrication projection distance and OOD score. Candidate $A$ dominates candidate $B$ when it is no worse in every objective and strictly better in at least one.

The implementation also reports a practical ranking score

$$
C
=
\operatorname{mean}(e_j)
+0.25p
+0.20\max(o-1,0)
+0.10(1-c),
$$

where

- $p$ is normalized fabrication projection distance,
- $o$ is normalized OOD score,
- $c$ is target-interval coverage fraction.

The displayed confidence ranking is

$$
R
=
\frac{e^{-\operatorname{clip}(C,0,50)}}{1+\max(o-1,0)}.
$$

**This ranking value is not a calibrated probability of physical success.** COMSOL or experimental verification remains a separate stage.

---

# 6. Scientifically valid data construction

The most important data rule in the repository is simple:

> **Sensitivity is never calculated across unrelated random geometries.**

Each base geometry is evaluated at multiple analyte RI values. The physical geometry remains fixed while $n_a$ changes. The same geometry group is also kept together during train/validation splitting to prevent RI-sweep leakage.

Generate software-validation data:

```powershell
python scripts/generate_synthetic_dataset.py `
  --samples 100 `
  --out data/processed/synthetic.parquet
```

With the default five RI values, 100 base geometries generate 500 spectrum rows.

Synthetic spectra are useful for testing metric extraction, model training, validation, ONNX/LiteRT deployment, and orchestration. They are not experimental PCF-SPR measurements.

---

# 7. Installation

## 7.1 Base environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
```

Supported Python versions:

```text
3.10
3.11
3.12
3.13
```

## 7.2 Optional capability groups

```powershell
# Development and tests
pip install -e ".[dev]"

# ONNX export/runtime
pip install -e ".[onnx]"

# Edge TensorFlow + LiteRT
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

COMSOL automation requires a licensed COMSOL installation compatible with the Python `mph` package.

---

# 8. Developer workflow

## 8.1 Train the conditioned tandem model

```powershell
python -m sprpcf.ml.train_tandem `
  --data data/processed/synthetic.parquet `
  --epochs 50 `
  --out models/tandem.pt `
  --onnx-out models/inverse_pcf_spr.onnx `
  --seed 7
```

The checkpoint stores model states, physical validation metrics, feature ordering, scaler parameters, and seed information.

## 8.2 Explain the forward surrogate

```powershell
python -m sprpcf.ml.explainability `
  --checkpoint models/tandem.pt `
  --data data/processed/synthetic.parquet `
  --out outputs/feature_attribution.csv `
  --heatmap outputs/feature_attribution.png
```

Integrated Gradients and optional SHAP operate in the same standardized five-variable input space used by the forward surrogate.

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

## 8.4 Generate Pareto-ranked inverse designs

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

A COMSOL sweep is configured explicitly, including unit conversion expectations.

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

If COMSOL returns wavelength in meters,

```yaml
wavelength_scale_to_nm: 1.0e9
```

Implausible wavelength ranges fail rather than being silently accepted.

## Closed-loop verification

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

For every proposed design, the closed loop creates an odd, target-centered analyte-RI sweep. It then calculates an independent fitted sensitivity, mean FWHM, fitted FOM, target resonance error, and RI linearity. A failed RI point can reject the design rather than allowing a partial successful sweep to appear valid.

Only accepted physics rows are eligible for dataset augmentation.

See [`docs/COMSOL_CLOSED_LOOP.md`](docs/COMSOL_CLOSED_LOOP.md).

## Advanced Pareto → COMSOL loop

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

Generate reviewer-facing validation artifacts:

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

The validation pack can report:

- fixed-geometry RI sweep sensitivity,
- FWHM and FOM,
- RI linearity $R^2$,
- deterministic percentile-bootstrap intervals,
- Ridge regression baseline versus neural surrogate,
- inverse target satisfaction,
- raw and post-projection fabrication validity,
- repeated-seed ablation,
- uncertainty diagnostics,
- provenance JSON,
- CSV evidence tables,
- manuscript-ready 300-dpi plots.

For a set of values $x_1,\ldots,x_N$, the reported bootstrap mean interval is obtained by resampling with replacement, calculating the mean for each resample, and taking the requested percentile bounds. The seed is explicit so the software-validation report is reproducible.

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

Generated model artifacts include

```text
models/edge_denoiser.keras
models/edge_ri_predictor.keras
models/edge_denoiser_quantized.tflite
models/edge_ri_predictor_quantized.tflite
```

The deployment path validates full integer quantization and uses LiteRT for the quantized runtime.

## 11.2 Calibrate the sensor runtime

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

Experimental calibration should use labeled held-out experimental spectra. Synthetic calibration must not be reported as measured experimental coverage.

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

## 11.4 Read from a serial spectrometer adapter

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

The preprocessing layer supports wavelength calibration, dark/reference correction, non-extrapolating resampling, normalization, denoising, RI/resonance inference, spectral OOD evaluation, and latency/memory benchmarking.

Reported runtime statistics can include mean latency, P50, P95, P99, throughput, peak Python heap, and process maximum RSS where the platform exposes it.

See [`docs/HARDWARE_RUNTIME.md`](docs/HARDWARE_RUNTIME.md).

---

# 12. Flagship research dashboard

The Streamlit dashboard is an interface to the existing research pipeline, not a separate calculation engine.

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

The dashboard can present:

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

The 3D geometry view is a schematic representation for communication and inspection; it is not presented as a COMSOL finite-element mesh.

See [`docs/DASHBOARD.md`](docs/DASHBOARD.md).

---

# 13. Evidence hierarchy

| Evidence source | Valid use | Must not be claimed as |
|---|---|---|
| Synthetic spectra | software tests, algorithm checks, CI, pipeline validation | experimental sensor performance |
| Surrogate prediction | candidate screening and inverse-design ranking | independent physics proof |
| COMSOL simulation | numerical-physics verification when model/configuration are valid | laboratory measurement |
| Recorded sensor data | hardware pipeline and experimental evaluation | broader evidence than the measurement protocol supports |
| Real-time device benchmark | measured latency/memory on that exact device | performance on untested hardware |

This hierarchy is intentionally reflected in manifests and reports so that evidence classes remain distinguishable during manuscript preparation.

---

# 14. Reproducibility

Validate release metadata:

```powershell
python scripts/verify_release.py
```

Create a hash-bound reproducibility bundle:

```powershell
python scripts/create_reproducibility_bundle.py `
  --out outputs/repro/experiment_001 `
  --name experiment_001 `
  --seed 7 `
  --data data/processed/training.parquet `
  --checkpoint models/tandem.pt `
  --config configs/experiment.yaml
```

A bundle records the Git state, seed, experiment configuration, installed Python-package snapshot, artifact paths, and SHA-256 checksums. Typical outputs are

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

GitHub Actions currently validates:

- Python 3.10
- Python 3.11
- Python 3.12
- Python 3.13
- package imports
- release metadata
- wheel/source build on the primary packaging job
- core scientific/ML tests with warnings treated as errors
- edge/full-pipeline tests
- dashboard dependency installation and app import
- dashboard tests with warnings treated as errors

The CI policy is deliberately strict because numerical research software can produce apparently reasonable outputs even when an upstream warning indicates a broken or deprecated execution path.

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
Every major reported quantity should be reproducible from source data, equations, configuration, and code.

### Fixed-geometry sensing calculations
RI sensitivity is calculated only while physical geometry remains fixed.

### Independent verification
Learned labels are not the sole source of validation. The scientific-validation layer independently fits RI sensitivity and checks FOM and linearity.

### Explicit physical units
Geometry, wavelength, loss, RI, and deployment interfaces use documented units. COMSOL scaling is configuration-controlled.

### Constraint transparency
Both raw generator validity and post-projection validity are retained.

### Uncertainty before expensive physics
Ensemble disagreement, calibration intervals, OOD distance, and Pareto ranking help prioritize which designs deserve COMSOL evaluation.

### No evidence inflation
Synthetic, surrogate, COMSOL, and experimental results are labeled according to what they actually demonstrate.

### Reproducible engineering
Seeds, hashes, model metadata, package snapshots, CI, and portable evidence bundles are part of the research workflow rather than afterthoughts.

---

# 18. Known limitations

- A neural surrogate is only reliable within the quality and coverage of its reference data.
- Mahalanobis OOD scoring is a practical statistical diagnostic, not a proof that all in-domain predictions are correct.
- Conformal intervals describe calibration-set residual behavior and should be re-calibrated when the data distribution changes.
- The compact gold dispersion regularizer is not a substitute for measured optical constants in high-fidelity electromagnetic analysis.
- COMSOL automation cannot establish physical correctness if the underlying `.mph` model, boundary conditions, mesh, materials, or units are wrong.
- Hardware latency must be measured on the target device before making device-specific performance claims.
- Experimental claims require actual measured sensor data and an appropriate experimental protocol.

---

# 19. Citation and research use

If this repository contributes to academic work, use the metadata in [`CITATION.cff`](CITATION.cff) and record the exact repository version or commit used for the reported experiment.

For a manuscript-quality result, retain at minimum:

```text
Git commit
experiment seed
input dataset hash
model/checkpoint hash
COMSOL model and configuration hash when applicable
calibration configuration
validation thresholds
software environment snapshot
result/evidence hashes
```

This makes a reported number traceable to the exact computational state that produced it.

---

# 20. License

CyberPhotonics-SPR is distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for the complete terms.
