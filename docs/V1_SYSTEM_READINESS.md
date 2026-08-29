# v1.0 System Readiness Milestone

This milestone turns CyberPhotonics-SPR from a collection of validated subsystems into a single release-gated research system.

## Goal

A user must be able to move through the complete software path without hidden prerequisites:

```text
install/package
→ dataset/simulation contract
→ conditioned inverse design
→ Pareto + uncertainty/OOD
→ exact-candidate physics gate
→ edge/hardware pipeline
→ dashboard
→ reviewer package
→ submission package
→ release validation
→ stable-release decision
```

The system must also distinguish software completeness from physical/experimental evidence completeness.

## Definition of done

### 1. Repository and package completeness

- Python 3.10-3.13 supported.
- Core dependencies import successfully.
- Critical `sprpcf` modules are package-visible.
- Runtime directories exist and are writable.
- version, package metadata, citation metadata, RC metadata, license, cards, Docker/devcontainer, publication tooling and release tooling are present.
- a built wheel is smoke-imported from outside the repository source tree.

### 2. Scientific workflow completeness

- synthetic and COMSOL interfaces use explicit evidence classes;
- exact selected candidates enter the physics gate;
- fixed-geometry RI sweeps are used for sensitivity validation;
- failed COMSOL points cannot be silently upgraded to accepted evidence;
- only accepted physics rows can augment the training dataset;
- uncertainty/OOD scores are not described as fabrication-success probabilities.

### 3. COMSOL hand-off completeness

A tracked example configuration is provided at:

```text
configs/comsol_sweep.example.yaml
```

It is a contract template only. Study names, evaluated expressions, wavelength/loss scaling, materials, mesh, boundary conditions and the `.mph` model must be verified for the actual experiment.

### 4. Edge/hardware completeness

The software provides calibration, preprocessing, LiteRT inference, serial/JSONL/array sources, OOD/confidence handling and runtime benchmarking. Exact-device performance remains an evidence requirement and cannot be inferred from workstation or CI runs.

### 5. Publication and release completeness

Reviewer and manuscript-submission bundles retain evidence classes, hashes, manifests and missing-evidence flags. Stable software release does not require fabricated physical data; manuscript claims that explicitly depend on COMSOL, laboratory measurements or target-device benchmarks do require those evidence classes.

## Authoritative readiness command

Software/repository check:

```bash
python scripts/check_system_readiness.py --profile software --strict
```

Release-candidate check:

```bash
python scripts/check_system_readiness.py \
  --profile release \
  --expected-version 1.0.0rc1 \
  --json-out outputs/readiness/release.json \
  --markdown-out outputs/readiness/release.md \
  --strict
```

Full evidence check after real research artifacts exist:

```bash
python scripts/check_system_readiness.py \
  --profile full \
  --reviewer-package outputs/reviewer_package \
  --submission-package outputs/submission_package \
  --json-out outputs/readiness/full.json \
  --markdown-out outputs/readiness/full.md \
  --strict
```

`full` requires the evidence classes:

- `comsol_physics`
- `experimental_sensor`
- `device_benchmark`

A failure here is an evidence gap, not a request to substitute synthetic data.

## Optional capability reporting

The readiness report also records whether the current environment has optional runtime modules for:

- Streamlit dashboard,
- ONNX,
- serial hardware,
- COMSOL `mph`,
- SHAP XAI,
- LiteRT,
- TensorFlow.

Optional capability absence does not invalidate the core package because extras are intentionally installable independently. Dedicated CI jobs validate dashboard and edge environments.

## Stable v1.0 gate

Stable `v1.0.0` should be created only after:

1. exact-head CI is green;
2. release readiness is green;
3. the built wheel smoke-import passes;
4. reviewer/submission package integrity passes;
5. release notes and citation metadata are reviewed;
6. claims in the manuscript match the evidence actually supplied;
7. real COMSOL/experimental/device results are attached before making those corresponding claims.

The system can therefore be software-release-ready while still truthfully reporting that some physical research evidence has not yet been supplied.
