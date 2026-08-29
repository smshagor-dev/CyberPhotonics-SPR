# CyberPhotonics-SPR Control Center

The default interactive entry point is now:

```powershell
python main.py
```

That command launches the Streamlit control center on `localhost:8501`. Researchers do not need to memorize CLI subcommands for normal operation.

## Pages

### Overview

Shows the current workspace state before a long experiment starts:

- synthetic/reference dataset availability;
- tandem checkpoint and ONNX export status;
- Keras and quantized TFLite edge artifacts;
- HIL report availability;
- optional runtime capabilities such as TensorFlow, LiteRT, COMSOL `mph`, serial hardware, and SHAP;
- recent dashboard-run task history.

### Data & Training

Provides forms for all data/model build operations:

- synthetic fixed-geometry RI-sweep generation;
- tandem inverse-model training;
- separate forward/inverse epoch schedules;
- ONNX export;
- edge denoiser and RI-predictor training;
- optional quantized TFLite export.

### Pipeline & Streaming

Runs the complete software path from one form:

```text
synthetic data
  -> tandem inverse model
  -> edge denoiser / RI predictor
  -> INT8 TFLite export
  -> streaming benchmark
```

The streaming page can also benchmark already-built TFLite models without retraining.

### HIL Lab

Runs the Phase 4 hardware-in-the-loop benchmark with:

- mock transport;
- serial transport;
- socket transport;
- configurable duration/FPS/buffer size;
- optional thermal-drift injection;
- JSON report output.

### Research Design

Uses the trained tandem checkpoint and reference dataset for calibrated multi-objective inverse design, fabrication projection, OOD scoring, uncertainty calibration, and Pareto selection.

### Physics Gate

Preserves the exact selected design and verifies it with either:

- the synthetic/software-validation backend; or
- a user-supplied COMSOL model and sweep configuration.

Acceptance thresholds remain explicit and reviewer-facing.

### Evidence & Report

Loads validation, XAI, and hardware evidence, calculates file hashes, and exports a reviewer-readable Markdown evidence report.

## How operations are executed

The dashboard does not reimplement training code. Each long operation is launched through the same `main.py` CLI backend in an isolated child process using the current Python interpreter.

Commands are constructed as argument lists and are never passed through a shell. This provides three useful properties:

1. dashboard and CLI behavior stay aligned;
2. TensorFlow/PyTorch runtime state cannot contaminate the Streamlit server process;
3. user-entered paths and values are not interpreted as shell syntax.

Combined stdout/stderr is streamed back into the dashboard and kept in a short in-session operation history.

## CLI compatibility

The deterministic CLI remains available for automation, CI, scripts, and reproducible experiment manifests:

```powershell
python main.py generate-data
python main.py train-inverse
python main.py train-edge
python main.py run-pipeline
python main.py simulate-stream
python main.py hil-benchmark
python main.py dashboard
```

Use `python main.py -h` or `<subcommand> -h` when working from a terminal.

## Evidence boundary

The control center preserves the repository evidence policy:

- synthetic results validate software and analytical flow;
- verified COMSOL runs may support numerical-physics claims;
- experimental performance claims require measured sensor data.

The dashboard does not relabel surrogate or synthetic outputs as experimental measurements.
