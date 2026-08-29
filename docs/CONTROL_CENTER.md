# CyberPhotonics-SPR Native Control Center

The default interactive entry point is:

```powershell
python main.py
```

That command opens the **native PySide6 desktop application**. It does not start a browser, local HTTP server, Streamlit session, or localhost dashboard.

The desktop composition follows the CyberPhotonics-SPR control-center design: persistent navigation on the left, global system state at the top, six readiness cards, sensorgram and pipeline panels, training/edge/HIL panels, quick actions, and a bottom system/session bar.

## Main dashboard

The overview surface contains:

- dataset state and row count;
- tandem checkpoint and ONNX readiness;
- INT8 denoiser and RI-predictor readiness;
- A → B → C pipeline state;
- HIL evidence state;
- overall workspace health;
- live spectrum preview from the project dataset when available;
- resonance/FWHM/RI context;
- pipeline stage visualization;
- forward/inverse training panels;
- edge deployment metrics;
- HIL metrics and thermal-drift visualization;
- eight quick actions.

## Native operations

Every normal workflow can be started from a Qt form. The user does not have to type a command.

### Data & training

- Generate Dataset
- Train Inverse Model
- Train Edge Models
- Open exported models

### Pipeline & streaming

- Run full A → B → C pipeline
- Run streaming benchmark

### HIL

- mock transport
- serial transport
- socket transport
- target FPS / buffer / duration
- optional thermal-drift injection
- JSON benchmark report

### Research design

- calibrated multi-objective inverse design
- candidate population generation
- fabrication projection
- uncertainty/OOD scoring
- Pareto selection

### Physics gate

- exact selected-candidate verification
- synthetic software-validation backend
- COMSOL backend when model/configuration are supplied
- fixed-geometry RI sweep
- reviewer-facing acceptance evidence

### Evidence & report

- open generated evidence artifacts
- produce reviewer-readable Markdown research evidence

## Execution model

Long-running work is started with Qt `QProcess` using the current Python executable and the same project backend used by the CLI. Output is streamed into a native process-console dialog.

No user-provided path or value is passed through a shell.

This keeps:

1. GUI and CLI behavior aligned;
2. TensorFlow/PyTorch training isolated from the GUI process;
3. the Qt event loop responsive during long operations;
4. command execution safe from shell interpolation.

## CLI compatibility

The CLI remains available for automation and reproducible scripted experiments:

```powershell
python main.py generate-data
python main.py train-inverse
python main.py train-edge
python main.py run-pipeline
python main.py simulate-stream
python main.py hil-benchmark
```

The aliases below open the same native GUI:

```powershell
python main.py gui
python main.py dashboard
```

The former Streamlit implementation is retained only as an explicit legacy option:

```powershell
python main.py web-dashboard
```

It is not used by the normal desktop workflow.

## Evidence boundary

The native interface preserves the repository evidence policy:

- synthetic outputs validate software and analytical flow;
- verified COMSOL outputs may support numerical-physics claims;
- experimental performance claims require measured sensor data.

The GUI does not relabel surrogate or synthetic values as experimental measurements.
