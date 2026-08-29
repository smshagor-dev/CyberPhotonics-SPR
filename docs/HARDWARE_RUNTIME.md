# Hardware and Real-Sensor Runtime

This phase turns the edge models into a calibrated sensor-runtime interface instead of treating stored spectra as if they were hardware.

## Runtime contract

A hardware driver emits one newline-delimited JSON object per spectrum:

```json
{
  "index": 42,
  "timestamp_s": 1787950000.25,
  "axis_kind": "wavelength_nm",
  "signal_kind": "loss_db_per_cm",
  "axis": [600.0, 601.0, 602.0, 603.0],
  "signal": [0.8, 0.9, 1.5, 0.7],
  "metadata": {"device": "spectrometer-01"}
}
```

For a raw spectrometer, `axis_kind` may be `pixel` and `signal_kind` may be `intensity`. Pixel data requires a polynomial wavelength calibration. Intensity data requires dark/reference calibration and a positive optical path length.

The serial transport uses exactly the same JSON object followed by `\n`. This keeps the acquisition layer device-independent: a vendor SDK, microcontroller, or USB spectrometer can translate its native frame into this protocol.

## Measurement preprocessing

The runtime performs these operations in order:

1. validate finite, equal-length axis/signal arrays;
2. convert pixel index to wavelength using the calibrated polynomial when needed;
3. convert measured intensity to loss using dark/reference correction:

```text
T(lambda) = (I(lambda) - dark(lambda)) / (reference(lambda) - dark(lambda))
loss_db_per_cm(lambda) = -10 log10(T(lambda)) / path_length_cm
```

4. sort and validate a unique monotonic wavelength axis;
5. linearly resample onto the exact model wavelength grid;
6. reject any request that would require extrapolation;
7. normalize the measured loss spectrum using that frame's observed mean/std;
8. run the denoiser and optional RI/resonance predictor;
9. compute the physical resonance wavelength again from the denoised loss spectrum;
10. attach spectral OOD and held-out prediction intervals when a calibration bundle is available.

The runtime never labels an OOD/confidence number as a probability of physical success.

## Build a held-out edge calibration bundle

Use labeled COMSOL or, preferably, experimental spectra. The split is grouped by geometry so the same base geometry cannot appear in both reference and held-out calibration groups.

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

Outputs:

- `edge_calibration.json` — spectral OOD reference distribution and absolute-error prediction intervals;
- `edge_calibration.json.meta.json` — data/model hashes, split size, seed, runtime, and evidence class.

If calibration data are synthetic, the resulting intervals remain software-only evidence. Experimental calibration is required before claiming sensor accuracy or coverage in a physical deployment.

## JSONL replay

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

## Serial spectrometer

Install the optional serial transport:

```powershell
pip install -e ".[edge,hardware]"
```

Then run:

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

For a device that sends pixel/intensity frames:

```powershell
python scripts/run_hardware_pipeline.py `
  --source serial `
  --serial-port COM5 `
  --grid-data data/processed/training.parquet `
  --denoiser models/edge_denoiser_quantized.tflite `
  --predictor models/edge_ri_predictor_quantized.tflite `
  --wavelength-coefficients "401.82,0.4217,-0.0000021" `
  --dark-npy calibration/dark.npy `
  --reference-npy calibration/reference.npy `
  --path-length-cm 1.0 `
  --frames 100 `
  --benchmark-iterations 500
```

Polynomial coefficients are ascending order: `c0,c1,c2,...` for

```text
lambda_nm = c0 + c1*pixel + c2*pixel^2 + ...
```

## Runtime evidence

Each output JSONL row contains:

- measured resonance wavelength from the denoised physical loss spectrum;
- peak loss;
- predicted RI and model-predicted resonance wavelength when the predictor is enabled;
- denoiser correction RMSE;
- normalized spectral OOD score and in-distribution flag;
- calibrated RI and resonance intervals when available;
- end-to-end frame latency.

The benchmark report contains:

- P50 latency;
- P95 latency;
- P99 latency;
- mean latency;
- throughput in frames/s;
- peak Python heap;
- process max RSS when the platform exposes it.

These are runtime measurements, not fabricated Raspberry Pi/Jetson performance. Device-specific benchmark claims must be generated on the actual device and preserved with the hardware/software metadata used for that run.

## Deployment progression

Recommended evidence order:

1. CI mock frames;
2. stored experimental JSONL replay;
3. workstation USB/serial spectrometer;
4. Raspberry Pi or Jetson with LiteRT INT8;
5. repeated environmental/temperature sessions;
6. final held-out experimental validation.

Do not replace COMSOL or experimental sensing evidence with synthetic runtime tests.
