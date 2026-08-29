from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

import numpy as np

from sprpcf.edge.hardware import (
    EdgeCalibrationBundle,
    JSONLineSpectrumSource,
    KerasModel,
    LiteRTModel,
    LiveSensorPipeline,
    RawSpectrumFrame,
    SerialJSONLineSource,
    SpectrumPreprocessor,
    TransmissionCalibration,
    WavelengthCalibration,
    benchmark_pipeline,
    parse_csv_floats,
    write_inference_jsonl,
)
from sprpcf.ml.dataset import read_table
from sprpcf.utils.reproducibility import sha256_file


def _target_grid(path: Path) -> np.ndarray:
    frame = read_table(path).dropna(subset=["wavelength_nm"])
    if frame.empty:
        raise ValueError("Grid dataset does not contain wavelength_nm rows.")
    grids = [np.fromstring(str(value), sep=",", dtype=np.float64) for value in frame["wavelength_nm"].head(8)]
    if not grids or grids[0].size < 4:
        raise ValueError("wavelength_nm must contain comma-separated wavelength samples.")
    reference = grids[0]
    for grid in grids[1:]:
        if grid.shape != reference.shape or not np.allclose(grid, reference, rtol=0.0, atol=1e-9):
            raise ValueError("Grid dataset contains inconsistent wavelength axes.")
    if np.any(np.diff(reference) <= 0):
        raise ValueError("Model wavelength grid must be strictly increasing.")
    return reference


def _load_vector(path: Path | None) -> np.ndarray | None:
    if path is None:
        return None
    values = np.asarray(np.load(path), dtype=np.float64).reshape(-1)
    if values.size < 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Calibration vector {path} is empty or non-finite.")
    return values


def _model(path: Path, runtime: str):
    if runtime == "litert":
        return LiteRTModel(path)
    if runtime == "keras":
        return KerasModel(path)
    raise ValueError("runtime must be 'litert' or 'keras'.")


def _raw_frame_payload(frame: RawSpectrumFrame) -> dict[str, object]:
    frame.validate()
    return {
        "index": int(frame.index),
        "timestamp_s": float(frame.timestamp_s),
        "axis_kind": frame.axis_kind,
        "signal_kind": frame.signal_kind,
        "source": frame.source,
        "axis": np.asarray(frame.axis, dtype=np.float64).tolist(),
        "signal": np.asarray(frame.signal, dtype=np.float64).tolist(),
        "metadata": frame.metadata or {},
    }


def _write_raw_frames(frames: list[RawSpectrumFrame], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for frame in frames:
            handle.write(json.dumps(_raw_frame_payload(frame), sort_keys=True) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibrated PCF-SPR inference from a hardware or JSONL sensor feed.")
    parser.add_argument("--source", choices=["serial", "jsonl"], required=True)
    parser.add_argument("--serial-port")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--input-jsonl", type=Path)
    parser.add_argument("--grid-data", type=Path, required=True, help="Dataset whose wavelength_nm defines the model grid.")
    parser.add_argument("--denoiser", type=Path, required=True)
    parser.add_argument("--predictor", type=Path)
    parser.add_argument("--runtime", choices=["litert", "keras"], default="litert")
    parser.add_argument("--calibration", type=Path, help="Edge calibration bundle from calibrate_edge_runtime.py.")
    parser.add_argument("--wavelength-coefficients", help="Ascending polynomial coefficients for pixel->nm, comma-separated.")
    parser.add_argument("--dark-npy", type=Path)
    parser.add_argument("--reference-npy", type=Path)
    parser.add_argument("--path-length-cm", type=float, default=1.0)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument(
        "--raw-out-jsonl",
        type=Path,
        default=Path("outputs/hardware/raw_frames.jsonl"),
        help="Archive acquired raw frames before preprocessing; required for traceable experimental evidence.",
    )
    parser.add_argument("--out-jsonl", type=Path, default=Path("outputs/hardware/live_inference.jsonl"))
    parser.add_argument("--benchmark-iterations", type=int, default=0)
    parser.add_argument("--benchmark-warmup", type=int, default=5)
    parser.add_argument("--benchmark-out", type=Path, default=Path("outputs/hardware/benchmark.json"))
    parser.add_argument("--device-name", help="Human-readable exact target device identifier for benchmark provenance.")
    parser.add_argument("--os-name", help="Target OS image/version. Defaults to platform.platform().")
    parser.add_argument("--accelerator", help="CPU/GPU/NPU/other accelerator identity when applicable.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.frames < 1:
        raise ValueError("--frames must be >= 1.")
    if args.source == "serial" and not args.serial_port:
        raise ValueError("--source serial requires --serial-port.")
    if args.source == "jsonl" and args.input_jsonl is None:
        raise ValueError("--source jsonl requires --input-jsonl.")
    if (args.dark_npy is None) != (args.reference_npy is None):
        raise ValueError("--dark-npy and --reference-npy must be supplied together.")

    target_grid = _target_grid(args.grid_data)
    wavelength_calibration = (
        WavelengthCalibration(parse_csv_floats(args.wavelength_coefficients))
        if args.wavelength_coefficients
        else None
    )
    dark = _load_vector(args.dark_npy)
    reference = _load_vector(args.reference_npy)
    transmission_calibration = (
        TransmissionCalibration(dark=dark, reference=reference, path_length_cm=args.path_length_cm)
        if dark is not None and reference is not None
        else None
    )
    calibration = EdgeCalibrationBundle.load(args.calibration) if args.calibration else EdgeCalibrationBundle()

    preprocessor = SpectrumPreprocessor(
        target_wavelength_nm=target_grid,
        wavelength_calibration=wavelength_calibration,
        transmission_calibration=transmission_calibration,
    )
    pipeline = LiveSensorPipeline(
        preprocessor=preprocessor,
        denoiser=_model(args.denoiser, args.runtime),
        predictor=_model(args.predictor, args.runtime) if args.predictor else None,
        calibration=calibration,
    )

    source = (
        SerialJSONLineSource(args.serial_port, baudrate=args.baudrate, timeout_s=args.timeout_s)
        if args.source == "serial"
        else JSONLineSpectrumSource(args.input_jsonl)
    )
    collected = list(islice(iter(source), args.frames))
    try:
        close = getattr(source, "close", None)
        if callable(close):
            close()
    finally:
        if not collected:
            raise RuntimeError("Sensor source produced no frames.")

    _write_raw_frames(collected, args.raw_out_jsonl)
    results = [pipeline.process(frame) for frame in collected]
    write_inference_jsonl(results, args.out_jsonl)
    for result in results:
        print(json.dumps(result.to_dict(), sort_keys=True))

    summary: dict[str, object] = {
        "frames": len(results),
        "raw_output": str(args.raw_out_jsonl),
        "output": str(args.out_jsonl),
        "runtime": args.runtime,
        "source": args.source,
        "raw_sha256": sha256_file(args.raw_out_jsonl),
    }
    if args.benchmark_iterations > 0:
        stats = benchmark_pipeline(
            pipeline,
            collected,
            iterations=args.benchmark_iterations,
            warmup=args.benchmark_warmup,
        )
        benchmark_payload: dict[str, object] = {
            **stats,
            "schema_version": 2,
            "qualification_status": "unqualified_candidate",
            "source": args.source,
            "runtime": args.runtime,
            "device_name": args.device_name,
            "os_name": args.os_name or platform.platform(),
            "accelerator": args.accelerator,
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "denoiser_sha256": sha256_file(args.denoiser),
            "predictor_sha256": sha256_file(args.predictor) if args.predictor else None,
            "raw_frames_sha256": sha256_file(args.raw_out_jsonl),
        }
        args.benchmark_out.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_out.write_text(
            json.dumps(benchmark_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary["benchmark"] = benchmark_payload
        summary["benchmark_output"] = str(args.benchmark_out)

    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
