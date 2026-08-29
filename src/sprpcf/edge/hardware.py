from __future__ import annotations

import json
import math
import sys
import time
import tracemalloc
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from sprpcf.simulation.metrics import resonance_wavelength

AxisKind = Literal["wavelength_nm", "pixel"]
SignalKind = Literal["loss_db_per_cm", "intensity"]


class PredictiveModel(Protocol):
    def predict(self, inputs: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class RawSpectrumFrame:
    """One sensor frame before wavelength/signal calibration."""

    index: int
    axis: np.ndarray
    signal: np.ndarray
    timestamp_s: float
    axis_kind: AxisKind = "wavelength_nm"
    signal_kind: SignalKind = "loss_db_per_cm"
    source: str = "unknown"
    metadata: dict[str, Any] | None = None

    def validate(self) -> None:
        axis = np.asarray(self.axis, dtype=np.float64)
        signal = np.asarray(self.signal, dtype=np.float64)
        if axis.ndim != 1 or signal.ndim != 1 or axis.size != signal.size or axis.size < 4:
            raise ValueError("Sensor axis and signal must be one-dimensional, equal-length arrays with >= 4 points.")
        if not np.all(np.isfinite(axis)) or not np.all(np.isfinite(signal)):
            raise ValueError("Sensor frame contains non-finite axis or signal values.")
        if self.axis_kind not in {"wavelength_nm", "pixel"}:
            raise ValueError("axis_kind must be 'wavelength_nm' or 'pixel'.")
        if self.signal_kind not in {"loss_db_per_cm", "intensity"}:
            raise ValueError("signal_kind must be 'loss_db_per_cm' or 'intensity'.")


@dataclass(frozen=True)
class ProcessedSpectrumFrame:
    index: int
    timestamp_s: float
    wavelength_nm: np.ndarray
    loss_db_per_cm: np.ndarray
    source: str
    metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class WavelengthCalibration:
    """Polynomial pixel -> wavelength calibration using ascending coefficients."""

    coefficients_nm: tuple[float, ...]

    def apply(self, pixels: np.ndarray) -> np.ndarray:
        if not self.coefficients_nm:
            raise ValueError("At least one wavelength-calibration coefficient is required.")
        values = np.polynomial.polynomial.polyval(np.asarray(pixels, dtype=np.float64), self.coefficients_nm)
        if not np.all(np.isfinite(values)):
            raise ValueError("Wavelength calibration produced non-finite values.")
        return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class TransmissionCalibration:
    """Dark/reference calibration for converting measured intensity to optical loss."""

    dark: np.ndarray
    reference: np.ndarray
    path_length_cm: float = 1.0
    minimum_transmission: float = 1e-9

    def to_loss_db_per_cm(self, intensity: np.ndarray) -> np.ndarray:
        if self.path_length_cm <= 0:
            raise ValueError("path_length_cm must be > 0.")
        if not 0 < self.minimum_transmission < 1:
            raise ValueError("minimum_transmission must be within (0, 1).")
        measured = np.asarray(intensity, dtype=np.float64)
        dark = np.asarray(self.dark, dtype=np.float64)
        reference = np.asarray(self.reference, dtype=np.float64)
        try:
            numerator = measured - dark
            denominator = reference - dark
        except ValueError as exc:
            raise ValueError("dark/reference calibration arrays are not broadcast-compatible with the spectrum.") from exc
        if np.any(denominator <= 0):
            raise ValueError("reference - dark must be strictly positive at every wavelength.")
        transmission = numerator / denominator
        transmission = np.clip(transmission, self.minimum_transmission, None)
        return (-10.0 * np.log10(transmission) / self.path_length_cm).astype(np.float64)


def _strictly_increasing_axis(axis: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(axis)
    x = np.asarray(axis, dtype=np.float64)[order]
    y = np.asarray(signal, dtype=np.float64)[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("Wavelength axis must contain unique sample locations.")
    return x, y


def resample_spectrum(
    wavelength_nm: np.ndarray,
    values: np.ndarray,
    target_wavelength_nm: np.ndarray,
) -> np.ndarray:
    """Linearly resample without extrapolation onto the edge-model wavelength grid."""
    source_x, source_y = _strictly_increasing_axis(wavelength_nm, values)
    target = np.asarray(target_wavelength_nm, dtype=np.float64)
    if target.ndim != 1 or target.size < 4 or not np.all(np.isfinite(target)):
        raise ValueError("target_wavelength_nm must be a finite one-dimensional array with >= 4 points.")
    if np.any(np.diff(target) <= 0):
        raise ValueError("target_wavelength_nm must be strictly increasing.")
    tolerance = max(1e-9, 1e-9 * max(abs(source_x[0]), abs(source_x[-1]), 1.0))
    if target[0] < source_x[0] - tolerance or target[-1] > source_x[-1] + tolerance:
        raise ValueError("Target wavelength grid extends outside the measured sensor range; extrapolation is disabled.")
    return np.interp(target, source_x, source_y).astype(np.float32)


class SpectrumPreprocessor:
    """Apply wavelength calibration, optical calibration, and model-grid resampling."""

    def __init__(
        self,
        target_wavelength_nm: np.ndarray,
        wavelength_calibration: WavelengthCalibration | None = None,
        transmission_calibration: TransmissionCalibration | None = None,
    ) -> None:
        target = np.asarray(target_wavelength_nm, dtype=np.float64)
        if target.ndim != 1 or target.size < 4 or np.any(np.diff(target) <= 0):
            raise ValueError("target_wavelength_nm must be a strictly increasing one-dimensional grid.")
        self.target_wavelength_nm = target
        self.wavelength_calibration = wavelength_calibration
        self.transmission_calibration = transmission_calibration

    def process(self, frame: RawSpectrumFrame) -> ProcessedSpectrumFrame:
        frame.validate()
        if frame.axis_kind == "pixel":
            if self.wavelength_calibration is None:
                raise ValueError("Pixel-axis sensor frames require wavelength_calibration.")
            wavelength_nm = self.wavelength_calibration.apply(frame.axis)
        else:
            wavelength_nm = np.asarray(frame.axis, dtype=np.float64)

        if frame.signal_kind == "intensity":
            if self.transmission_calibration is None:
                raise ValueError("Intensity sensor frames require transmission_calibration.")
            loss = self.transmission_calibration.to_loss_db_per_cm(frame.signal)
        else:
            loss = np.asarray(frame.signal, dtype=np.float64)

        resampled = resample_spectrum(wavelength_nm, loss, self.target_wavelength_nm)
        return ProcessedSpectrumFrame(
            index=frame.index,
            timestamp_s=float(frame.timestamp_s),
            wavelength_nm=self.target_wavelength_nm.astype(np.float32),
            loss_db_per_cm=resampled,
            source=frame.source,
            metadata=frame.metadata,
        )


class ArraySpectrumSource:
    """Deterministic in-memory source used for CI and hardware-driver contract tests."""

    def __init__(self, frames: Sequence[RawSpectrumFrame]) -> None:
        self.frames = tuple(frames)

    def __iter__(self) -> Iterator[RawSpectrumFrame]:
        yield from self.frames


class JSONLineSpectrumSource:
    """Read the documented newline-delimited JSON sensor protocol from a file."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def __iter__(self) -> Iterator[RawSpectrumFrame]:
        with self.path.open("r", encoding="utf-8") as handle:
            for fallback_index, line in enumerate(handle):
                payload = line.strip()
                if not payload:
                    continue
                yield frame_from_json(json.loads(payload), fallback_index=fallback_index, source=str(self.path))


class SerialJSONLineSource:
    """Read newline-delimited JSON sensor frames from a serial device.

    pyserial is imported lazily so software-only CI does not require hardware dependencies.
    """

    def __init__(self, port: str, baudrate: int = 115200, timeout_s: float = 2.0) -> None:
        if baudrate <= 0 or timeout_s <= 0:
            raise ValueError("baudrate and timeout_s must be > 0.")
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("Serial acquisition requires the 'hardware' extra: pip install -e '.[hardware]'.") from exc
        self._serial = serial.Serial(port=port, baudrate=baudrate, timeout=timeout_s)
        self.port = port

    def __iter__(self) -> Iterator[RawSpectrumFrame]:
        index = 0
        while True:
            line = self._serial.readline()
            if not line:
                continue
            payload = json.loads(line.decode("utf-8"))
            yield frame_from_json(payload, fallback_index=index, source=f"serial:{self.port}")
            index += 1

    def close(self) -> None:
        self._serial.close()


def frame_from_json(payload: dict[str, Any], fallback_index: int = 0, source: str = "json") -> RawSpectrumFrame:
    if "axis" not in payload or "signal" not in payload:
        raise ValueError("Sensor JSON frame must contain 'axis' and 'signal' arrays.")
    frame = RawSpectrumFrame(
        index=int(payload.get("index", fallback_index)),
        axis=np.asarray(payload["axis"], dtype=np.float64),
        signal=np.asarray(payload["signal"], dtype=np.float64),
        timestamp_s=float(payload.get("timestamp_s", time.time())),
        axis_kind=str(payload.get("axis_kind", "wavelength_nm")),  # type: ignore[arg-type]
        signal_kind=str(payload.get("signal_kind", "loss_db_per_cm")),  # type: ignore[arg-type]
        source=source,
        metadata=dict(payload.get("metadata", {})),
    )
    frame.validate()
    return frame


def _normalize_spectrum(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    spectrum = np.asarray(values, dtype=np.float32)
    mean = float(np.mean(spectrum))
    std = float(np.std(spectrum) + 1e-6)
    return ((spectrum - mean) / std).astype(np.float32), mean, std


@dataclass(frozen=True)
class SpectralOODDetector:
    """Reference-distribution spectral OOD detector with a normalized threshold score."""

    center: np.ndarray
    scale: np.ndarray
    threshold: float
    coverage: float

    @classmethod
    def fit(cls, normalized_spectra: np.ndarray, coverage: float = 0.99) -> "SpectralOODDetector":
        data = np.asarray(normalized_spectra, dtype=np.float64)
        if data.ndim != 2 or data.shape[0] < 3:
            raise ValueError("OOD calibration requires at least three normalized spectra.")
        if not 0.5 < coverage < 1.0:
            raise ValueError("coverage must be within (0.5, 1.0).")
        center = data.mean(axis=0)
        scale = data.std(axis=0) + 1e-6
        raw = np.sqrt(np.mean(np.square((data - center) / scale), axis=1))
        threshold = float(np.quantile(raw, coverage))
        return cls(center=center, scale=scale, threshold=max(threshold, 1e-9), coverage=float(coverage))

    def score(self, normalized_spectrum: np.ndarray) -> float:
        sample = np.asarray(normalized_spectrum, dtype=np.float64).reshape(-1)
        if sample.shape != self.center.shape:
            raise ValueError("Spectrum length does not match OOD calibration.")
        raw = float(np.sqrt(np.mean(np.square((sample - self.center) / self.scale))))
        return raw / self.threshold

    def to_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.tolist(),
            "scale": self.scale.tolist(),
            "threshold": self.threshold,
            "coverage": self.coverage,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SpectralOODDetector":
        return cls(
            center=np.asarray(payload["center"], dtype=np.float64),
            scale=np.asarray(payload["scale"], dtype=np.float64),
            threshold=float(payload["threshold"]),
            coverage=float(payload["coverage"]),
        )


@dataclass(frozen=True)
class PredictionIntervalCalibration:
    """Split-calibration absolute-error intervals for RI and resonance prediction."""

    coverage: float
    ri_margin: float
    lambda_margin_nm: float

    @classmethod
    def fit(
        cls,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        coverage: float = 0.95,
    ) -> "PredictionIntervalCalibration":
        truth = np.asarray(y_true, dtype=np.float64)
        pred = np.asarray(y_pred, dtype=np.float64)
        if truth.shape != pred.shape or truth.ndim != 2 or truth.shape[1] < 2 or truth.shape[0] < 3:
            raise ValueError("Prediction calibration requires matching [N, >=2] arrays with N >= 3.")
        if not 0.5 < coverage < 1.0:
            raise ValueError("coverage must be within (0.5, 1.0).")
        errors = np.abs(truth[:, :2] - pred[:, :2])
        return cls(
            coverage=float(coverage),
            ri_margin=float(np.quantile(errors[:, 0], coverage)),
            lambda_margin_nm=float(np.quantile(errors[:, 1], coverage)),
        )

    def intervals(self, predicted_ri: float, predicted_lambda_nm: float) -> dict[str, float]:
        return {
            "ri_lower": float(predicted_ri - self.ri_margin),
            "ri_upper": float(predicted_ri + self.ri_margin),
            "lambda_lower_nm": float(predicted_lambda_nm - self.lambda_margin_nm),
            "lambda_upper_nm": float(predicted_lambda_nm + self.lambda_margin_nm),
        }

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PredictionIntervalCalibration":
        return cls(
            coverage=float(payload["coverage"]),
            ri_margin=float(payload["ri_margin"]),
            lambda_margin_nm=float(payload["lambda_margin_nm"]),
        )


@dataclass(frozen=True)
class EdgeCalibrationBundle:
    spectral_ood: SpectralOODDetector | None = None
    prediction_interval: PredictionIntervalCalibration | None = None

    def save(self, path: Path) -> None:
        payload = {
            "schema_version": 1,
            "spectral_ood": self.spectral_ood.to_dict() if self.spectral_ood is not None else None,
            "prediction_interval": (
                self.prediction_interval.to_dict() if self.prediction_interval is not None else None
            ),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "EdgeCalibrationBundle":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        ood_payload = payload.get("spectral_ood")
        interval_payload = payload.get("prediction_interval")
        return cls(
            spectral_ood=SpectralOODDetector.from_dict(ood_payload) if ood_payload else None,
            prediction_interval=PredictionIntervalCalibration.from_dict(interval_payload) if interval_payload else None,
        )


class LiteRTModel:
    """Lazy LiteRT model adapter for the hardware runtime."""

    def __init__(self, model_path: Path) -> None:
        from sprpcf.edge.quantization import TFLiteModelRunner

        self.runner = TFLiteModelRunner(Path(model_path))

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return self.runner.predict(inputs)


class KerasModel:
    """Lazy Keras model adapter, useful for workstation validation before INT8 export."""

    def __init__(self, model_path: Path) -> None:
        import tensorflow as tf

        self.model = tf.keras.models.load_model(model_path, compile=False)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(inputs, verbose=0), dtype=np.float32)


@dataclass(frozen=True)
class SensorInference:
    index: int
    timestamp_s: float
    source: str
    measured_lambda_res_nm: float
    measured_peak_loss_db_per_cm: float
    predicted_ri: float | None
    predicted_lambda_res_nm: float | None
    denoiser_correction_rmse: float
    spectral_ood_score: float | None
    in_distribution: bool | None
    latency_ms: float
    intervals: dict[str, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LiveSensorPipeline:
    """Calibrate, resample, denoise, infer RI, and attach OOD/calibrated intervals."""

    def __init__(
        self,
        preprocessor: SpectrumPreprocessor,
        denoiser: PredictiveModel,
        predictor: PredictiveModel | None = None,
        calibration: EdgeCalibrationBundle | None = None,
    ) -> None:
        self.preprocessor = preprocessor
        self.denoiser = denoiser
        self.predictor = predictor
        self.calibration = calibration or EdgeCalibrationBundle()

    def process(self, raw_frame: RawSpectrumFrame) -> SensorInference:
        started = time.perf_counter()
        frame = self.preprocessor.process(raw_frame)
        normalized, mean, std = _normalize_spectrum(frame.loss_db_per_cm)
        denoised_normalized = np.asarray(
            self.denoiser.predict(normalized[None, :, None])[0, :, 0],
            dtype=np.float32,
        )
        physical_denoised = denoised_normalized * std + mean
        measured_lambda, peak_loss = resonance_wavelength(frame.wavelength_nm, physical_denoised)

        predicted_ri: float | None = None
        predicted_lambda: float | None = None
        intervals: dict[str, float] | None = None
        if self.predictor is not None:
            prediction = np.asarray(self.predictor.predict(denoised_normalized[None, :, None])[0], dtype=np.float64)
            if prediction.size < 2:
                raise ValueError("RI predictor must output at least [analyte_ri, lambda_res_nm].")
            predicted_ri = float(prediction[0])
            predicted_lambda = float(prediction[1])
            if self.calibration.prediction_interval is not None:
                intervals = self.calibration.prediction_interval.intervals(predicted_ri, predicted_lambda)

        ood_score: float | None = None
        in_distribution: bool | None = None
        if self.calibration.spectral_ood is not None:
            ood_score = float(self.calibration.spectral_ood.score(normalized))
            in_distribution = bool(ood_score <= 1.0)

        correction_rmse = float(np.sqrt(np.mean(np.square(denoised_normalized - normalized))))
        latency_ms = (time.perf_counter() - started) * 1000.0
        return SensorInference(
            index=frame.index,
            timestamp_s=frame.timestamp_s,
            source=frame.source,
            measured_lambda_res_nm=float(measured_lambda),
            measured_peak_loss_db_per_cm=float(peak_loss),
            predicted_ri=predicted_ri,
            predicted_lambda_res_nm=predicted_lambda,
            denoiser_correction_rmse=correction_rmse,
            spectral_ood_score=ood_score,
            in_distribution=in_distribution,
            latency_ms=float(latency_ms),
            intervals=intervals,
        )


def _max_rss_mb() -> float | None:
    try:
        import resource
    except ImportError:
        return None
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def benchmark_pipeline(
    pipeline: LiveSensorPipeline,
    frames: Sequence[RawSpectrumFrame],
    iterations: int = 100,
    warmup: int = 5,
) -> dict[str, float | int | None]:
    """Benchmark end-to-end hardware processing with P50/P95/P99 and memory evidence."""
    if not frames:
        raise ValueError("At least one frame is required for benchmarking.")
    if iterations < 1 or warmup < 0:
        raise ValueError("iterations must be >= 1 and warmup must be >= 0.")

    for index in range(warmup):
        pipeline.process(frames[index % len(frames)])

    tracemalloc.start()
    started = time.perf_counter()
    latencies: list[float] = []
    for index in range(iterations):
        result = pipeline.process(frames[index % len(frames)])
        latencies.append(result.latency_ms)
    elapsed = max(time.perf_counter() - started, 1e-12)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    values = np.asarray(latencies, dtype=np.float64)
    return {
        "iterations": int(iterations),
        "latency_ms_p50": float(np.percentile(values, 50)),
        "latency_ms_p95": float(np.percentile(values, 95)),
        "latency_ms_p99": float(np.percentile(values, 99)),
        "latency_ms_mean": float(np.mean(values)),
        "throughput_fps": float(iterations / elapsed),
        "python_heap_peak_mb": float(peak_bytes / (1024.0 * 1024.0)),
        "process_max_rss_mb": _max_rss_mb(),
    }


def write_inference_jsonl(results: Iterable[SensorInference], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")


def parse_csv_floats(value: str) -> tuple[float, ...]:
    parsed = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not parsed or any(not math.isfinite(item) for item in parsed):
        raise ValueError("Expected a comma-separated list of finite numbers.")
    return parsed
