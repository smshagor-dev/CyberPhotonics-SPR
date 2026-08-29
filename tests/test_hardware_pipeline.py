from __future__ import annotations

import json

import numpy as np
import pytest

from sprpcf.edge.hardware import (
    ArraySpectrumSource,
    EdgeCalibrationBundle,
    LiveSensorPipeline,
    PredictionIntervalCalibration,
    RawSpectrumFrame,
    SpectralOODDetector,
    SpectrumPreprocessor,
    TransmissionCalibration,
    WavelengthCalibration,
    benchmark_pipeline,
    frame_from_json,
    resample_spectrum,
)


class IdentityModel:
    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.asarray(inputs, dtype=np.float32)


class ConstantPredictor:
    def __init__(self, ri: float = 1.36, wavelength_nm: float = 650.0) -> None:
        self.output = np.asarray([ri, wavelength_nm], dtype=np.float32)

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        return np.repeat(self.output[None, :], inputs.shape[0], axis=0)


def _frame(signal: np.ndarray, axis: np.ndarray | None = None, signal_kind: str = "loss_db_per_cm"):
    if axis is None:
        axis = np.linspace(600.0, 700.0, signal.size)
    return RawSpectrumFrame(
        index=0,
        axis=np.asarray(axis, dtype=float),
        signal=np.asarray(signal, dtype=float),
        timestamp_s=1.0,
        axis_kind="wavelength_nm",
        signal_kind=signal_kind,  # type: ignore[arg-type]
        source="test",
    )


def test_wavelength_calibration_and_resampling() -> None:
    pixels = np.arange(5, dtype=float)
    calibration = WavelengthCalibration((500.0, 2.0))
    wavelength = calibration.apply(pixels)
    assert np.allclose(wavelength, [500.0, 502.0, 504.0, 506.0, 508.0])

    values = np.asarray([0.0, 1.0, 4.0, 9.0, 16.0])
    resampled = resample_spectrum(wavelength, values, np.asarray([501.0, 503.0, 505.0, 507.0]))
    assert np.allclose(resampled, [0.5, 2.5, 6.5, 12.5])


def test_resampling_rejects_extrapolation() -> None:
    with pytest.raises(ValueError, match="extrapolation"):
        resample_spectrum(
            np.asarray([500.0, 510.0, 520.0, 530.0]),
            np.ones(4),
            np.asarray([490.0, 500.0, 510.0, 520.0]),
        )


def test_intensity_dark_reference_to_loss() -> None:
    dark = np.asarray([10.0, 10.0, 10.0, 10.0])
    reference = np.asarray([110.0, 110.0, 110.0, 110.0])
    measured = np.asarray([60.0, 60.0, 60.0, 60.0])
    calibration = TransmissionCalibration(dark=dark, reference=reference, path_length_cm=2.0)
    loss = calibration.to_loss_db_per_cm(measured)
    expected = -10.0 * np.log10(0.5) / 2.0
    assert np.allclose(loss, expected)


def test_json_frame_contract_and_array_source() -> None:
    payload = {
        "index": 8,
        "timestamp_s": 12.5,
        "axis_kind": "wavelength_nm",
        "signal_kind": "loss_db_per_cm",
        "axis": [600.0, 610.0, 620.0, 630.0],
        "signal": [1.0, 2.0, 3.0, 2.0],
        "metadata": {"device": "mock"},
    }
    frame = frame_from_json(json.loads(json.dumps(payload)), source="unit")
    assert frame.index == 8
    assert frame.metadata == {"device": "mock"}
    frames = list(ArraySpectrumSource([frame]))
    assert len(frames) == 1 and frames[0].index == 8


def test_preprocessor_pixel_and_intensity_path() -> None:
    target = np.asarray([500.0, 502.0, 504.0, 506.0])
    raw = RawSpectrumFrame(
        index=1,
        axis=np.arange(4, dtype=float),
        signal=np.asarray([60.0, 60.0, 60.0, 60.0]),
        timestamp_s=1.0,
        axis_kind="pixel",
        signal_kind="intensity",
        source="spectrometer",
    )
    preprocessor = SpectrumPreprocessor(
        target,
        wavelength_calibration=WavelengthCalibration((500.0, 2.0)),
        transmission_calibration=TransmissionCalibration(
            dark=np.full(4, 10.0),
            reference=np.full(4, 110.0),
        ),
    )
    processed = preprocessor.process(raw)
    assert np.allclose(processed.wavelength_nm, target)
    assert np.all(np.isfinite(processed.loss_db_per_cm))


def test_ood_detector_and_prediction_intervals() -> None:
    reference = np.asarray(
        [
            [-1.0, -0.5, 0.5, 1.0],
            [-0.9, -0.4, 0.4, 0.9],
            [-1.1, -0.6, 0.6, 1.1],
            [-1.0, -0.45, 0.45, 1.0],
        ]
    )
    detector = SpectralOODDetector.fit(reference, coverage=0.9)
    assert detector.score(reference[0]) < detector.score(np.asarray([8.0, 8.0, 8.0, 8.0]))

    truth = np.asarray([[1.33, 620.0], [1.34, 630.0], [1.35, 640.0], [1.36, 650.0]])
    pred = truth + np.asarray([[0.001, 1.0], [-0.002, -2.0], [0.003, 3.0], [-0.001, 1.5]])
    intervals = PredictionIntervalCalibration.fit(truth, pred, coverage=0.75)
    bounds = intervals.intervals(1.35, 640.0)
    assert bounds["ri_lower"] < 1.35 < bounds["ri_upper"]
    assert bounds["lambda_lower_nm"] < 640.0 < bounds["lambda_upper_nm"]


def test_live_pipeline_and_benchmark() -> None:
    wavelength = np.linspace(600.0, 700.0, 16)
    loss = 1.0 + 4.0 * np.exp(-0.5 * ((wavelength - 650.0) / 6.0) ** 2)
    normalized = (loss - loss.mean()) / (loss.std() + 1e-6)
    reference = np.stack([normalized * factor for factor in (0.98, 1.0, 1.01, 1.02)], axis=0)
    calibration = EdgeCalibrationBundle(
        spectral_ood=SpectralOODDetector.fit(reference, coverage=0.9),
        prediction_interval=PredictionIntervalCalibration(
            coverage=0.95,
            ri_margin=0.01,
            lambda_margin_nm=5.0,
        ),
    )
    pipeline = LiveSensorPipeline(
        SpectrumPreprocessor(wavelength),
        denoiser=IdentityModel(),
        predictor=ConstantPredictor(),
        calibration=calibration,
    )
    frame = _frame(loss, wavelength)
    result = pipeline.process(frame)
    assert abs(result.measured_lambda_res_nm - 650.0) < 5.0
    assert result.predicted_ri == pytest.approx(1.36)
    assert result.predicted_lambda_res_nm == pytest.approx(650.0)
    assert result.intervals is not None
    assert result.spectral_ood_score is not None
    assert result.latency_ms >= 0.0

    stats = benchmark_pipeline(pipeline, [frame], iterations=4, warmup=1)
    assert stats["iterations"] == 4
    assert stats["latency_ms_p99"] >= 0.0
    assert stats["throughput_fps"] > 0.0
    assert stats["python_heap_peak_mb"] >= 0.0
