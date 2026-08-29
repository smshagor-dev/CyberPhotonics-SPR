from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="tensorflow is not installed")


def _identity_conv_model(spectrum_length: int):
    import tensorflow as tf

    inputs = tf.keras.Input(shape=(spectrum_length, 1), name="spectrum")
    outputs = tf.keras.layers.Conv1D(1, 1, use_bias=False, name="identity_conv")(inputs)
    model = tf.keras.Model(inputs, outputs)
    model.get_layer("identity_conv").set_weights([np.ones((1, 1, 1), dtype=np.float32)])
    return model


def test_hil_streaming_buffer_stability() -> None:
    from sprpcf.edge.hil_engine import HILBenchmarkEngine

    clean = np.tile(np.linspace(-1.0, 1.0, 16, dtype=np.float32), (12, 1))
    engine = HILBenchmarkEngine(clean, buffer_size=4, seed=2)

    frames = engine.collect_stream(duration_sec=1.0, fps=10.0, inject_thermal_drift=False)

    assert len(frames) == 10
    assert engine.frames_received == 10
    assert engine.frames_lost == 0
    assert len(engine.buffer) == 4
    assert [frame.sequence_id for frame in engine.buffer] == [6, 7, 8, 9]


def test_latency_profile_has_zero_frame_loss(tmp_path) -> None:
    from sprpcf.edge.hil_engine import HILBenchmarkEngine
    from sprpcf.edge.quantization import convert_model_to_int8_tflite

    clean = np.zeros((8, 16), dtype=np.float32)
    tflite_dir = tmp_path / "models"
    model_path = tflite_dir / "edge_denoiser_quantized.tflite"
    convert_model_to_int8_tflite(_identity_conv_model(16), model_path, clean + 0.1)

    engine = HILBenchmarkEngine(clean, buffer_size=16, seed=4)
    report = engine.benchmark(
        tflite_dir=tflite_dir,
        duration_sec=0.2,
        fps=10.0,
        report_path=tmp_path / "phase4_hil_benchmark.json",
    )

    assert report.frames_received == 2
    assert report.frames_lost == 0
    assert report.models["int8"]["frames_processed"] == 2
    assert report.models["int8"]["frames_lost"] == 0
    assert report.models["int8"]["average_latency_ms"] >= 0.0


def test_thermal_drift_evaluation_and_report_output(tmp_path) -> None:
    from sprpcf.edge.hil_engine import HILBenchmarkEngine
    from sprpcf.edge.quantization import convert_model_to_int8_tflite

    clean = np.tile(np.linspace(-0.5, 0.5, 16, dtype=np.float32), (16, 1))
    tflite_dir = tmp_path / "models"
    model_path = tflite_dir / "edge_denoiser_quantized.tflite"
    convert_model_to_int8_tflite(_identity_conv_model(16), model_path, clean + 0.1)

    output_path = tmp_path / "reports" / "phase4_hil_benchmark.json"
    engine = HILBenchmarkEngine(clean, buffer_size=32, seed=6)
    report = engine.benchmark(
        tflite_dir=tflite_dir,
        duration_sec=0.5,
        fps=8.0,
        inject_thermal_drift=True,
        report_path=output_path,
    )

    assert report.thermal_drift.enabled is True
    assert report.thermal_drift.temperature_start_c == pytest.approx(20.0)
    assert report.thermal_drift.temperature_end_c > report.thermal_drift.temperature_start_c
    assert report.thermal_drift.max_baseline_drift > 0.0
    assert 0.0 <= report.thermal_drift.tolerance_score <= 1.0
    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["thermal_drift"]["enabled"] is True
    assert "int8" in payload["models"]
