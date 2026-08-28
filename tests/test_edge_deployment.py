from __future__ import annotations

import importlib.util

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


def test_int8_tflite_export_has_valid_flatbuffer_and_accuracy(tmp_path) -> None:
    from sprpcf.edge.quantization import convert_model_to_int8_tflite, run_tflite_inference

    rng = np.random.default_rng(3)
    samples = rng.normal(0.0, 0.4, (12, 16)).astype(np.float32)
    output_path = tmp_path / "edge_denoiser_quantized.tflite"

    convert_model_to_int8_tflite(_identity_conv_model(16), output_path, samples)
    assert output_path.exists()
    assert output_path.stat().st_size > 1024
    assert output_path.read_bytes()[4:8] == b"TFL3"

    prediction = run_tflite_inference(output_path, samples[:1, :, None])
    assert prediction.shape == (1, 16, 1)
    assert np.mean(np.abs(prediction[0, :, 0] - samples[0])) < 0.05


def test_sensor_feed_simulator_reports_latency(tmp_path) -> None:
    from sprpcf.edge.quantization import convert_model_to_int8_tflite
    from sprpcf.edge.simulator import SensorFeedSimulator

    clean = np.zeros((4, 16), dtype=np.float32)
    output_path = tmp_path / "edge_denoiser_quantized.tflite"
    convert_model_to_int8_tflite(_identity_conv_model(16), output_path, clean + 0.1)

    simulator = SensorFeedSimulator(clean, output_path, noise_std=0.01, drift_std=0.0)
    frames = list(simulator.stream(frames=2))
    assert len(frames) == 2
    assert all(frame.latency_ms >= 0.0 for frame in frames)
    assert frames[0].denoised_spectrum.shape == (16,)


def test_int8_cli_requires_calibration_data(tmp_path) -> None:
    from sprpcf.edge.export_tflite import convert_to_tflite

    with pytest.raises(ValueError, match="calibration-data"):
        convert_to_tflite(tmp_path / "model.keras", tmp_path / "model.tflite", "int8", None)
