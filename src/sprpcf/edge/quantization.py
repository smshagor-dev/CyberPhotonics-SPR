from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def representative_dataset(samples: np.ndarray, limit: int = 128) -> Iterable[list[np.ndarray]]:
    """Yield representative spectra for full INT8 TFLite calibration."""
    calibration = samples[: min(limit, samples.shape[0])].astype(np.float32)
    for sample in calibration:
        yield [sample[None, :, None]]


def convert_model_to_int8_tflite(
    model: tf.keras.Model,
    output_path: Path,
    calibration_samples: np.ndarray,
    inference_type: tf.dtypes.DType = tf.int8,
) -> None:
    """Export a Keras model using full integer post-training quantization."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(calibration_samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = inference_type
    converter.inference_output_type = inference_type
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def run_tflite_inference(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    """Run one batch through a quantized or float TFLite model."""
    return TFLiteModelRunner(model_path).predict(inputs)


class TFLiteModelRunner:
    """Reusable TFLite interpreter wrapper for edge-feed loops."""

    def __init__(self, model_path: Path) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        input_data = inputs.astype(np.float32)

        if self.input_details["dtype"] in (np.int8, np.uint8):
            scale, zero_point = self.input_details["quantization"]
            input_data = np.round(input_data / scale + zero_point).astype(self.input_details["dtype"])
        else:
            input_data = input_data.astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])

        if self.output_details["dtype"] in (np.int8, np.uint8):
            scale, zero_point = self.output_details["quantization"]
            output = (output.astype(np.float32) - zero_point) * scale
        return output.astype(np.float32)
