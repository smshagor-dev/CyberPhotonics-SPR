from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def representative_dataset(samples: np.ndarray, limit: int = 128) -> Iterable[list[np.ndarray]]:
    """Yield shape-correct representative batches for full INT8 calibration."""
    calibration = np.asarray(samples[: min(limit, samples.shape[0])], dtype=np.float32)
    for sample in calibration:
        if sample.ndim == 1:
            sample = sample[:, None]
        yield [sample[None, ...]]


def convert_model_to_int8_tflite(
    model: tf.keras.Model,
    output_path: Path,
    calibration_samples: np.ndarray,
    inference_type: tf.dtypes.DType = tf.int8,
) -> None:
    """Export a Keras model using full integer post-training quantization."""
    samples = np.asarray(calibration_samples)
    if samples.ndim not in (2, 3) or samples.shape[0] < 1:
        raise ValueError("calibration_samples must have shape [N, L] or [N, L, C].")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: representative_dataset(samples)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = inference_type
    converter.inference_output_type = inference_type
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def run_tflite_inference(model_path: Path, inputs: np.ndarray) -> np.ndarray:
    return TFLiteModelRunner(model_path).predict(inputs)


class TFLiteModelRunner:
    """Reusable TFLite interpreter wrapper with quantization and dynamic-batch handling."""

    def __init__(self, model_path: Path) -> None:
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self._refresh_details()

    def _refresh_details(self) -> None:
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

    @staticmethod
    def _quantize(values: np.ndarray, details: dict) -> np.ndarray:
        scale, zero_point = details["quantization"]
        if scale <= 0:
            raise ValueError("Quantized TFLite tensor has invalid scale <= 0.")
        quantized = np.round(values / scale + zero_point)
        limits = np.iinfo(details["dtype"])
        return np.clip(quantized, limits.min, limits.max).astype(details["dtype"])

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        input_data = np.asarray(inputs, dtype=np.float32)
        expected_shape = tuple(int(value) for value in self.input_details["shape"])
        if expected_shape != input_data.shape:
            signature = tuple(int(value) for value in self.input_details.get("shape_signature", expected_shape))
            can_resize = len(signature) == input_data.ndim and all(
                expected == actual or expected == -1 for expected, actual in zip(signature, input_data.shape)
            )
            if not can_resize:
                raise ValueError(f"TFLite input shape {input_data.shape} is incompatible with {signature}.")
            self.interpreter.resize_tensor_input(self.input_details["index"], input_data.shape, strict=False)
            self.interpreter.allocate_tensors()
            self._refresh_details()

        if self.input_details["dtype"] in (np.int8, np.uint8):
            input_data = self._quantize(input_data, self.input_details)
        else:
            input_data = input_data.astype(self.input_details["dtype"])

        self.interpreter.set_tensor(self.input_details["index"], input_data)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])

        if self.output_details["dtype"] in (np.int8, np.uint8):
            scale, zero_point = self.output_details["quantization"]
            if scale <= 0:
                raise ValueError("Quantized TFLite output tensor has invalid scale <= 0.")
            output = (output.astype(np.float32) - zero_point) * scale
        return output.astype(np.float32)
