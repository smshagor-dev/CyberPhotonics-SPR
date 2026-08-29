from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from sprpcf.edge.quantization import convert_model_to_int8_tflite
from sprpcf.edge.train_denoiser import normalize_spectra, parse_spectra
from sprpcf.ml.dataset import read_table


def convert_to_tflite(model_path: Path, output_path: Path, quantization: str, calibration_data: Path | None = None) -> None:
    """Convert Keras to TFLite without silently downgrading requested INT8 mode."""
    if quantization == "int8" and calibration_data is None:
        raise ValueError("Full INT8 quantization requires --calibration-data.")
    model = tf.keras.models.load_model(model_path, compile=False)
    if quantization == "int8":
        frame = read_table(calibration_data).dropna(subset=["loss_db_per_cm"])
        spectra, _, _ = normalize_spectra(parse_spectra(frame))
        convert_model_to_int8_tflite(model, output_path, spectra)
        return

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantization == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(tflite_model)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export denoising model to TensorFlow Lite.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quantization", choices=["none", "fp16", "int8"], default="fp16")
    parser.add_argument("--calibration-data", type=Path, default=None)
    args = parser.parse_args()
    convert_to_tflite(args.model, args.out, args.quantization, args.calibration_data)


if __name__ == "__main__":
    main()
