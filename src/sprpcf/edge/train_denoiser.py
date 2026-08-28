from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import warnings

import numpy as np
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit

from sprpcf.edge.denoising import build_denoising_autoencoder, build_ri_predictor
from sprpcf.edge.quantization import TFLiteModelRunner, convert_model_to_int8_tflite
from sprpcf.ml.dataset import GEOMETRY_COLUMNS, geometry_group_labels, read_table
from sprpcf.utils.reproducibility import seed_everything


def parse_spectra(frame) -> np.ndarray:
    """Parse comma-separated spectral loss vectors and reject ragged/invalid data."""
    spectra = frame["loss_db_per_cm"].map(lambda value: np.fromstring(str(value), sep=",")).to_list()
    lengths = {spectrum.size for spectrum in spectra}
    if not spectra or len(lengths) != 1 or 0 in lengths:
        raise ValueError("loss_db_per_cm must contain non-empty spectra with a common wavelength length.")
    array = np.asarray(spectra, dtype=np.float32)
    if not np.all(np.isfinite(array)):
        raise ValueError("Spectra contain non-finite values.")
    return array


def normalize_spectra(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True) + 1e-6
    return ((spectra - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def add_sensor_noise(
    clean: np.ndarray,
    noise_std: float = 0.08,
    drift_std: float = 0.03,
    seed: int = 7,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    wavelength_axis = np.linspace(-1.0, 1.0, clean.shape[1], dtype=np.float32)
    thermal = rng.normal(0.0, noise_std, clean.shape).astype(np.float32)
    drift = rng.normal(0.0, drift_std, (clean.shape[0], 1)).astype(np.float32) * wavelength_axis
    ripple = 0.015 * np.sin(np.linspace(0.0, 8.0 * np.pi, clean.shape[1], dtype=np.float32))[None, :]
    return (clean + thermal + drift + ripple).astype(np.float32)


def train_val_group_split(frame, *arrays: np.ndarray, val_fraction: float = 0.2, seed: int = 7) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split by base geometry to prevent RI-sweep leakage into validation."""
    groups = geometry_group_labels(frame)
    if np.unique(groups).size < 2:
        raise ValueError("At least two unique geometries are required for edge validation.")
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, groups=groups))
    return [array[train_idx] for array in arrays], [array[val_idx] for array in arrays]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def psnr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error = mse(y_true, y_pred)
    if error <= 1e-12:
        return float("inf")
    data_range = float(np.max(y_true) - np.min(y_true))
    return float(20.0 * np.log10(max(data_range, 1e-6)) - 10.0 * np.log10(error))


def ssim_1d(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    true = y_true.reshape(y_true.shape[0], -1)
    pred = y_pred.reshape(y_pred.shape[0], -1)
    c1 = 0.01**2
    c2 = 0.03**2
    mu_x = true.mean(axis=1)
    mu_y = pred.mean(axis=1)
    var_x = true.var(axis=1)
    var_y = pred.var(axis=1)
    cov_xy = ((true - mu_x[:, None]) * (pred - mu_y[:, None])).mean(axis=1)
    score = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(np.mean(score))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.sum(np.square(y_true - y_pred), axis=0)
    total = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)
    scores = 1.0 - residual / np.maximum(total, 1e-12)
    return float(np.mean(scores))


def r2_columns_np(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    residual = np.sum(np.square(y_true - y_pred), axis=0)
    total = np.sum(np.square(y_true - np.mean(y_true, axis=0)), axis=0)
    return 1.0 - residual / np.maximum(total, 1e-12)


def weighted_regression_loss(target_scale: np.ndarray):
    scale = tf.constant(target_scale.reshape(1, -1), dtype=tf.float32)

    def loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        return tf.reduce_mean(tf.square((y_true - y_pred) / scale))

    return loss


def configure_device(device: str) -> str:
    if device == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError as exc:
            raise RuntimeError("TensorFlow device visibility must be configured before GPU initialization.") from exc
        return "/CPU:0"
    if device == "auto":
        return "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"
    return device


def _tflite_predict_many(runner: TFLiteModelRunner, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    latencies: list[float] = []
    for sample in inputs:
        started = time.perf_counter()
        outputs.append(runner.predict(sample[None, ...])[0])
        latencies.append((time.perf_counter() - started) * 1000.0)
    return np.asarray(outputs, dtype=np.float32), np.asarray(latencies, dtype=np.float64)


def _save_keras_model_without_external_numpy_warning(model: tf.keras.Model, output_path: Path) -> None:
    """Save Keras model while isolating a NumPy-2/Keras dependency deprecation."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"__array__ implementation doesn't accept a copy keyword.*",
            category=DeprecationWarning,
            module=r"keras\.src\.backend\.tensorflow\.core",
        )
        model.save(output_path, include_optimizer=False)


def train_edge_models(
    data_path: Path,
    denoiser_out: Path,
    predictor_out: Path,
    epochs: int,
    batch_size: int,
    device: str,
    quantize: bool,
    denoiser_tflite_out: Path,
    predictor_tflite_out: Path,
    seed: int = 7,
) -> dict[str, float]:
    """Train edge models and, when requested, validate the actual INT8 artifacts."""
    device_scope = configure_device(device)
    seed_everything(seed, include_tensorflow=True)
    required = ["loss_db_per_cm", "analyte_ri", "lambda_res_nm", *GEOMETRY_COLUMNS]
    frame = read_table(data_path).dropna(subset=required).reset_index(drop=True)
    clean_raw = parse_spectra(frame)
    clean, _, _ = normalize_spectra(clean_raw)
    noisy = add_sensor_noise(clean, seed=seed)
    targets = frame[["analyte_ri", "lambda_res_nm"]].to_numpy(np.float32)

    train_arrays, val_arrays = train_val_group_split(frame, noisy, clean, targets, seed=seed)
    noisy_train, clean_train, target_train = train_arrays
    noisy_val, clean_val, target_val = val_arrays

    with tf.device(device_scope):
        denoiser = build_denoising_autoencoder(clean.shape[1])
        denoiser.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
        denoiser.fit(
            noisy_train[..., None], clean_train[..., None],
            validation_data=(noisy_val[..., None], clean_val[..., None]),
            epochs=epochs, batch_size=batch_size, verbose=2,
        )

        denoised_train = denoiser.predict(noisy_train[..., None], verbose=0)
        denoised_val = denoiser.predict(noisy_val[..., None], verbose=0)
        target_scale = target_train.std(axis=0) + 1e-6
        predictor = build_ri_predictor(clean.shape[1])
        predictor.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss=weighted_regression_loss(target_scale), metrics=["mae"],
        )
        predictor.fit(
            denoised_train, target_train,
            validation_data=(denoised_val, target_val),
            epochs=epochs, batch_size=batch_size, verbose=2,
        )

    denoiser_out.parent.mkdir(parents=True, exist_ok=True)
    predictor_out.parent.mkdir(parents=True, exist_ok=True)
    _save_keras_model_without_external_numpy_warning(denoiser, denoiser_out)
    predictor.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    _save_keras_model_without_external_numpy_warning(predictor, predictor_out)

    denoised_prediction = denoiser.predict(noisy_val[..., None], verbose=0)
    target_prediction = predictor.predict(denoised_prediction, verbose=0)
    target_r2 = r2_columns_np(target_val, target_prediction)
    metrics: dict[str, float] = {
        "denoising_mse": mse(clean_val[..., None], denoised_prediction),
        "denoising_psnr": psnr(clean_val[..., None], denoised_prediction),
        "denoising_ssim": ssim_1d(clean_val[..., None], denoised_prediction),
        "ri_mae": mae(target_val[:, 0], target_prediction[:, 0]),
        "ri_r2": float(target_r2[0]),
        "lambda_res_mae_nm": mae(target_val[:, 1], target_prediction[:, 1]),
        "lambda_res_r2": float(target_r2[1]),
        "ri_lambda_r2_mean": r2_score_np(target_val, target_prediction),
    }

    if quantize:
        convert_model_to_int8_tflite(denoiser, denoiser_tflite_out, noisy_train)
        convert_model_to_int8_tflite(predictor, predictor_tflite_out, denoised_train)
        denoiser_runner = TFLiteModelRunner(denoiser_tflite_out)
        predictor_runner = TFLiteModelRunner(predictor_tflite_out)
        int8_denoised, latencies = _tflite_predict_many(denoiser_runner, noisy_val[..., None])
        int8_targets, predictor_latencies = _tflite_predict_many(predictor_runner, int8_denoised)
        int8_r2 = r2_columns_np(target_val, int8_targets)
        metrics.update(
            {
                "int8_denoising_psnr": psnr(clean_val[..., None], int8_denoised),
                "int8_denoising_ssim": ssim_1d(clean_val[..., None], int8_denoised),
                "int8_ri_mae": mae(target_val[:, 0], int8_targets[:, 0]),
                "int8_ri_r2": float(int8_r2[0]),
                "int8_lambda_res_mae_nm": mae(target_val[:, 1], int8_targets[:, 1]),
                "int8_lambda_res_r2": float(int8_r2[1]),
                "int8_denoiser_latency_ms_p50": float(np.percentile(latencies, 50)),
                "int8_denoiser_latency_ms_p95": float(np.percentile(latencies, 95)),
                "int8_predictor_latency_ms_p50": float(np.percentile(predictor_latencies, 50)),
                "int8_predictor_latency_ms_p95": float(np.percentile(predictor_latencies, 95)),
                "int8_denoiser_size_bytes": float(denoiser_tflite_out.stat().st_size),
                "int8_predictor_size_bytes": float(predictor_tflite_out.stat().st_size),
                "int8_ri_mae_delta_vs_float": mae(target_val[:, 0], int8_targets[:, 0]) - metrics["ri_mae"],
                "int8_lambda_mae_delta_vs_float_nm": mae(target_val[:, 1], int8_targets[:, 1]) - metrics["lambda_res_mae_nm"],
            }
        )

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and deploy PCF-SPR edge denoising models.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="auto", help="auto, cpu, /GPU:0, or /CPU:0")
    parser.add_argument("--quantize", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=Path("models/edge_denoiser.keras"))
    parser.add_argument("--ri-out", type=Path, default=Path("models/edge_ri_predictor.keras"))
    parser.add_argument("--denoiser-tflite-out", type=Path, default=Path("models/edge_denoiser_quantized.tflite"))
    parser.add_argument("--ri-tflite-out", type=Path, default=Path("models/edge_ri_predictor_quantized.tflite"))
    args = parser.parse_args()

    metrics = train_edge_models(
        data_path=args.data, denoiser_out=args.out, predictor_out=args.ri_out,
        epochs=args.epochs, batch_size=args.batch_size, device=args.device, quantize=args.quantize,
        denoiser_tflite_out=args.denoiser_tflite_out, predictor_tflite_out=args.ri_tflite_out,
        seed=args.seed,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
