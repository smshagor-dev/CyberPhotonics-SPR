from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sprpcf.edge.train_denoiser import parse_spectra
from sprpcf.ml.dataset import read_table
from sprpcf.simulation.metrics import resonance_wavelength


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate real-time noisy PCF-SPR spectral feed.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ri-model", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()

    frame = read_table(args.data).dropna(subset=["loss_db_per_cm", "wavelength_nm"])
    spectra = parse_spectra(frame)
    wavelengths = np.asarray(
        [np.fromstring(value, sep=",") for value in frame["wavelength_nm"].to_list()],
        dtype=np.float32,
    )
    mean = spectra.mean(axis=1, keepdims=True)
    std = spectra.std(axis=1, keepdims=True) + 1e-6
    normalized = (spectra - mean) / std
    model = tf.keras.models.load_model(args.model)
    ri_model = tf.keras.models.load_model(args.ri_model) if args.ri_model is not None else None
    rng = np.random.default_rng(11)

    for index in range(min(args.frames, normalized.shape[0])):
        noisy = normalized[index] + rng.normal(0.0, 0.08, normalized[index].shape)
        denoised = model.predict(noisy[None, :, None], verbose=0)[0, :, 0]
        physical_denoised = denoised * std[index, 0] + mean[index, 0]
        lambda_res, peak_loss = resonance_wavelength(wavelengths[index], physical_denoised)
        ri_text = ""
        if ri_model is not None:
            predicted = ri_model.predict(denoised[None, :, None], verbose=0)[0]
            ri_text = f" predicted_ri={float(predicted[0]):.6f} predicted_lambda_res_nm={float(predicted[1]):.2f}"
        print(f"frame={index:04d} lambda_res_nm={lambda_res:.2f} peak_loss_db_per_cm={peak_loss:.3f}{ri_text}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
