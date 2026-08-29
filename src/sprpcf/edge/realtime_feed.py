from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from sprpcf.edge.train_denoiser import normalize_spectra, parse_spectra
from sprpcf.ml.dataset import read_table
from sprpcf.simulation.metrics import resonance_wavelength


def _simulate_noisy_measurement(spectrum: np.ndarray, rng: np.random.Generator, noise_std: float) -> np.ndarray:
    """Create a noisy physical-domain measurement from a stored clean spectrum."""
    signal_scale = max(float(np.std(spectrum)), 1e-6)
    return spectrum + rng.normal(0.0, noise_std * signal_scale, spectrum.shape)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay stored PCF-SPR spectra as a simulated real-time sensor feed.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ri-model", type=Path, default=None)
    parser.add_argument("--interval", type=float, default=0.25)
    parser.add_argument("--frames", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.08, help="Noise as a fraction of spectrum standard deviation.")
    args = parser.parse_args()

    frame = read_table(args.data).dropna(subset=["loss_db_per_cm", "wavelength_nm"])
    spectra = parse_spectra(frame)
    wavelengths = np.asarray(
        [np.fromstring(str(value), sep=",") for value in frame["wavelength_nm"].to_list()],
        dtype=np.float32,
    )
    if wavelengths.shape != spectra.shape:
        raise ValueError("wavelength_nm and loss_db_per_cm arrays must have matching shapes.")

    model = tf.keras.models.load_model(args.model, compile=False)
    ri_model = tf.keras.models.load_model(args.ri_model, compile=False) if args.ri_model is not None else None
    rng = np.random.default_rng(11)

    for index in range(min(args.frames, spectra.shape[0])):
        measured = _simulate_noisy_measurement(spectra[index], rng, args.noise_std)
        normalized, mean, std = normalize_spectra(measured[None, :])
        denoised = model.predict(normalized[..., None], verbose=0)[0, :, 0]
        physical_denoised = denoised * std[0, 0] + mean[0, 0]
        lambda_res, peak_loss = resonance_wavelength(wavelengths[index], physical_denoised)
        ri_text = ""
        if ri_model is not None:
            predicted = ri_model.predict(denoised[None, :, None], verbose=0)[0]
            ri_text = f" predicted_ri={float(predicted[0]):.6f} predicted_lambda_res_nm={float(predicted[1]):.2f}"
        print(f"frame={index:04d} lambda_res_nm={lambda_res:.2f} peak_loss_db_per_cm={peak_loss:.3f}{ri_text}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
