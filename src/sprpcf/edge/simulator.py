from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from sprpcf.edge.quantization import TFLiteModelRunner


@dataclass
class SensorFrame:
    """One simulated sensor sample and its timing metadata."""

    index: int
    noisy_spectrum: np.ndarray
    denoised_spectrum: np.ndarray
    latency_ms: float


class SensorFeedSimulator:
    """Stream mock real-time noisy spectra through an edge denoising model."""

    def __init__(
        self,
        clean_spectra: np.ndarray,
        denoiser_tflite: Path,
        noise_std: float = 0.08,
        drift_std: float = 0.03,
        seed: int = 11,
    ) -> None:
        self.clean_spectra = clean_spectra.astype(np.float32)
        self.denoiser_tflite = denoiser_tflite
        self.noise_std = noise_std
        self.drift_std = drift_std
        self.rng = np.random.default_rng(seed)
        self.runner = TFLiteModelRunner(denoiser_tflite)

    def noisy_frame(self, index: int) -> np.ndarray:
        clean = self.clean_spectra[index % self.clean_spectra.shape[0]]
        drift = self.rng.normal(0.0, self.drift_std) * np.linspace(-1.0, 1.0, clean.size)
        noise = self.rng.normal(0.0, self.noise_std, clean.shape)
        return (clean + drift + noise).astype(np.float32)

    def stream(self, frames: int = 20) -> Iterator[SensorFrame]:
        for index in range(frames):
            noisy = self.noisy_frame(index)
            started = time.perf_counter()
            denoised = self.runner.predict(noisy[None, :, None])[0, :, 0]
            latency_ms = (time.perf_counter() - started) * 1000.0
            yield SensorFrame(index=index, noisy_spectrum=noisy, denoised_spectrum=denoised, latency_ms=latency_ms)
