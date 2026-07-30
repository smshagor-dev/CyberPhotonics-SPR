from __future__ import annotations

import numpy as np

from sprpcf.simulation.metrics import extract_metrics, resonance_wavelength


def test_resonance_wavelength_finds_peak() -> None:
    wavelength = np.linspace(500.0, 700.0, 201)
    loss = 1.0 / (1.0 + ((wavelength - 612.0) / 10.0) ** 2)
    lambda_res, peak_loss = resonance_wavelength(wavelength, loss)
    assert abs(lambda_res - 612.0) <= 1.0
    assert peak_loss > 0.99


def test_extract_metrics_returns_positive_fwhm() -> None:
    wavelength = np.linspace(500.0, 700.0, 201)
    loss = 1.0 / (1.0 + ((wavelength - 612.0) / 10.0) ** 2)
    metrics = extract_metrics(wavelength, loss, sensitivity_nm_per_riu=1000.0)
    assert metrics.fwhm_nm > 0.0
    assert metrics.fom_per_riu is not None
