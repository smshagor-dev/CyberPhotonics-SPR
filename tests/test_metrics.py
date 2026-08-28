from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sprpcf.simulation.metrics import (
    extract_metrics,
    finite_difference_sensitivity,
    grouped_finite_difference_sensitivity,
    resonance_wavelength,
)


def test_resonance_wavelength_finds_peak() -> None:
    wavelength = np.linspace(500.0, 700.0, 201)
    loss = 1.0 / (1.0 + ((wavelength - 612.0) / 10.0) ** 2)
    lambda_res, peak_loss = resonance_wavelength(wavelength, loss)
    assert abs(lambda_res - 612.0) <= 1.0
    assert peak_loss > 0.99


def test_extract_metrics_interpolates_positive_fwhm() -> None:
    wavelength = np.linspace(500.0, 700.0, 201)
    loss = 1.0 / (1.0 + ((wavelength - 612.0) / 10.0) ** 2)
    metrics = extract_metrics(wavelength, loss, sensitivity_nm_per_riu=1000.0)
    assert 18.0 <= metrics.fwhm_nm <= 22.0
    assert metrics.fom_per_riu is not None


def test_sensitivity_rejects_duplicate_ri_values() -> None:
    with pytest.raises(ValueError, match="unique"):
        finite_difference_sensitivity(np.array([600.0, 610.0, 620.0]), np.array([1.33, 1.33, 1.35]))


def test_grouped_sensitivity_never_mixes_geometries() -> None:
    frame = pd.DataFrame(
        {
            "pitch_um": [2.0, 2.0, 2.0, 2.5, 2.5, 2.5],
            "d_over_lambda": [0.5] * 6,
            "metal_thickness_nm": [45.0] * 6,
            "channel_radius_um": [0.6] * 6,
            "analyte_ri": [1.33, 1.35, 1.37] * 2,
            "lambda_res_nm": [600.0, 620.0, 640.0, 700.0, 710.0, 720.0],
        }
    )
    result = grouped_finite_difference_sensitivity(
        frame,
        ["pitch_um", "d_over_lambda", "metal_thickness_nm", "channel_radius_um"],
    )
    assert np.allclose(result.iloc[:3], 1000.0)
    assert np.allclose(result.iloc[3:], 500.0)
