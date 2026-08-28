from __future__ import annotations

import numpy as np
import pytest

from sprpcf.simulation.comsol_sweep import _normalize_evaluated_spectrum


def test_comsol_unit_scale_is_explicit_and_validated() -> None:
    wavelength_m = np.array([500e-9, 600e-9, 700e-9])
    loss = np.array([1.0, 2.0, 1.0])
    wavelength_nm, scaled_loss = _normalize_evaluated_spectrum(
        wavelength_m,
        loss,
        {"wavelength_scale_to_nm": 1e9, "loss_scale_to_db_per_cm": 1.0, "expected_wavelength_nm": [400, 800]},
    )
    assert np.allclose(wavelength_nm, [500.0, 600.0, 700.0])
    assert np.allclose(scaled_loss, loss)


def test_comsol_wrong_units_fail_instead_of_silently_corrupting_results() -> None:
    with pytest.raises(ValueError, match="wavelength_scale_to_nm"):
        _normalize_evaluated_spectrum(
            np.array([500e-9, 600e-9, 700e-9]),
            np.array([1.0, 2.0, 1.0]),
            {"expected_wavelength_nm": [400, 800]},
        )
