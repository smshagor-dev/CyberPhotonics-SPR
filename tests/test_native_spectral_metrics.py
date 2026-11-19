"""Constructed spectra validate numerical postprocessing, not sensor physics."""
import pytest
from scripts.native_spectral_metrics import analyze


def test_triangular_peak():
    result = analyze([1, 2, 3, 4, 5, 6, 7], [0, 1, 2, 4, 2, 1, 0])
    assert result["peak_wavelength_um"] == 4
    assert result["fwhm_um"] == 2
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False


@pytest.mark.parametrize("loss", [
    [0, 1, 2, 3, 4, 5, 6],
    [0, 2, 0, 2, 0, 2, 0],
    [0, 0, 1, 2, 3, 4, 5],
    [0, 0, 0, 1, 0, 0, float("nan")],
])
def test_invalid_peak(loss):
    with pytest.raises(ValueError):
        analyze([1, 2, 3, 4, 5, 6, 7], loss)


def test_duplicate_wavelength():
    with pytest.raises(ValueError):
        analyze([1, 2, 3, 3, 5, 6, 7], [0, 1, 2, 4, 2, 1, 0])
