from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks

from sprpcf.simulation.schema import SpectralMetrics


def resonance_wavelength(wavelength_nm: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
    """Return resonance wavelength and peak loss from a confinement-loss spectrum."""
    if wavelength_nm.ndim != 1 or loss.ndim != 1 or wavelength_nm.size != loss.size:
        raise ValueError("wavelength_nm and loss must be one-dimensional arrays of equal length.")

    peaks, _ = find_peaks(loss)
    peak_index = int(peaks[np.argmax(loss[peaks])]) if peaks.size else int(np.argmax(loss))
    return float(wavelength_nm[peak_index]), float(loss[peak_index])


def fwhm(wavelength_nm: np.ndarray, loss: np.ndarray) -> float:
    """Estimate full width at half maximum around the strongest resonance peak."""
    lambda_res, peak_loss = resonance_wavelength(wavelength_nm, loss)
    peak_index = int(np.argmin(np.abs(wavelength_nm - lambda_res)))
    baseline = float(np.min(loss))
    half_height = baseline + 0.5 * (peak_loss - baseline)

    left = peak_index
    while left > 0 and loss[left] > half_height:
        left -= 1

    right = peak_index
    while right < loss.size - 1 and loss[right] > half_height:
        right += 1

    if left == peak_index or right == peak_index:
        return float("nan")

    return float(wavelength_nm[right] - wavelength_nm[left])


def extract_metrics(
    wavelength_nm: np.ndarray,
    loss: np.ndarray,
    sensitivity_nm_per_riu: float | None = None,
) -> SpectralMetrics:
    """Extract resonance wavelength, peak loss, FWHM, and FOM."""
    lambda_res, peak_loss = resonance_wavelength(wavelength_nm, loss)
    width = fwhm(wavelength_nm, loss)
    fom = None
    if sensitivity_nm_per_riu is not None and np.isfinite(width) and width > 0:
        fom = float(sensitivity_nm_per_riu / width)
    return SpectralMetrics(
        lambda_res_nm=lambda_res,
        peak_loss_db_per_cm=peak_loss,
        fwhm_nm=width,
        sensitivity_nm_per_riu=sensitivity_nm_per_riu,
        fom_per_riu=fom,
    )


def finite_difference_sensitivity(lambda_res_nm: np.ndarray, analyte_ri: np.ndarray) -> np.ndarray:
    """Compute local wavelength sensitivity in nm/RIU using finite differences."""
    order = np.argsort(analyte_ri)
    sorted_ri = analyte_ri[order]
    sorted_lambda = lambda_res_nm[order]
    sensitivity = np.gradient(sorted_lambda, sorted_ri)
    restored = np.empty_like(sensitivity)
    restored[order] = sensitivity
    return restored
