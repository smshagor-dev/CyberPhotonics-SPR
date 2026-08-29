from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sprpcf.simulation.schema import SpectralMetrics


def _validate_spectrum(wavelength_nm: np.ndarray, loss: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength_nm, dtype=float)
    spectrum = np.asarray(loss, dtype=float)
    if wavelength.ndim != 1 or spectrum.ndim != 1 or wavelength.size != spectrum.size:
        raise ValueError("wavelength_nm and loss must be one-dimensional arrays of equal length.")
    if wavelength.size < 3:
        raise ValueError("At least three wavelength samples are required.")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(spectrum)):
        raise ValueError("Spectrum contains non-finite values.")

    delta = np.diff(wavelength)
    if np.all(delta < 0):
        wavelength = wavelength[::-1]
        spectrum = spectrum[::-1]
        delta = np.diff(wavelength)
    if not np.all(delta > 0):
        raise ValueError("wavelength_nm must be strictly monotonic.")
    return wavelength, spectrum


def resonance_wavelength(wavelength_nm: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
    """Return resonance wavelength and peak loss from a confinement-loss spectrum."""
    wavelength, spectrum = _validate_spectrum(wavelength_nm, loss)
    peaks, _ = find_peaks(spectrum)
    peak_index = int(peaks[np.argmax(spectrum[peaks])]) if peaks.size else int(np.argmax(spectrum))
    peak_wavelength = float(wavelength[peak_index])
    peak_loss = float(spectrum[peak_index])

    # Refine the peak below the wavelength-grid spacing with a local quadratic fit.
    # This reduces discretization bias in sensitivity without extrapolating beyond
    # the three measured samples around the strongest peak.
    if 0 < peak_index < wavelength.size - 1:
        local_x = wavelength[peak_index - 1 : peak_index + 2] - wavelength[peak_index]
        local_y = spectrum[peak_index - 1 : peak_index + 2]
        a, b, c = np.polyfit(local_x, local_y, 2)
        if np.isfinite(a) and np.isfinite(b) and a < 0:
            vertex = -b / (2.0 * a)
            if local_x[0] <= vertex <= local_x[-1]:
                peak_wavelength = float(wavelength[peak_index] + vertex)
                peak_loss = float(a * vertex**2 + b * vertex + c)
    return peak_wavelength, peak_loss


def _linear_crossing(x0: float, y0: float, x1: float, y1: float, target: float) -> float:
    if y1 == y0:
        return 0.5 * (x0 + x1)
    fraction = (target - y0) / (y1 - y0)
    return float(x0 + fraction * (x1 - x0))


def fwhm(wavelength_nm: np.ndarray, loss: np.ndarray) -> float:
    """Estimate FWHM around the strongest resonance with linear interpolation."""
    wavelength, spectrum = _validate_spectrum(wavelength_nm, loss)
    lambda_res, peak_loss = resonance_wavelength(wavelength, spectrum)
    peak_index = int(np.argmin(np.abs(wavelength - lambda_res)))
    baseline = float(np.min(spectrum))
    half_height = baseline + 0.5 * (peak_loss - baseline)

    left = peak_index
    while left > 0 and spectrum[left] > half_height:
        left -= 1
    right = peak_index
    while right < spectrum.size - 1 and spectrum[right] > half_height:
        right += 1

    if left == peak_index or right == peak_index:
        return float("nan")
    if spectrum[left] > half_height or spectrum[right] > half_height:
        return float("nan")

    left_cross = _linear_crossing(
        float(wavelength[left]),
        float(spectrum[left]),
        float(wavelength[left + 1]),
        float(spectrum[left + 1]),
        half_height,
    )
    right_cross = _linear_crossing(
        float(wavelength[right - 1]),
        float(spectrum[right - 1]),
        float(wavelength[right]),
        float(spectrum[right]),
        half_height,
    )
    width = right_cross - left_cross
    return float(width) if width > 0 else float("nan")


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
        fom = float(abs(sensitivity_nm_per_riu) / width)
    return SpectralMetrics(
        lambda_res_nm=lambda_res,
        peak_loss_db_per_cm=peak_loss,
        fwhm_nm=width,
        sensitivity_nm_per_riu=sensitivity_nm_per_riu,
        fom_per_riu=fom,
    )


def finite_difference_sensitivity(lambda_res_nm: np.ndarray, analyte_ri: np.ndarray) -> np.ndarray:
    """Compute local wavelength sensitivity for a single fixed geometry.

    The refractive-index samples must be unique. Passing mixed geometries or duplicate
    RI values is a scientific error and is rejected instead of silently producing
    infinities or cross-geometry gradients.
    """
    lambda_res = np.asarray(lambda_res_nm, dtype=float)
    ri = np.asarray(analyte_ri, dtype=float)
    if lambda_res.ndim != 1 or ri.ndim != 1 or lambda_res.size != ri.size:
        raise ValueError("lambda_res_nm and analyte_ri must be one-dimensional arrays of equal length.")
    if lambda_res.size < 2:
        raise ValueError("At least two RI samples are required to estimate sensitivity.")
    if not np.all(np.isfinite(lambda_res)) or not np.all(np.isfinite(ri)):
        raise ValueError("Sensitivity inputs contain non-finite values.")

    order = np.argsort(ri)
    sorted_ri = ri[order]
    sorted_lambda = lambda_res[order]
    if np.any(np.diff(sorted_ri) <= 0):
        raise ValueError("analyte_ri values must be unique within each fixed-geometry sweep.")

    edge_order = 2 if sorted_ri.size >= 3 else 1
    sensitivity = np.gradient(sorted_lambda, sorted_ri, edge_order=edge_order)
    restored = np.empty_like(sensitivity)
    restored[order] = sensitivity
    return restored


def grouped_finite_difference_sensitivity(
    frame: pd.DataFrame,
    group_columns: Sequence[str],
    ri_column: str = "analyte_ri",
    lambda_column: str = "lambda_res_nm",
) -> pd.Series:
    """Compute sensitivity only within rows sharing an identical sensor geometry."""
    required = [*group_columns, ri_column, lambda_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for grouped sensitivity: {missing}")

    result = pd.Series(np.nan, index=frame.index, dtype=float)
    grouped = frame.groupby(list(group_columns), sort=False, dropna=False)
    for _, group in grouped:
        valid = group.dropna(subset=[ri_column, lambda_column])
        if valid.shape[0] < 2:
            continue
        sensitivity = finite_difference_sensitivity(
            valid[lambda_column].to_numpy(dtype=float),
            valid[ri_column].to_numpy(dtype=float),
        )
        result.loc[valid.index] = sensitivity
    return result
