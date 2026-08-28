from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from sprpcf.simulation.schema import SpectralMetrics

SENSITIVITY_GROUP_COLUMNS = [
    "d_over_lambda",
    "pitch_um",
    "metal_thickness_nm",
    "channel_radius_um",
]


def _validate_spectrum(wavelength_nm: np.ndarray, loss: np.ndarray) -> None:
    if wavelength_nm.ndim != 1 or loss.ndim != 1 or wavelength_nm.size != loss.size:
        raise ValueError("wavelength_nm and loss must be one-dimensional arrays of equal length.")
    if wavelength_nm.size < 3:
        raise ValueError("At least three wavelength samples are required.")
    if not np.all(np.isfinite(wavelength_nm)) or not np.all(np.isfinite(loss)):
        raise ValueError("Spectrum contains non-finite values.")
    if np.any(np.diff(wavelength_nm) <= 0):
        raise ValueError("wavelength_nm must be strictly increasing.")


def resonance_wavelength(wavelength_nm: np.ndarray, loss: np.ndarray) -> tuple[float, float]:
    _validate_spectrum(wavelength_nm, loss)
    peaks, _ = find_peaks(loss)
    peak_index = int(peaks[np.argmax(loss[peaks])]) if peaks.size else int(np.argmax(loss))
    return float(wavelength_nm[peak_index]), float(loss[peak_index])


def _interpolated_crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if y1 == y0:
        return x0
    return x0 + (level - y0) / (y1 - y0) * (x1 - x0)


def fwhm(wavelength_nm: np.ndarray, loss: np.ndarray) -> float:
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
    if left == peak_index or right == peak_index or (left == 0 and loss[left] > half_height) or (right == loss.size - 1 and loss[right] > half_height):
        return float("nan")
    left_crossing = _interpolated_crossing(float(wavelength_nm[left]), float(loss[left]), float(wavelength_nm[left + 1]), float(loss[left + 1]), half_height)
    right_crossing = _interpolated_crossing(float(wavelength_nm[right - 1]), float(loss[right - 1]), float(wavelength_nm[right]), float(loss[right]), half_height)
    return float(right_crossing - left_crossing)


def extract_metrics(wavelength_nm: np.ndarray, loss: np.ndarray, sensitivity_nm_per_riu: float | None = None) -> SpectralMetrics:
    lambda_res, peak_loss = resonance_wavelength(wavelength_nm, loss)
    width = fwhm(wavelength_nm, loss)
    fom = float(sensitivity_nm_per_riu / width) if sensitivity_nm_per_riu is not None and np.isfinite(width) and width > 0 else None
    return SpectralMetrics(lambda_res, peak_loss, width, sensitivity_nm_per_riu, fom)


def finite_difference_sensitivity(lambda_res_nm: np.ndarray, analyte_ri: np.ndarray) -> np.ndarray:
    lambda_res_nm = np.asarray(lambda_res_nm, dtype=float)
    analyte_ri = np.asarray(analyte_ri, dtype=float)
    if lambda_res_nm.ndim != 1 or analyte_ri.ndim != 1 or lambda_res_nm.size != analyte_ri.size:
        raise ValueError("lambda_res_nm and analyte_ri must be one-dimensional arrays of equal length.")
    if lambda_res_nm.size < 2:
        return np.full(lambda_res_nm.shape, np.nan, dtype=float)
    if not np.all(np.isfinite(lambda_res_nm)) or not np.all(np.isfinite(analyte_ri)):
        raise ValueError("Sensitivity inputs must be finite.")
    order = np.argsort(analyte_ri, kind="stable")
    sorted_ri = analyte_ri[order]
    sorted_lambda = lambda_res_nm[order]
    if np.any(np.diff(sorted_ri) <= 0):
        raise ValueError("analyte_ri values must be unique within a fixed-geometry sensitivity group.")
    sensitivity = np.gradient(sorted_lambda, sorted_ri, edge_order=2 if sorted_ri.size >= 3 else 1)
    restored = np.empty_like(sensitivity)
    restored[order] = sensitivity
    return restored


def assign_grouped_sensitivity(frame: pd.DataFrame, group_columns: Sequence[str] = SENSITIVITY_GROUP_COLUMNS, *, lambda_column: str = "lambda_res_nm", ri_column: str = "analyte_ri", fwhm_column: str = "fwhm_nm") -> pd.DataFrame:
    required = [*group_columns, lambda_column, ri_column, fwhm_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing columns for sensitivity calculation: {missing}")
    result = frame.copy()
    result["sensitivity_nm_per_riu"] = np.nan
    result["fom_per_riu"] = np.nan
    for _, group in result.groupby(list(group_columns), dropna=False, sort=False):
        valid = group[[lambda_column, ri_column]].dropna()
        if valid[ri_column].nunique() < 2:
            continue
        averaged = valid.groupby(ri_column, as_index=False)[lambda_column].mean().sort_values(ri_column)
        sensitivity = finite_difference_sensitivity(averaged[lambda_column].to_numpy(float), averaged[ri_column].to_numpy(float))
        lookup = dict(zip(averaged[ri_column].to_numpy(float), sensitivity))
        indices = group.index
        result.loc[indices, "sensitivity_nm_per_riu"] = [lookup.get(float(v), np.nan) for v in result.loc[indices, ri_column]]
    widths = result[fwhm_column].to_numpy(float)
    sensitivities = result["sensitivity_nm_per_riu"].to_numpy(float)
    valid_fom = np.isfinite(widths) & (widths > 0) & np.isfinite(sensitivities)
    result.loc[valid_fom, "fom_per_riu"] = sensitivities[valid_fom] / widths[valid_fom]
    return result
