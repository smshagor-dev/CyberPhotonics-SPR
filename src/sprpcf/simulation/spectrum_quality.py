"""Conservative quality controls for COMSOL confinement-loss spectra.

These checks reject malformed modal/solver exports and unresolved peaks, but
cannot establish source authenticity, electromagnetic correctness, material
dispersion, mesh convergence or publication-quality physical results.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from sprpcf.simulation.metrics import extract_metrics
from sprpcf.simulation.schema import SpectralMetrics


def _numeric_series(value: Any, label: str) -> np.ndarray:
    """Accept a single optical series; never concatenate multiple eigenmodes."""
    original = np.asarray(value)
    if original.ndim == 0 or original.ndim > 2:
        raise ValueError(f"{label}: expected a 1-D single-mode spectrum")
    if original.ndim == 2:
        if 1 not in original.shape:
            raise ValueError(
                f"{label}: multiple modal/solution columns cannot be flattened as one spectrum"
            )
        original = original.reshape(-1)
    if np.iscomplexobj(original):
        # Even small nonzero imaginary parts can hide a wrong COMSOL expression.
        if not np.isfinite(original).all() or np.any(original.imag != 0):
            raise ValueError(f"{label}: complex-valued optical export is ambiguous")
        original = original.real
    try:
        data = np.asarray(original, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label}: optical export must be real numeric") from exc
    if data.ndim != 1 or not np.isfinite(data).all():
        raise ValueError(f"{label}: nonfinite or invalid optical series")
    return data


def normalize_comsol_export(
    wavelength: Any, loss: Any, config: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize one COMSOL spectral series, rejecting multi-solution flattening."""
    # Never scale a view returned by MPh in place: the same Java/Numpy-backed
    # arrays may be reused by subsequent model evaluations or archived raw.
    w = _numeric_series(wavelength, "wavelength").copy()
    l = _numeric_series(loss, "confinement loss").copy()
    if w.size != l.size or w.size < 3:
        raise ValueError("COMSOL wavelength and loss arrays must have equal length >= 3")
    scales: dict[str, float] = {}
    for key in ("wavelength_scale_to_nm", "loss_scale_to_db_per_cm"):
        raw = config.get(key, 1.0)
        try:
            value = float(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a positive finite conversion factor") from exc
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be a positive finite conversion factor")
        scales[key] = value
    w *= scales["wavelength_scale_to_nm"]
    l *= scales["loss_scale_to_db_per_cm"]
    if not np.isfinite(w).all() or not np.isfinite(l).all():
        raise ValueError("COMSOL conversion produced nonfinite wavelengths/loss")
    change = np.diff(w)
    if np.all(change < 0):
        w, l = w[::-1].copy(), l[::-1].copy()
    elif not np.all(change > 0):
        raise ValueError("COMSOL wavelength samples must be strictly monotonic")
    expected = config.get("expected_wavelength_nm", [300.0, 2500.0])
    if not isinstance(expected, (tuple, list)) or len(expected) != 2:
        raise ValueError("expected_wavelength_nm must be [min_nm, max_nm]")
    try:
        lower, upper = float(expected[0]), float(expected[1])
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("expected_wavelength_nm requires finite numeric bounds") from exc
    if (not math.isfinite(lower) or not math.isfinite(upper) or lower <= 0
            or upper <= lower):
        raise ValueError("expected_wavelength_nm requires ordered positive finite bounds")
    if w[0] < lower or w[-1] > upper:
        raise ValueError(
            "COMSOL wavelength values are outside expected_wavelength_nm. "
            "Check wavelength_scale_to_nm and the model export expression."
        )
    return w, l


def validate_optical_spectrum(
    wavelength_nm: Any, loss_db_per_cm: Any, *,
    min_samples: int = 16, min_fwhm_steps: float = 3.0,
) -> SpectralMetrics:
    """Return derived metrics only if the peak is observable and resolved.

    This is a *numerical data-quality check*, not a physical verification of
    mode identity, material optical constants, Maxwell equations or meshing.
    """
    w = _numeric_series(wavelength_nm, "wavelength")
    loss = _numeric_series(loss_db_per_cm, "confinement loss")
    if w.shape != loss.shape or len(w) < min_samples:
        raise ValueError(f"Optical spectrum requires >= {min_samples} matched finite samples")
    steps = np.diff(w)
    if not np.all(steps > 0):
        raise ValueError("Optical spectrum must have strictly increasing wavelengths")
    if w[0] <= 0:
        raise ValueError("Wavelength values must be positive")
    if float(np.min(loss)) < -1e-9:
        raise ValueError("Optical confinement loss contains materially negative values")
    if float(np.max(loss) - np.min(loss)) <= max(
        1e-9, 1e-6 * float(np.max(np.abs(loss)))
    ):
        raise ValueError("Optical spectrum has no resolvable resonance contrast")
    try:
        metrics = extract_metrics(w, loss)
    except (TypeError, ValueError, FloatingPointError) as exc:
        raise ValueError("Optical resonance extraction failed") from exc
    if not math.isfinite(metrics.lambda_res_nm):
        raise ValueError("Optical resonance is missing/nonfinite")
    if not (w[2] < metrics.lambda_res_nm < w[-3]):
        raise ValueError("Optical resonance lies too close to the spectral boundary")
    if (not math.isfinite(metrics.fwhm_nm)
            or not math.isfinite(metrics.peak_loss_db_per_cm)
            or metrics.fwhm_nm <= 0):
        raise ValueError("Optical resonance or FWHM is missing/nonfinite")
    resolution = float(np.median(steps))
    if metrics.fwhm_nm < min_fwhm_steps * resolution:
        raise ValueError("Optical resonance FWHM is under-resolved by the wavelength grid")
    return metrics
