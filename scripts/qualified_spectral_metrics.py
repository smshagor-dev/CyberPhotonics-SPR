"""Fail-closed sampled loss-spectrum diagnostics; NOT qualified PCF or COMSOL evidence."""
from __future__ import annotations
import math
from collections.abc import Sequence


def resonance_metrics(wavelength_um: Sequence[float], loss_db_per_cm: Sequence[float]) -> dict:
    """Single sampled peak and linearly interpolated half-prominence width.

    Baseline is the straight line through measured endpoints. No peak interpolation
    or extrapolation is performed. Sampling uncertainty remains unresolved.
    """
    x = [float(v) for v in wavelength_um]
    y = [float(v) for v in loss_db_per_cm]
    if len(x) != len(y) or len(x) < 7:
        raise ValueError("at least seven paired spectral samples required")
    if not all(math.isfinite(v) for v in x + y) or any(v <= 0 for v in x) or any(v < 0 for v in y):
        raise ValueError("finite positive wavelengths and nonnegative losses required")
    if any(b <= a for a, b in zip(x, x[1:])):
        raise ValueError("wavelengths must be strictly increasing")
    baseline = [y[0] + (y[-1] - y[0]) * (v - x[0]) / (x[-1] - x[0]) for v in x]
    z = [a - b for a, b in zip(y, baseline)]
    peaks = [i for i in range(1, len(z) - 1) if z[i] > z[i-1] and z[i] > z[i+1]]
    if len(peaks) != 1:
        raise ValueError("requires exactly one resolved interior peak")
    p = peaks[0]
    if z[p] <= 0:
        raise ValueError("peak prominence must be positive")
    half = z[p] / 2
    left = [i for i in range(p) if z[i] <= half < z[i+1]]
    right = [i for i in range(p, len(z)-1) if z[i] > half >= z[i+1]]
    if len(left) != 1 or len(right) != 1:
        raise ValueError("two unique half-prominence crossings required")

    def crossing(i):
        return x[i] + (half - z[i]) * (x[i+1] - x[i]) / (z[i+1] - z[i])

    lo, hi = crossing(left[0]), crossing(right[0])
    if not lo < x[p] < hi:
        raise ValueError("invalid peak crossings")
    return {"peak_sample_wavelength_um": x[p], "peak_loss_db_per_cm": y[p],
            "fwhm_um": hi - lo, "left_half_um": lo, "right_half_um": hi,
            "peak_prominence_db_per_cm": z[p], "sampling_limited": True,
            "physics_qualified": False, "publication_ready": False,
            "comsol_verified": False}


def finite_difference_sensitivity(ri_a: float, peak_um_a: float, ri_b: float, peak_um_b: float) -> float:
    """Sample-peak displacement in nm/RIU; caller must independently verify geometry and mode identity."""
    values = (ri_a, peak_um_a, ri_b, peak_um_b)
    if not all(math.isfinite(v) for v in values) or not 1 <= ri_a <= 2 or not 1 <= ri_b <= 2:
        raise ValueError("finite plausible analyte indices required")
    if ri_a == ri_b or peak_um_a <= 0 or peak_um_b <= 0:
        raise ValueError("distinct RI and positive peak wavelengths required")
    return 1000 * (peak_um_b - peak_um_a) / (ri_b - ri_a)
