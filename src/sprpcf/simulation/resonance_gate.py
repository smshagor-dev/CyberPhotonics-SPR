"""Conservative sampled-loss peak diagnostic; not physical provenance evidence."""
import math

def bracketed_resonance(wavelengths_um, losses_db_cm):
    """Half-prominence linear-crossing diagnostic; no continuous-spectrum certification."""
    x, y = list(wavelengths_um), list(losses_db_cm)
    if len(x) != len(y) or len(x) < 7:
        raise ValueError("at least seven paired spectral samples required")
    if any(type(v) not in (float, int) or not math.isfinite(v) for v in x + y):
        raise ValueError("spectral coordinates and loss must be finite numeric values")
    if any(a <= 0 for a in x) or any(a < 0 for a in y):
        raise ValueError("wavelength must be positive and attenuation nonnegative")
    if any(a >= b for a, b in zip(x, x[1:])):
        raise ValueError("wavelength grid must be strictly increasing")
    peaks = [i for i in range(1, len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    if len(peaks) != 1:
        raise ValueError("exactly one strict interior sampled maximum required")
    p = peaks[0]
    if p < 2 or len(y)-p-1 < 2:
        raise ValueError("at least two samples on each side of the maximum required")
    baseline = max(min(y[:p]), min(y[p+1:]))
    if y[p] <= baseline:
        raise ValueError("no positive peak prominence")
    half = baseline + (y[p]-baseline)/2
    left = next((i for i in range(p-1, -1, -1) if y[i] <= half), None)
    right = next((i for i in range(p+1, len(y)) if y[i] <= half), None)
    if left is None or right is None:
        raise ValueError("half-prominence crossings not bracketed on both sides")
    def crossing(i, j):
        if y[i] == y[j]:
            raise ValueError("flat half-height crossing")
        return x[i] + (half-y[i])*(x[j]-x[i])/(y[j]-y[i])
    l = crossing(left, left+1)
    r = crossing(right-1, right)
    return {
        "sampled_peak_um": x[p],
        "sampled_peak_loss_db_cm": y[p],
        "half_prominence_loss_db_cm": half,
        "fwhm_diagnostic_um": r-l,
        "left_crossing_um": l,
        "right_crossing_um": r,
        "spectral_resolution_verified": False,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
    }
