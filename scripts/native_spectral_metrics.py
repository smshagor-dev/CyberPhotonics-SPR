"""Conservative native loss-spectrum analysis; never certifies Maxwell physics."""
import math


def analyze(wavelength_um, loss_db_cm, *, minimum_points=7):
    """Find a sampled interior peak and linearly interpolated FWHM, or fail closed."""
    x, y = list(wavelength_um), list(loss_db_cm)
    if len(x) != len(y) or len(x) < minimum_points:
        raise ValueError("insufficient spectral samples")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in x + y):
        raise ValueError("nonfinite or nonnumeric spectrum")
    if any(a >= b for a, b in zip(x, x[1:])) or x[0] <= 0 or min(y) < 0:
        raise ValueError("invalid wavelength grid or negative loss")
    maxima = [i for i in range(1, len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    if len(maxima) != 1:
        raise ValueError("spectrum requires one isolated interior peak")
    i = maxima[0]
    baseline = min(y[0], y[-1])
    half = baseline + (y[i] - baseline) / 2

    def crossing(indices):
        for a, b in indices:
            if (y[a] - half) * (y[b] - half) <= 0 and y[a] != y[b]:
                return x[a] + (half-y[a])*(x[b]-x[a])/(y[b]-y[a])
        raise ValueError("half-maximum crossing not bracketed")

    left = crossing((j, j+1) for j in range(i-1, -1, -1))
    right = crossing((j, j+1) for j in range(i, len(y)-1))
    if right <= left:
        raise ValueError("nonpositive FWHM")
    return {"peak_wavelength_um": x[i], "peak_loss_db_cm": y[i],
            "fwhm_um": right-left, "left_half_um": left, "right_half_um": right,
            "method": "sampled_peak_linear_half_height",
            "physics_qualified": False, "publication_ready": False, "comsol_verified": False}
