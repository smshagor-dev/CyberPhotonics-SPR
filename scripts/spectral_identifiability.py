"""Conservative sampled-spectrum diagnostic; never physics qualification."""
import math

def inspect_spectrum(wavelength_nm, loss_db_cm):
    out = dict(identified=False, resonance_verified=False, fwhm_verified=False,
               width_convention="half_prominence", reason=None, peak_nm=None, width_nm=None)
    try:
        wavelength_nm, loss_db_cm = list(wavelength_nm), list(loss_db_cm)
    except (TypeError, ValueError):
        out["reason"] = "invalid_sample_container"
        return out
    if len(wavelength_nm) != len(loss_db_cm) or len(wavelength_nm) < 5:
        out["reason"] = "requires_at_least_five_paired_samples"
        return out
    if any(isinstance(v, bool) for v in wavelength_nm + loss_db_cm):
        out["reason"] = "boolean_sample_rejected"
        return out
    try:
        x, y = [float(v) for v in wavelength_nm], [float(v) for v in loss_db_cm]
    except (ValueError, TypeError, OverflowError):
        out["reason"] = "non_numeric_input"
        return out
    if not all(math.isfinite(v) for v in x+y) or not all(a < b for a,b in zip(x,x[1:])) or any(v <= 0 for v in x):
        out["reason"] = "nonfinite_nonpositive_or_nonmonotonic_wavelength"
        return out
    if any(v < 0 for v in y):
        out["reason"] = "negative_attenuation"
        return out
    # Multiple local maxima, plateaus, or nonmonotone flanks invalidate a unique peak.
    maxima = [i for i in range(1,len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    if len(maxima) != 1 or any(y[i] == y[i+1] for i in range(len(y)-1)):
        out["reason"] = "ambiguous_local_maxima_or_plateau"
        return out
    p = maxima[0]
    if p < 2 or p > len(y)-3:
        out["reason"] = "insufficient_peak_flanks"
        return out
    if any(y[i] > y[i+1] for i in range(p)) or any(y[i] < y[i+1] for i in range(p,len(y)-1)):
        out["reason"] = "nonunimodal_flanks"
        return out
    baseline = max(min(y[:p]), min(y[p+1:]))
    prominence = y[p] - baseline
    if prominence <= 0:
        out["reason"] = "no_positive_prominence"
        return out
    half = y[p] - prominence/2
    left = next((i for i in range(p-1,-1,-1) if y[i] <= half), None)
    right = next((i for i in range(p+1,len(y)) if y[i] <= half), None)
    if left is None or right is None:
        out["reason"] = "half_prominence_not_bracketed"
        return out
    def cross(i,j):
        return x[i] + (half-y[i])*(x[j]-x[i])/(y[j]-y[i])
    xl,xr = cross(left,left+1),cross(right-1,right)
    if not xl < x[p] < xr:
        out["reason"] = "invalid_crossing_geometry"
        return out
    out.update(identified=True, reason="interpolated_half_prominence_only",
               peak_nm=x[p], width_nm=xr-xl, left_crossing_nm=xl,
               right_crossing_nm=xr, prominence_db_cm=prominence)
    return out
