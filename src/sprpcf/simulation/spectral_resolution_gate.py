"""Conservative software-only peak diagnostic. NOT Maxwell or publication evidence."""
from math import isfinite

def inspect_spectrum(wavelength_um, loss_db_cm, *, min_points=21, min_halfwidth_samples=3):
    """Half-prominence FWHM; no extrapolation and no physics qualification."""
    out = {"resolved": False, "publication_qualified": False,
           "resonance_um": None, "fwhm_um": None, "failures": []}
    fail = out["failures"]
    try:
        x = [float(v) for v in wavelength_um]
        y = [float(v) for v in loss_db_cm]
    except (TypeError, ValueError, OverflowError):
        fail.append("invalid_numeric_input")
        return out
    if len(x) != len(y) or len(x) < min_points:
        fail.append("insufficient_spectral_samples")
        return out
    if not all(isfinite(v) for v in x+y) or any(b <= a for a,b in zip(x,x[1:])):
        fail.append("nonfinite_or_unordered_samples")
        return out
    if any(v < 0 for v in y):
        fail.append("negative_loss")
        return out
    peaks = [i for i in range(1,len(y)-1) if y[i]>y[i-1] and y[i]>y[i+1]]
    if len(peaks) != 1:
        fail.append("peak_not_unique")
        return out
    p = peaks[0]
    baseline = max(min(y[:p+1]),min(y[p:]))
    half = baseline + (y[p]-baseline)/2
    if y[p] <= baseline:
        fail.append("zero_prominence")
        return out
    left = next((i for i in range(p-1,-1,-1) if y[i]<=half),None)
    right = next((i for i in range(p+1,len(y)) if y[i]<=half),None)
    if left is None or right is None:
        fail.append("unbracketed_half_prominence")
        return out
    if p-left<min_halfwidth_samples or right-p<min_halfwidth_samples:
        fail.append("underresolved_halfwidth")
        return out
    def cross(i,j):
        return x[i]+(half-y[i])*(x[j]-x[i])/(y[j]-y[i])
    a,b = cross(left,left+1),cross(right-1,right)
    if b<=a:
        fail.append("invalid_crossings")
        return out
    out.update(resolved=True,resonance_um=x[p],fwhm_um=b-a)
    return out
