"""Fail-closed sampled-spectrum resolution audit; not a physics qualification."""
import math

def audit_spectrum(wavelength_nm, loss_db_cm):
    x = [float(v) for v in wavelength_nm]
    y = [float(v) for v in loss_db_cm]
    if len(x) != len(y) or len(x) < 7 or any(not math.isfinite(v) for v in x+y):
        raise ValueError("invalid spectrum")
    if any(b <= a for a,b in zip(x,x[1:])) or min(y) < 0:
        raise ValueError("nonphysical samples or wavelength ordering")
    peaks = [i for i in range(1,len(y)-1) if y[i] > y[i-1] and y[i] > y[i+1]]
    if len(peaks) != 1:
        raise ValueError("ambiguous or unresolved peak")
    p=peaks[0]
    base=min(y[0],y[-1])
    if y[p] <= base:
        raise ValueError("peak not above endpoint baseline")
    half=base+(y[p]-base)/2
    left=[i for i in range(p) if y[i] <= half < y[i+1]]
    right=[i for i in range(p,len(y)-1) if y[i] > half >= y[i+1]]
    if len(left)!=1 or len(right)!=1:
        raise ValueError("FWHM unbracketed or ambiguous")
    def cross(i):
        return x[i]+(half-y[i])*(x[i+1]-x[i])/(y[i+1]-y[i])
    a,b=cross(left[0]),cross(right[0])
    width=b-a
    local_step=max(x[p]-x[p-1],x[p+1]-x[p])
    n_inside=sum(a <= v <= b for v in x)
    if width <= 0:
        raise ValueError("invalid FWHM")
    return {"sampled_peak_nm":x[p], "fwhm_nm":width,
            "max_local_step_nm":local_step, "samples_within_fwhm":n_inside,
            "points_per_fwhm":width/local_step,
            "spectral_resolution_gate":n_inside>=7 and width/local_step>=6,
            "physics_qualified":False,"publication_ready":False,"comsol_verified":False}
