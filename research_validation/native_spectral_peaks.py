"""Fail-closed spectral postprocessing; never establishes physics provenance."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Peak:
    resonance_nm: float
    fwhm_nm: float
    peak_loss_db_cm: float
    baseline_db_cm: float
    prominence_db_cm: float

def isolated_peak(wavelength_nm, loss_db_cm, *, min_points=9, min_prominence=0.0):
    """Reject clipped, multiple, undersampled, negative or nonfinite spectra."""
    x = [float(v) for v in wavelength_nm]
    y = [float(v) for v in loss_db_cm]
    if len(x) != len(y) or len(x) < min_points or min_points < 5:
        raise ValueError("insufficient or mismatched spectrum")
    if any(not math.isfinite(v) for v in x+y) or any(x[i+1] <= x[i] for i in range(len(x)-1)):
        raise ValueError("nonfinite or unordered spectrum")
    if any(v < 0 for v in y):
        raise ValueError("negative passive attenuation")
    maxima = [i for i in range(1,len(y)-1) if y[i]>y[i-1] and y[i]>y[i+1]]
    if len(maxima) != 1 or y[0]>=y[maxima[0]] or y[-1]>=y[maxima[0]]:
        raise ValueError("peak absent, multimodal, or clipped")
    i = maxima[0]
    if i < 2 or len(x)-i-1 < 2:
        raise ValueError("peak insufficiently sampled")
    baseline = max(min(y[:i]), min(y[i+1:]))
    prominence = y[i]-baseline
    if prominence <= max(0.0,min_prominence):
        raise ValueError("insufficient prominence")
    half = baseline + prominence/2
    left = next((j for j in range(i-1,-1,-1) if y[j]<=half),None)
    right = next((j for j in range(i+1,len(y)) if y[j]<=half),None)
    if left is None or right is None or left==0 or right==len(x)-1:
        raise ValueError("unbracketed half-prominence crossings")
    def cross(a,b):
        if y[a]==y[b]:
            raise ValueError("flat crossing")
        return x[a]+(half-y[a])*(x[b]-x[a])/(y[b]-y[a])
    xl = cross(left,left+1)
    xr = cross(right-1,right)
    h0=x[i]-x[i-1]; h1=x[i+1]-x[i]
    s0=(y[i]-y[i-1])/h0; s1=(y[i+1]-y[i])/h1
    a=(s1-s0)/(h0+h1)
    b=s0+a*h0
    if a>=0:
        raise ValueError("nonconcave peak")
    vertex=x[i]-b/(2*a)
    if not x[i-1]<=vertex<=x[i+1]:
        raise ValueError("vertex outside local bracket")
    return Peak(vertex,xr-xl,y[i],baseline,prominence)

def fixed_geometry_sensitivity(ri, peaks_nm, geometry_hashes):
    """Adjacent nm/RIU differences; requires a single immutable geometry."""
    r=list(map(float,ri)); p=list(map(float,peaks_nm))
    if len(r)<3 or len(r)!=len(p) or len(r)!=len(geometry_hashes):
        raise ValueError("incomplete RI series")
    if len(set(geometry_hashes))!=1 or not geometry_hashes[0]:
        raise ValueError("geometry changed")
    if any(not math.isfinite(v) for v in r+p) or any(r[i+1]<=r[i] for i in range(len(r)-1)):
        raise ValueError("invalid RI series")
    return [(p[i+1]-p[i])/(r[i+1]-r[i]) for i in range(len(r)-1)]
