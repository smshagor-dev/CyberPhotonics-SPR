"""Conservative spectral resolution diagnostic. Never a physics qualification."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Assessment:
    resolved: bool
    reasons: tuple
    resonance_um: float | None = None
    fwhm_um: float | None = None
    physics_qualified: bool = False
    publication_ready: bool = False
    comsol_verified: bool = False

def assess_spectrum(wavelength_um, loss_db_cm, min_points=21, min_samples_per_fwhm=8):
    x, y = tuple(map(float,wavelength_um)), tuple(map(float,loss_db_cm))
    if len(x)!=len(y) or len(x)<3:
        return Assessment(False,("invalid_length",))
    if not all(map(math.isfinite,x+y)) or min(x)<=0 or min(y)<0:
        return Assessment(False,("nonfinite_or_unphysical_input",))
    if any(b<=a for a,b in zip(x,x[1:])):
        return Assessment(False,("nonincreasing_wavelength",))
    reasons = ["sparse_spectrum"] if len(x)<min_points else []
    peaks=[i for i in range(1,len(y)-1) if y[i]>y[i-1] and y[i]>y[i+1]]
    if len(peaks)!=1:
        return Assessment(False,tuple(reasons+["ambiguous_or_absent_peak"]))
    i=peaks[0]
    baseline=min(y[0],y[-1])
    half=(y[i]+baseline)/2
    left=next((j for j in range(i-1,-1,-1) if y[j]<=half),None)
    right=next((j for j in range(i,len(y)-1) if y[j+1]<=half),None)
    if left is None or right is None:
        return Assessment(False,tuple(reasons+["unbracketed_half_max"]))
    def crossing(j):
        return x[j]+(half-y[j])*(x[j+1]-x[j])/(y[j+1]-y[j])
    width=crossing(right)-crossing(left)
    if not math.isfinite(width) or width<=0:
        return Assessment(False,tuple(reasons+["invalid_fwhm"]))
    if max(b-a for a,b in zip(x,x[1:]))>width/min_samples_per_fwhm:
        reasons.append("underresolved_fwhm")
    return Assessment(not reasons,tuple(reasons),x[i],width)
