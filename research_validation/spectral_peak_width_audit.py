"""Conservative, software-only wavelength-grid convergence audit.

Never interprets agreement of sampled peaks as independent Maxwell validation.
"""
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class Audit:
    refinement_ratio: float
    resonance_drift_nm: float
    fwhm_drift_nm: float
    passed: bool
    physics_qualified: bool = False
    publication_ready: bool = False


def _peak_width(w, y):
    if len(w) != len(y) or len(w) < 7:
        raise ValueError("at least seven matched samples required")
    if any(not isfinite(float(v)) for v in (*w, *y)):
        raise ValueError("nonfinite data")
    if any(b <= a for a, b in zip(w, w[1:])):
        raise ValueError("nonmonotonic wavelengths")
    p = max(range(len(y)), key=y.__getitem__)
    if p < 2 or p > len(y) - 3:
        raise ValueError("peak insufficiently interior")
    baseline = min(y[0], y[-1])
    if y[p] <= baseline:
        raise ValueError("no peak contrast")
    half = baseline + (y[p] - baseline) / 2
    left = [i for i in range(p) if y[i] <= half < y[i+1]]
    right = [i for i in range(p, len(y)-1) if y[i] >= half > y[i+1]]
    if len(left) != 1 or len(right) != 1:
        raise ValueError("unresolved or ambiguous half-height crossings")
    def crossing(i):
        return w[i] + (half-y[i])*(w[i+1]-w[i])/(y[i+1]-y[i])
    width = crossing(right[0]) - crossing(left[0])
    if width <= 0:
        raise ValueError("nonpositive FWHM")
    return w[p], width


def audit(coarse_w, coarse_loss, fine_w, fine_loss, *,
          max_resonance_drift_nm, max_fwhm_drift_nm):
    if not all(isfinite(v) and v > 0 for v in
               (max_resonance_drift_nm, max_fwhm_drift_nm)):
        raise ValueError("positive finite tolerances required")
    if len(coarse_w) < 7 or len(fine_w) < 7:
        raise ValueError("insufficient grid")
    if abs(coarse_w[0]-fine_w[0]) > 1e-10 or abs(coarse_w[-1]-fine_w[-1]) > 1e-10:
        raise ValueError("different wavelength domains")
    def step(w):
        return max(b-a for a,b in zip(w,w[1:]))
    cstep, fstep = step(coarse_w), step(fine_w)
    if not fstep < cstep:
        raise ValueError("grid not refined")
    cr, cw = _peak_width(coarse_w, coarse_loss)
    fr, fw = _peak_width(fine_w, fine_loss)
    dr, dw = abs(cr-fr), abs(cw-fw)
    return Audit(cstep/fstep, dr, dw,
                 dr <= max_resonance_drift_nm and dw <= max_fwhm_drift_nm)
