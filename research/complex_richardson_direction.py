"""Fail-closed complex Richardson diagnostic; NOT qualified Maxwell evidence."""
from cmath import phase
from math import isfinite, log, pi

def extrapolate(h, neff, *, phase_limit_deg=15, min_drift=1e-12):
    out = {"valid": False, "physics_qualified": False, "publication_ready": False, "reason": None}
    if len(h) != 3 or len(neff) != 3:
        out["reason"] = "three grids required"; return out
    if any(type(v) not in (int,float) or not isfinite(v) or v <= 0 for v in h):
        out["reason"] = "invalid grid"; return out
    if any(type(z) not in (int,float,complex) or not isfinite(complex(z).real)
           or not isfinite(complex(z).imag) for z in neff):
        out["reason"] = "invalid eigenvalue"; return out
    if not h[0] > h[1] > h[2]:
        out["reason"] = "nondecreasing mesh size"; return out
    r0,r1 = h[0]/h[1],h[1]/h[2]
    if abs(r0-r1)/max(r0,r1) > .01:
        out["reason"] = "nongeometric refinement"; return out
    z0,z1,z2 = map(complex,neff)
    d0,d1 = z0-z1,z1-z2
    if abs(d0) <= min_drift or abs(d1) <= min_drift:
        out["reason"] = "drift below resolution"; return out
    ratio = d0/d1
    phase_error = abs(phase(ratio))*180/pi
    if phase_error > phase_limit_deg:
        out["reason"] = "complex drift direction inconsistent"; return out
    if abs(ratio) <= 1:
        out["reason"] = "nondecreasing drift"; return out
    order = log(abs(ratio))/log(r1)
    if not isfinite(order) or order <= 0:
        out["reason"] = "invalid order"; return out
    # d1 = z_medium - z_fine, so the correction has a MINUS sign.
    z_limit = z2 - d1/(r1**order-1)
    out.update(valid=True, reason=None, observed_order=order,
               phase_error_deg=phase_error, limit_real=z_limit.real,
               limit_imag=z_limit.imag, fine_gap=abs(z_limit-z2))
    return out
