"""Unequal-spacing complex eigenvalue convergence diagnostic (software-only).

Requires matched physical mode and immutable geometry/material/PML provenance.
Never qualifies publication physics; actual field/PML/solver validation is separate.
"""
from math import expm1, isfinite, log

def _ratio(p, a, b):
    # Stable expression for (h0**p-h1**p)/(h1**p-h2**p).
    return expm1(p*a) / (-expm1(-p*b))

def observed_order(h, drift_ratio):
    if len(h) != 3 or not all(isfinite(x) and x > 0 for x in h):
        raise ValueError("invalid mesh sizes")
    h0, h1, h2 = h
    if not h0 > h1 > h2 or not isfinite(drift_ratio) or drift_ratio <= 0:
        raise ValueError("invalid refinement sequence or drift")
    a, b = log(h0/h1), log(h1/h2)
    if drift_ratio <= a/b * (1 + 1e-10):
        raise ValueError("no identifiable positive order")
    lo, hi = 0., 1.
    while hi < 64 and _ratio(hi, a, b) < drift_ratio:
        hi *= 2
    if _ratio(hi, a, b) < drift_ratio:
        raise ValueError("order exceeds numerical bound")
    for _ in range(80):
        mid = (lo+hi)/2
        if _ratio(mid, a, b) < drift_ratio:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2

def audit(h, neff, geometry_ids, material_ids, pml_ids, archive_hashes,
          mode_tracking_verified=False):
    out = {"physics_qualified": False, "publication_ready": False,
           "integrity_passed": False, "convergence_diagnostic_passed": False,
           "errors": []}
    try:
        if not all(len(v) == 3 for v in
                   (h, neff, geometry_ids, material_ids, pml_ids, archive_hashes)):
            raise ValueError("three independent meshes required")
        for name, values in (("geometry", geometry_ids), ("material", material_ids),
                             ("PML", pml_ids)):
            if len(set(values)) != 1 or not all(isinstance(v,str) and len(v)==64
                 and all(c in "0123456789abcdef" for c in v) for v in values):
                raise ValueError(name+" provenance mismatch or invalid digest")
        if len(set(archive_hashes)) != 3 or not all(isinstance(v,str) and len(v)==64
                 and all(c in "0123456789abcdef" for c in v) for v in archive_hashes):
            raise ValueError("invalid or reused archive SHA256")
        if not mode_tracking_verified:
            raise ValueError("mode identity unverified")
        if not all(isfinite(x) and x > 0 for x in h):
            raise ValueError("invalid mesh sizes")
        if not all(isfinite(z.real) and isfinite(z.imag) and z.imag >= 0 for z in neff):
            raise ValueError("nonfinite or nonpassive eigenvalue")
        if not h[0] > h[1] > h[2]:
            raise ValueError("mesh sequence not strictly decreasing")
        out["integrity_passed"] = True
        d0, d1 = neff[0]-neff[1], neff[1]-neff[2]
        if abs(d0)==0 or abs(d1)==0:
            raise ValueError("zero drift cannot identify convergence order")
        p = observed_order(h, abs(d0)/abs(d1))
        alignment = (d0.conjugate()*d1).real/(abs(d0)*abs(d1))
        out.update(observed_order=p, complex_direction_cosine=alignment,
                   fine_relative_drift=abs(d1)/max(abs(neff[2]),1e-30))
        if alignment < 0.99:
            raise ValueError("complex eigenvalue branch direction inconsistent")
        denominator = expm1(p*log(h[1]/h[2]))
        if denominator <= 0 or not isfinite(denominator):
            raise ValueError("invalid Richardson denominator")
        extrapolated = neff[2]+(neff[2]-neff[1])/denominator
        if not all(isfinite(v) for v in (extrapolated.real,extrapolated.imag)) or extrapolated.imag < 0:
            raise ValueError("nonpassive or nonfinite extrapolation")
        out.update(richardson_neff=[extrapolated.real,extrapolated.imag],
                   richardson_gap=abs(extrapolated-neff[2]),
                   convergence_diagnostic_passed=True)
    except (ValueError, OverflowError, TypeError, ZeroDivisionError) as exc:
        out["errors"].append(str(exc))
    return out
