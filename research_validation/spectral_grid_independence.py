"""Fail-closed spectral-grid independence prerequisite for convergence claims.

Software-only validation; never qualifies PCF-SPR Maxwell physics.
"""
from math import isfinite

def validate_spectral_grids(coarse_wavelength_um, fine_wavelength_um, *, min_refinement_ratio=1.5, endpoint_tolerance_um=1e-10):
    c, f = list(coarse_wavelength_um), list(fine_wavelength_um)
    if not isfinite(min_refinement_ratio) or min_refinement_ratio <= 1:
        raise ValueError("invalid refinement ratio threshold")
    if not isfinite(endpoint_tolerance_um) or endpoint_tolerance_um < 0:
        raise ValueError("invalid endpoint tolerance")
    if len(c) < 11 or len(f) < 11:
        raise ValueError("undersampled spectral grid")
    if any(not isfinite(v) for v in c + f):
        raise ValueError("nonfinite wavelength")
    if any(b <= a for a,b in zip(c,c[1:])) or any(b <= a for a,b in zip(f,f[1:])):
        raise ValueError("nonmonotonic wavelength grid")
    if abs(c[0]-f[0]) > endpoint_tolerance_um or abs(c[-1]-f[-1]) > endpoint_tolerance_um:
        raise ValueError("spectral domain mismatch")
    if len(f) <= len(c):
        raise ValueError("fine grid not finer")
    hc=max(b-a for a,b in zip(c,c[1:]))
    hf=max(b-a for a,b in zip(f,f[1:]))
    ratio=hc/hf
    if ratio < min_refinement_ratio:
        raise ValueError("insufficient spectral grid refinement")
    return {"grid_refinement_verified":True,"max_spacing_ratio":ratio,
            "coarse_samples":len(c),"fine_samples":len(f),
            "physics_qualified":False,"publication_ready":False}
