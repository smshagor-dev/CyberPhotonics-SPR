"""Independent TM boundary-condition residual for planar SPP modes.

Software-only limiting-case check, not a second full-vector Maxwell eigensolver,
PCF evidence, measured Au verification, or publication qualification.
"""
from __future__ import annotations
import cmath
import math


def audit_tm_boundary(*, wavelength_um: float, dielectric_ri: float,
                      epsilon_m_real: float, epsilon_m_imag: float,
                      neff_real: float, neff_imag: float) -> dict:
    values = (wavelength_um, dielectric_ri, epsilon_m_real, epsilon_m_imag,
              neff_real, neff_imag)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or
           not math.isfinite(v) for v in values):
        raise ValueError("All numerical inputs must be finite real scalars")
    if wavelength_um <= 0 or dielectric_ri <= 0 or epsilon_m_imag <= 0:
        raise ValueError("Positive wavelength, dielectric RI, passive metal loss required")
    eps_d = dielectric_ri ** 2
    eps_m = complex(epsilon_m_real, epsilon_m_imag)
    if epsilon_m_real >= -eps_d:
        raise ValueError("Not in bound-SPP negative-permittivity regime")
    n = complex(neff_real, neff_imag)
    if n.real <= 0 or n.imag <= 0:
        raise ValueError("Forward passive mode required")
    # Positive-real roots enforce transverse decay.
    kd = cmath.sqrt(n*n - eps_d)
    km = cmath.sqrt(n*n - eps_m)
    if kd.real <= 0 or km.real <= 0:
        raise ValueError("Fields do not decay away from the interface")
    # Tangential H and E continuity: kappa_d/eps_d + kappa_m/eps_m = 0.
    td = kd / eps_d
    tm = km / eps_m
    residual = abs(td + tm) / (abs(td) + abs(tm))
    if not math.isfinite(residual):
        raise ValueError("Nonfinite boundary residual")
    k0 = 2*math.pi/wavelength_um
    return {"normalized_tm_boundary_residual": residual,
            "dielectric_decay_per_um": k0*kd.real,
            "metal_decay_per_um": k0*km.real,
            "independent_full_vector_verified": False,
            "publication_ready": False, "comsol_verified": False}
