"""Independent planar-interface TM SPP analytic reference; NOT PCF physics evidence."""
from __future__ import annotations
import cmath
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class PlanarSPP:
    wavelength_nm: float
    epsilon_m: complex
    epsilon_d: float
    neff: complex
    beta_per_m: complex
    propagation_length_um: float
    decay_metal_nm: float
    decay_dielectric_nm: float

def _passive_sqrt(z: complex) -> complex:
    root = cmath.sqrt(z)
    return -root if root.real < 0 or (root.real == 0 and root.imag < 0) else root

def planar_tm_spp(wavelength_nm: float, epsilon_m: complex, epsilon_d: float) -> PlanarSPP:
    """Planar TM SPP with exp(i beta z - i omega t).

    Metal epsilon MUST come from a separately verified optical dataset.
    This analytic interface is NOT a PCF mode, independent FEM solver,
    geometry/PML convergence check, or source of publication claims.
    """
    if isinstance(wavelength_nm, bool) or not isinstance(wavelength_nm, (int, float)) or not math.isfinite(wavelength_nm) or wavelength_nm <= 0:
        raise ValueError("wavelength_nm must be finite positive")
    if isinstance(epsilon_d, bool) or not isinstance(epsilon_d, (int, float)) or not math.isfinite(epsilon_d) or epsilon_d <= 0:
        raise ValueError("epsilon_d must be finite positive")
    if isinstance(epsilon_m, bool) or not isinstance(epsilon_m, (int, float, complex)):
        raise ValueError("epsilon_m must be complex finite")
    em = complex(epsilon_m)
    if not math.isfinite(em.real) or not math.isfinite(em.imag) or em.imag <= 0 or em.real >= -epsilon_d:
        raise ValueError("requires passive metal and bound-SPP real-permittivity regime")
    k0 = 2 * math.pi / (wavelength_nm * 1e-9)
    neff = _passive_sqrt(em * epsilon_d / (em + epsilon_d))
    beta = k0 * neff
    if beta.imag <= 0 or neff.real <= 0:
        raise ValueError("non-attenuating or backward branch")
    km = _passive_sqrt(beta * beta - k0 * k0 * em)
    kd = _passive_sqrt(beta * beta - k0 * k0 * epsilon_d)
    if km.real <= 0 or kd.real <= 0:
        raise ValueError("unbound transverse field")
    return PlanarSPP(float(wavelength_nm), em, float(epsilon_d), neff, beta,
                     1e6 / (2 * beta.imag), 1e9 / km.real, 1e9 / kd.real)
