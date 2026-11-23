"""Independent passive planar TM interface reference; NOT PCF or COMSOL evidence."""
from __future__ import annotations
import cmath
import math


def passive_kz(epsilon: complex, beta: complex, k0: float) -> complex:
    """Outgoing/decaying transverse wave number, exp(+i kz z - i wt)."""
    if not math.isfinite(k0) or k0 <= 0:
        raise ValueError("positive finite vacuum wavenumber required")
    e, b = complex(epsilon), complex(beta)
    if not all(math.isfinite(v) for v in (e.real, e.imag, b.real, b.imag)):
        raise ValueError("finite optical parameters required")
    if e.imag < -1e-12:
        raise ValueError("gain media not supported")
    z = cmath.sqrt(k0*k0*e-b*b)
    if z.imag < 0 or (abs(z.imag) < 1e-14 and z.real < 0):
        z = -z
    return z


def tm_reflectance(epsilon_inc: complex, epsilon_out: complex, theta_rad: float) -> float:
    """Semi-infinite TM Fresnel power reflectance; lossless dielectric incidence."""
    ei, eo = complex(epsilon_inc), complex(epsilon_out)
    if ei.imag != 0 or ei.real <= 0 or not math.isfinite(theta_rad) or not 0 <= theta_rad < math.pi/2:
        raise ValueError("lossless dielectric incidence and physical angle required")
    k0 = 2*math.pi
    beta = k0*math.sqrt(ei.real)*math.sin(theta_rad)
    ki, ko = passive_kz(ei,beta,k0), passive_kz(eo,beta,k0)
    yi, yo = ki/ei, ko/eo
    r = (yi-yo)/(yi+yo)
    R = abs(r)**2
    if not math.isfinite(R) or R > 1+1e-10:
        raise ArithmeticError("passive interface violated energy bound")
    return min(R,1.0)
