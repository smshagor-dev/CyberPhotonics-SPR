"""Planar TM Maxwell thin-film reference; software validation only, NOT PCF FEM/COMSOL.

exp(-i omega t) convention. Optical epsilon inputs require independent provenance.
No publication/physics qualification is implied.
"""
import cmath
import math


def _kz(eps, q):
    z = cmath.sqrt(complex(eps) - q*q)
    return -z if z.imag < 0 or (abs(z.imag) < 1e-14 and z.real < 0) else z


def tm_reflectance(wavelength_um, theta_rad, incident_eps, substrate_eps, layers):
    """Power reflectance of isotropic nonmagnetic planar stack.

    layers: ordered incidence-to-substrate (relative_epsilon, thickness_um).
    """
    if isinstance(wavelength_um, bool) or not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("invalid wavelength")
    if isinstance(theta_rad, bool) or not math.isfinite(theta_rad) or abs(theta_rad) >= math.pi/2:
        raise ValueError("invalid angle")
    ei, es = complex(incident_eps), complex(substrate_eps)
    if not all(map(math.isfinite, (ei.real, ei.imag, es.real, es.imag))):
        raise ValueError("nonfinite media")
    if ei.imag != 0 or ei.real <= 0 or es.imag < 0:
        raise ValueError("invalid incident medium or active substrate")
    q = math.sqrt(ei.real)*math.sin(theta_rad)
    yi, y = ei/_kz(ei,q), es/_kz(es,q)
    for eps, thickness in reversed(list(layers)):
        eps = complex(eps)
        if not all(map(math.isfinite, (eps.real, eps.imag))) or eps.imag < 0:
            raise ValueError("nonpassive or nonfinite layer")
        if isinstance(thickness, bool) or not math.isfinite(thickness) or thickness <= 0:
            raise ValueError("invalid layer thickness")
        kz = _kz(eps,q)
        eta = eps/kz
        delta = 2*math.pi*thickness*kz/wavelength_um
        if abs(delta.imag) > 40:
            raise ValueError("thick evanescent layer: use stable scattering matrix")
        t = cmath.tan(delta)
        den = eta - 1j*y*t
        if abs(den) < 1e-14:
            raise ArithmeticError("singular admittance transform")
        y = eta*(y - 1j*eta*t)/den
    r = (yi-y)/(yi+y)
    reflectance = abs(r)**2
    if not math.isfinite(reflectance) or reflectance > 1+1e-8:
        raise ArithmeticError("unphysical passive reflectance")
    return min(reflectance,1.0)
