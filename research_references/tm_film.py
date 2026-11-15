"""Passive three-layer TM Fresnel film reference, not a PCF eigenmode solver.

Time convention exp(-i omega t); optical constants n+ik with k>=0.
This exact planar scattering reference checks metal-film coupling without
claiming PCF resonance or PML convergence.
"""
import cmath
import math


def _kz(epsilon: complex, k_parallel: complex, k0: float) -> complex:
    z = cmath.sqrt((k0 * k0) * epsilon - k_parallel * k_parallel)
    # Outgoing/decaying transmitted branch for exp(+i kz z).
    if z.imag < -1e-13 or (abs(z.imag) <= 1e-13 and z.real < 0):
        z = -z
    return z


def tm_reflection(wavelength_nm: float, thickness_nm: float,
                  eps_incident: complex, eps_film: complex,
                  eps_exit: complex, angle_deg: float) -> complex:
    """Complex p-polarized amplitude for one homogeneous film.

    Incident and exit media must be real positive dielectrics; the film is
    passive (Im epsilon >= 0). Incident angle is real, 0<=angle<90.
    """
    vals = (wavelength_nm, thickness_nm, angle_deg)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError('nonfinite wavelength, thickness or angle')
    if wavelength_nm <= 0 or thickness_nm < 0 or not (0 <= angle_deg < 90):
        raise ValueError('invalid wavelength, thickness or incidence angle')
    es = [complex(eps_incident), complex(eps_film), complex(eps_exit)]
    if not all(math.isfinite(e.real) and math.isfinite(e.imag) for e in es):
        raise ValueError('nonfinite permittivity')
    if any(e.imag < -1e-12 for e in es):
        raise ValueError('active medium not supported')
    if any(abs(es[j].imag)>1e-12 or es[j].real<=0 for j in (0,2)):
        raise ValueError('incident and exit media must be lossless dielectrics')
    if abs(es[1]) < 1e-15:
        raise ValueError('zero film permittivity unsupported')
    k0 = 2 * math.pi / wavelength_nm
    kp = k0 * math.sqrt(es[0].real) * math.sin(math.radians(angle_deg))
    kz = [_kz(e, kp, k0) for e in es]
    # TM admittance Y=epsilon/kz. Fresnel r=(Y_i-Y_j)/(Y_i+Y_j).
    y = [e/k for e,k in zip(es,kz)]
    r01 = (y[0]-y[1])/(y[0]+y[1])
    r12 = (y[1]-y[2])/(y[1]+y[2])
    phase = cmath.exp(2j*kz[1]*thickness_nm)
    return (r01+r12*phase)/(1+r01*r12*phase)


def reflectance(*args, **kwargs):
    return abs(tm_reflection(*args, **kwargs))**2
