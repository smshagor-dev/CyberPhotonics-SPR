"""Analytic planar TM surface-plasmon reference; NOT PCF or COMSOL evidence."""
from __future__ import annotations
import cmath
import math
from scripts.planar_tm_fresnel_reference import passive_kz


def planar_spp_beta(epsilon_d: complex, epsilon_m: complex, wavelength_um: float) -> complex:
    """Closed-form planar SPP beta [rad/um], exp(i beta x-i omega t)."""
    ed, em = complex(epsilon_d), complex(epsilon_m)
    if not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("wavelength must be finite and positive")
    if not all(math.isfinite(x) for x in (ed.real, ed.imag, em.real, em.imag)):
        raise ValueError("finite permittivities required")
    if ed.imag != 0 or ed.real <= 0 or em.imag < 0:
        raise ValueError("lossless positive dielectric and passive metal required")
    if em.real >= 0:
        raise ValueError("negative-real-permittivity metal required for this SPP reference")
    if em.imag == 0 and em.real >= -ed.real:
        raise ValueError("lossless metal does not support a bound TM SPP at this interface")
    if ed + em == 0:
        raise ValueError("surface plasmon pole: zero denominator")
    beta = (2 * math.pi / wavelength_um) * cmath.sqrt(ed * em / (ed + em))
    if not all(math.isfinite(x) for x in (beta.real, beta.imag)):
        raise ArithmeticError("nonfinite SPP propagation constant")
    if beta.real < 0:
        beta = -beta
    if beta.imag < -1e-12:
        raise ArithmeticError("gain-like SPP propagation branch")
    return beta


def planar_spp_diagnostics(epsilon_d: complex, epsilon_m: complex, wavelength_um: float) -> dict:
    """Check transverse TM admittance residual and decay; never qualify PCF."""
    k0 = 2 * math.pi / wavelength_um
    beta = planar_spp_beta(epsilon_d, epsilon_m, wavelength_um)
    kd = passive_kz(epsilon_d, beta, k0)
    km = passive_kz(epsilon_m, beta, k0)
    residual = abs(kd / epsilon_d + km / epsilon_m)
    scale = abs(kd / epsilon_d) + abs(km / epsilon_m)
    return {
        "beta_rad_per_um": beta,
        "neff": beta / k0,
        "intensity_length_um": math.inf if beta.imag == 0 else 1 / (2 * beta.imag),
        "tm_boundary_relative_residual": residual / scale if scale else math.inf,
        "decays_both_halfspaces": kd.imag > 0 and km.imag > 0,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
    }
