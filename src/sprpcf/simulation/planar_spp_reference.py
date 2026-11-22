"""Analytical TM surface-plasmon reference for a planar metal/dielectric half-space.

Independent closed-form Maxwell dispersion benchmark, NOT a PCF eigenmode solver.
No COMSOL, PML, thin-film, geometry-frozen, or sensor qualification is implied.
Convention: exp(i*beta*z-i*omega*t); wavelength is in micrometres.
"""
from __future__ import annotations

import cmath
import math


def planar_spp_reference(wavelength_um: float, metal_epsilon: complex,
                         dielectric_epsilon: float) -> dict[str, float | bool]:
    """Compute beta=k0*sqrt(eps_m*eps_d/(eps_m+eps_d)) and dB/cm power loss.

    Domain: passive metal, lossless dielectric, Re(eps_m)<-eps_d.
    This operating-domain filter does not establish an experimentally observed mode.
    """
    if not isinstance(wavelength_um, (int, float)) or not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("wavelength_um must be finite and positive")
    if not isinstance(dielectric_epsilon, (int, float)) or not math.isfinite(dielectric_epsilon) or dielectric_epsilon <= 0:
        raise ValueError("dielectric_epsilon must be finite and positive")
    try:
        eps_m = complex(metal_epsilon)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("metal epsilon must be finite") from exc
    if not (math.isfinite(eps_m.real) and math.isfinite(eps_m.imag)):
        raise ValueError("metal epsilon must be finite")
    if eps_m.imag <= 0 or eps_m.real >= -dielectric_epsilon:
        raise ValueError("outside passive TM-SPP reference domain")
    n_eff = cmath.sqrt(eps_m * dielectric_epsilon / (eps_m + dielectric_epsilon))
    if n_eff.real <= 0 or n_eff.imag < 0:
        raise ValueError("invalid forward attenuating branch")
    beta = (2 * math.pi / wavelength_um) * n_eff
    attenuation = (20 / math.log(10)) * beta.imag * 1e4
    if not all(math.isfinite(v) for v in (beta.real, beta.imag, attenuation)):
        raise ValueError("nonfinite planar dispersion")
    return {
        "n_eff_real": n_eff.real, "n_eff_imag": n_eff.imag,
        "beta_real_per_um": beta.real, "beta_imag_per_um": beta.imag,
        "power_attenuation_db_per_cm": attenuation,
        "publication_qualified": False, "pcf_mode_verified": False,
        "comsol_verified": False,
    }
