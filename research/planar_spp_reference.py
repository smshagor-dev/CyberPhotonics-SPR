"""Independent planar-interface TM SPP dispersion reference (NOT a PCF solver).

Analytic infinite, flat, isotropic metal/dielectric half-space reference.
Useful as a unit/branch/sign sanity check for a separate numerical Maxwell solver.
It cannot validate finite Au films, cylindrical PCF geometry, PML or resonances.
"""
from __future__ import annotations

import cmath
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanarSPP:
    wavelength_um: float
    analyte_ri: float
    epsilon_metal: complex
    n_eff: complex
    beta_per_um: complex
    intensity_length_um: float
    attenuation_db_per_cm: float
    publication_qualified: bool = False


def planar_tm_spp(wavelength_um: float, analyte_ri: float, epsilon_metal: complex) -> PlanarSPP:
    """Return passive TM SPP branch with exp(i beta z - i omega t) convention.

    The field intensity decays as exp(-2 Im(beta) z); dB/cm uses 10/log(10).
    Caller must supply independently sourced dispersive epsilon_metal.
    """
    if not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("wavelength_um must be finite and positive")
    if not math.isfinite(analyte_ri) or analyte_ri <= 0:
        raise ValueError("analyte_ri must be finite and positive")
    metal = complex(epsilon_metal)
    if not all(map(math.isfinite, (metal.real, metal.imag))):
        raise ValueError("metal permittivity must be finite")
    if metal.imag <= 0:
        raise ValueError("passive exp(-i omega t) convention requires Im(epsilon_metal)>0")
    dielectric = analyte_ri ** 2
    if metal.real >= -dielectric:
        raise ValueError("outside bound planar SPP region Re(eps_m)<-eps_d")
    ratio = metal * dielectric / (metal + dielectric)
    n_eff = cmath.sqrt(ratio)
    if n_eff.real < 0:
        n_eff = -n_eff
    beta = 2 * math.pi * n_eff / wavelength_um
    if beta.imag <= 0:
        raise ValueError("nonattenuating/growing branch is invalid for passive SPP")
    length = 1 / (2 * beta.imag)
    db_cm = (20 / math.log(10)) * beta.imag * 1e4
    return PlanarSPP(wavelength_um, analyte_ri, metal, n_eff, beta, length, db_cm)
