"""Analytic planar TM SPP dispersion control; NOT a PCF eigenmode solve.

For exp(-i omega t): beta/k0 = sqrt(eps_m * eps_d / (eps_m + eps_d)).
Material epsilon must be supplied from a separately traceable source.
"""
import cmath
import math


def planar_spp(wavelength_um, eps_m_real, eps_m_imag, analyte_ri):
    for name, value in (("wavelength_um", wavelength_um),
                        ("eps_m_real", eps_m_real),
                        ("eps_m_imag", eps_m_imag), ("analyte_ri", analyte_ri)):
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"{name}: finite numeric scalar required")
    if wavelength_um <= 0 or analyte_ri <= 0 or eps_m_imag < 0:
        raise ValueError("positive wavelength/RI and passive metal required")
    ed = analyte_ri ** 2
    em = complex(eps_m_real, eps_m_imag)
    if abs(em + ed) <= 1e-12 * max(1, abs(em), ed):
        raise ValueError("singular surface-plasmon denominator")
    neff = cmath.sqrt(em * ed / (em + ed))
    if neff.real < 0 or (neff.real == 0 and neff.imag < 0):
        neff = -neff
    if neff.real <= 0 or neff.imag < -1e-12:
        raise ValueError("no supported passive propagating SPP branch")
    loss = (40 * math.pi * 1e4 / math.log(10)) * neff.imag / wavelength_um
    if not math.isfinite(loss):
        raise ValueError("nonfinite attenuation")
    return {"neff_real": neff.real, "neff_imag": neff.imag,
            "attenuation_db_per_cm": loss,
            "source_type": "analytic_planar_interface_control",
            "pcf_physics_qualified": False, "publication_ready": False,
            "comsol_verified": False}
