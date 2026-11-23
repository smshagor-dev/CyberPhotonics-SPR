"""Fail-closed propagation-regime audit for analytic planar TM SPP references.

This is NOT a full-vector PCF eigenmode calculation or measured-Au evidence.
"""
from __future__ import annotations
import math
from scripts.planar_spp_reference import planar_spp_diagnostics


def audit_planar_propagation(epsilon_d: float, epsilon_m: complex, wavelength_um: float) -> dict:
    """Report, rather than hide, below-light-line and overdamped SPP roots.

    Above-light-line and propagation-length criteria are diagnostics, not
    sufficient evidence of a bound PCF-SPR eigenmode.
    """
    d = planar_spp_diagnostics(epsilon_d, epsilon_m, wavelength_um)
    beta = d["beta_rad_per_um"]
    k0 = 2 * math.pi / wavelength_um
    light_line = k0 * math.sqrt(epsilon_d)
    above = beta.real > light_line
    ratio = beta.imag / beta.real if beta.real > 0 else math.inf
    # The intensity 1/e propagation length contains beta_real/(4*pi*beta_imag)
    # phase cycles. This is a diagnostic, not a universal bound-mode criterion.
    intensity_length_um = math.inf if beta.imag == 0 else 1 / (2 * beta.imag)
    intensity_efold_phase_cycles = (
        math.inf if beta.imag == 0 else beta.real / (4 * math.pi * beta.imag)
    )
    residual_ok = math.isfinite(d["tm_boundary_relative_residual"]) and d["tm_boundary_relative_residual"] <= 1e-8
    decays = bool(d["decays_both_halfspaces"])
    return {
        "above_dielectric_light_line": above,
        "attenuation_to_phase_ratio": ratio,
        "intensity_attenuation_length_um": intensity_length_um,
        "intensity_efold_phase_cycles": intensity_efold_phase_cycles,
        "overdamped": not math.isfinite(ratio) or ratio >= 1,
        "interface_residual_pass": residual_ok,
        "transverse_decay_pass": decays,
        "planar_reference_screen_pass": bool(above and ratio < 1 and residual_ok and decays),
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
    }
