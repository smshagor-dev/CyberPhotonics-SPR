"""Analytic planar TM SPP reference; NOT PCF/FEM/COMSOL evidence."""
import cmath
import math

def planar_spp(wavelength_um: float, eps_m: complex, eps_d: float) -> dict:
    """exp(-i omega t), exp(i beta z); passive metal Im(epsilon)>0."""
    if not (math.isfinite(wavelength_um) and wavelength_um > 0):
        raise ValueError("invalid wavelength")
    if not (math.isfinite(eps_d) and eps_d > 0):
        raise ValueError("invalid dielectric")
    if not (math.isfinite(eps_m.real) and math.isfinite(eps_m.imag)):
        raise ValueError("nonfinite metal")
    if eps_m.imag <= 0 or eps_m.real >= -eps_d:
        raise ValueError("outside conventional passive TM SPP regime")
    k0 = 2 * math.pi / wavelength_um
    beta = k0 * cmath.sqrt(eps_m * eps_d / (eps_m + eps_d))
    if beta.real < 0:
        beta = -beta
    if beta.imag <= 0:
        raise ValueError("nonattenuating branch")
    kd = cmath.sqrt(beta**2 - k0**2 * eps_d)
    km = cmath.sqrt(beta**2 - k0**2 * eps_m)
    if kd.real < 0: kd = -kd
    if km.real < 0: km = -km
    if min(kd.real, km.real) <= 0:
        raise ValueError("unbound branch")
    residual = abs(kd / eps_d + km / eps_m) / max(abs(kd / eps_d), abs(km / eps_m))
    if residual > 1e-7:
        raise ValueError("TM boundary residual")
    return {
        "wavelength_um": wavelength_um, "eps_d": eps_d,
        "eps_m_real": eps_m.real, "eps_m_imag": eps_m.imag,
        "neff_real": (beta/k0).real, "neff_imag": (beta/k0).imag,
        "beta_real_per_um": beta.real, "beta_imag_per_um": beta.imag,
        "intensity_propagation_length_um": 1/(2*beta.imag),
        "loss_db_per_cm": 20/math.log(10)*beta.imag*1e4,
        "dielectric_decay_length_um": 1/kd.real,
        "metal_decay_length_um": 1/km.real,
        "tm_boundary_relative_residual": residual,
        "solver": "analytic_planar_TM_SPP", "geometry": "single_flat_interface",
        "physics_qualified": False, "publication_ready": False, "comsol_verified": False,
    }
