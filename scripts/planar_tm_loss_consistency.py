"""Dimensional/passivity audit of numerical planar TM roots; NOT PCF/COMSOL evidence."""
import math
from scripts.planar_tm_root_audit import numerical_planar_tm_root


def planar_tm_loss_consistency(wavelength_um, eps_m, eps_d):
    root = numerical_planar_tm_root(wavelength_um, eps_m, eps_d)
    n_imag = root["neff_imag"]
    if not math.isfinite(n_imag) or n_imag <= 0:
        raise RuntimeError("nonpassive numerical root")
    beta_imag_per_um = (2 * math.pi / wavelength_um) * n_imag
    # Intensity I(z)=I(0) exp(-2 Im(beta) z); 1 cm = 10000 um.
    loss_db_per_cm = (20 / math.log(10)) * beta_imag_per_um * 10000
    intensity_length_um = 1 / (2 * beta_imag_per_um)
    alternative = (40 * math.pi / math.log(10)) * n_imag * 10000 / wavelength_um
    if not math.isclose(loss_db_per_cm, alternative, rel_tol=1e-13):
        raise RuntimeError("attenuation conversion inconsistency")
    return {
        **root,
        "beta_imag_per_um": beta_imag_per_um,
        "intensity_1e_length_um": intensity_length_um,
        "loss_db_per_cm": loss_db_per_cm,
        "loss_definition": "intensity_dB_per_cm_20log10e_Imbeta",
        "source_material_verified": False,
        "independent_fem_verified": False,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
    }
