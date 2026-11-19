"""Analytic planar TM SPP reference, NOT PCF full-vector FEM or COMSOL.

Assumes homogeneous isotropic metal and lossless dielectric, exp(i beta z-i omega t).
Only a limiting-case material/sign-convention diagnostic; no sensor claims.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import cmath
import json
import math
from pathlib import Path

@dataclass(frozen=True)
class PlanarSPP:
    wavelength_um: float
    analyte_ri: float
    epsilon_m_real: float
    epsilon_m_imag: float
    neff_real: float
    neff_imag: float
    beta_imag_per_um: float
    power_loss_db_per_cm: float
    metal_decay_real_per_um: float
    dielectric_decay_real_per_um: float
    source_id: str
    source_independently_verified: bool = False
    pcf_full_vector_verified: bool = False
    publication_ready: bool = False
    comsol_verified: bool = False

def planar_tm_spp(*, wavelength_um: float, analyte_ri: float,
                  epsilon_m_real: float, epsilon_m_imag: float,
                  source_id: str, source_independently_verified: bool = False) -> PlanarSPP:
    values = (wavelength_um, analyte_ri, epsilon_m_real, epsilon_m_imag)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("Physical inputs must be finite real numbers")
    if wavelength_um <= 0 or analyte_ri <= 0:
        raise ValueError("Wavelength and analyte RI must be positive")
    if epsilon_m_imag <= 0:
        raise ValueError("Passive metal requires Im(epsilon)>0 for exp(-i omega t)")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("Material source identifier required")
    if not isinstance(source_independently_verified, bool):
        raise ValueError("Source verification must be boolean")
    eps_d = analyte_ri ** 2
    if epsilon_m_real >= -eps_d:
        raise ValueError("No bound planar SPP in negative-permittivity regime")
    eps_m = complex(epsilon_m_real, epsilon_m_imag)
    neff = cmath.sqrt(eps_m * eps_d / (eps_m + eps_d))
    k0 = 2 * math.pi / wavelength_um
    beta = k0 * neff
    kappa_d = cmath.sqrt(beta**2 - k0**2 * eps_d)
    kappa_m = cmath.sqrt(beta**2 - k0**2 * eps_m)
    if beta.imag <= 0 or kappa_d.real <= 0 or kappa_m.real <= 0:
        raise ValueError("Mode does not attenuate and decay on both sides")
    loss = 20 / math.log(10) * beta.imag * 1e4
    return PlanarSPP(wavelength_um, analyte_ri, epsilon_m_real, epsilon_m_imag,
                     neff.real, neff.imag, beta.imag, loss, kappa_m.real,
                     kappa_d.real, source_id, source_independently_verified)

def write_reference_json(path: str | Path, result: PlanarSPP) -> None:
    """Immutable, deterministic diagnostic output; never upgrades evidence."""
    path = Path(path)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
