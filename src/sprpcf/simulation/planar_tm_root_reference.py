"""Independent numerical planar TM-SPP boundary-condition root reference.

Not a PCF mode, FEM cross-solver validation, or sensor loss prediction.
Uses exp(-i omega t), passive metal, and dimensionless beta/k0.
"""
from __future__ import annotations

import math
import numpy as np
from scipy.optimize import root

from sprpcf.simulation.materials_reference import au_johnson_christy_permittivity


def solve_planar_tm_root(wavelength_um: float, analyte_ri: float) -> dict[str, float]:
    """Solve kappa_m/epsilon_m + kappa_d/epsilon_d = 0 independently.

    The closed-form dispersion initializes the nonlinear root but the
    numerical root solves the unsquared electromagnetic interface condition.
    This is an algebraic reference, NOT an independent PCF Maxwell solver.
    """
    if (not math.isfinite(wavelength_um) or not math.isfinite(analyte_ri)
            or wavelength_um <= 0 or not 1 < analyte_ri < 2):
        raise ValueError("invalid wavelength or analyte RI")
    metal = complex(au_johnson_christy_permittivity(wavelength_um))
    dielectric = analyte_ri**2
    if not (metal.imag > 0 and metal.real < -dielectric):
        raise ValueError("bound passive planar TM-SPP condition not met")
    closed = complex(np.sqrt(metal * dielectric / (metal + dielectric)))
    if closed.real < 0:
        closed = -closed

    def decaying_root(z: complex) -> complex:
        k = complex(np.sqrt(z))
        return -k if k.real < 0 else k

    def residual(values: np.ndarray) -> list[float]:
        beta = complex(float(values[0]), float(values[1]))
        km = decaying_root(beta**2 - metal)
        kd = decaying_root(beta**2 - dielectric)
        boundary = km / metal + kd / dielectric
        return [boundary.real, boundary.imag]

    initial = [closed.real * 1.0001, closed.imag * 1.0001]
    result = root(residual, initial, tol=1e-12)
    numerical = complex(float(result.x[0]), float(result.x[1]))
    relative_error = abs(numerical - closed) / abs(closed)
    boundary_residual = float(np.linalg.norm(residual(result.x)))
    if (not np.isfinite(result.x).all() or relative_error > 1e-8
            or boundary_residual > 1e-8 or numerical.imag <= 0):
        raise RuntimeError("planar TM-SPP nonlinear root failed physical agreement")
    k0_cm = 2 * math.pi / (wavelength_um * 1e-4)
    return {
        "wavelength_um": wavelength_um,
        "analyte_ri": analyte_ri,
        "neff_real": numerical.real,
        "neff_imag": numerical.imag,
        "relative_closed_form_error": float(relative_error),
        "boundary_residual": boundary_residual,
        "intensity_attenuation_db_per_cm": 8.685889638 * k0_cm * numerical.imag,
        "publication_qualified": False,
        "pcf_eigenmode_verified": False,
        "independent_pcf_solver_verified": False,
    }
