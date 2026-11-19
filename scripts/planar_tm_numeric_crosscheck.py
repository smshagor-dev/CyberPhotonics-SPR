"""Numerical planar TM boundary-root limiting-case check; NOT PCF/COMSOL evidence.

Illustrative permittivities are not independently authenticated Au measurements.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import root


def _decay(z: complex) -> complex:
    q = np.sqrt(complex(z))
    return -q if q.real < 0 else q


def check(eps_m: complex, eps_d: complex, wavelength_um: float) -> dict:
    if not (np.isfinite([eps_m.real, eps_m.imag, eps_d.real, eps_d.imag, wavelength_um]).all()
            and wavelength_um > 0 and eps_d.real > 0 and eps_d.imag == 0
            and eps_m.imag > 0 and eps_m.real < -eps_d.real):
        raise ValueError("requires finite passive metal, lossless dielectric, bound-SPP regime")

    def residual(x):
        n = complex(*x)
        f = _decay(n*n-eps_m)/eps_m + _decay(n*n-eps_d)/eps_d
        return [f.real, f.imag]

    guess = complex(np.sqrt(eps_d.real)*1.01, 0.001)
    sol = root(residual, [guess.real, guess.imag], tol=1e-12)
    n = complex(*sol.x)
    norm = float(np.linalg.norm(residual(sol.x)))
    if not sol.success and norm > 1e-10:
        raise RuntimeError(f"boundary root failed: {sol.message}; residual={norm}")
    if norm > 1e-9 or n.real <= 0 or n.imag <= 0:
        raise RuntimeError("nonphysical or unconverged numerical TM root")
    analytic = np.sqrt(eps_m*eps_d/(eps_m+eps_d))
    if analytic.real < 0:
        analytic = -analytic
    k0 = 2*np.pi/wavelength_um
    return {
        "numerical_neff": [float(n.real), float(n.imag)],
        "analytical_neff": [float(analytic.real), float(analytic.imag)],
        "absolute_neff_error": float(abs(n-analytic)),
        "normalized_boundary_residual": norm,
        "power_loss_db_per_cm": float((20/np.log(10))*k0*n.imag*1e4),
        "qualification": {
            "planar_limiting_case_only": True,
            "measured_au_verified": False,
            "pcf_full_vector_verified": False,
            "comsol_verified": False,
            "physics_qualified": False,
            "publication_ready": False,
        },
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--eps-metal-real", type=float, default=-12)
    p.add_argument("--eps-metal-imag", type=float, default=1.2)
    p.add_argument("--eps-dielectric", type=float, default=2.1)
    p.add_argument("--wavelength-um", type=float, default=0.8)
    a = p.parse_args()
    result = check(complex(a.eps_metal_real, a.eps_metal_imag),
                   complex(a.eps_dielectric), a.wavelength_um)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x", encoding="utf-8") as f:
        json.dump({"source_type": "illustrative_analytic_planar_tm",
                   "solver": "scipy.optimize.root", **result},
                  f, indent=2, allow_nan=False)
        f.write("\n")


if __name__ == "__main__":
    main()
