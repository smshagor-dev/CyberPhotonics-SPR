"""Independent planar Au/analyte TM-SPP dispersion control, NOT PCF evidence.

Uses source-transcribed Johnson-Christy Au n,k from the repository, with no
extrapolation. A complex root of the TM interface boundary equation independently
checks the closed-form dispersion. No finite PCF, PML or sensor performance claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import root

from sprpcf.simulation.materials_reference import (
    au_johnson_christy_permittivity, au_reference_manifest,
)


def _decay_root(z: complex) -> complex:
    value = complex(np.sqrt(complex(z)))
    return -value if value.real < 0 else value


def planar_tm_control(wavelength_um: float, analyte_ri: float) -> dict:
    if not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ValueError("positive finite wavelength required")
    if not math.isfinite(analyte_ri) or not 1.0 < analyte_ri < 2.0:
        raise ValueError("analyte refractive index must be in (1,2)")
    eps_m = complex(au_johnson_christy_permittivity(wavelength_um))
    eps_d = analyte_ri**2
    if eps_m.real >= -eps_d or eps_m.imag <= 0:
        raise ValueError("not in a conventional bound planar TM-SPP regime")
    k0 = 2 * math.pi / wavelength_um
    beta_closed = k0 * np.sqrt(eps_m * eps_d / (eps_m + eps_d))
    beta_closed = complex(beta_closed)
    if beta_closed.real <= 0 or beta_closed.imag <= 0:
        raise ValueError("passive forward TM-SPP branch not obtained")

    def boundary(beta: complex) -> complex:
        km = _decay_root(beta**2 - eps_m * k0**2)
        kd = _decay_root(beta**2 - eps_d * k0**2)
        return eps_m / km + eps_d / kd

    def equations(x: np.ndarray) -> np.ndarray:
        val = boundary(complex(float(x[0]), float(x[1])))
        return np.array([val.real, val.imag], dtype=float)

    result = root(equations, [beta_closed.real * 1.00001, beta_closed.imag * 0.99999],
                  method="hybr", options={"xtol": 1e-12})
    beta_root = complex(*map(float, result.x))
    residual = abs(boundary(beta_root))
    relative_difference = abs(beta_root - beta_closed) / abs(beta_closed)
    if (not np.isfinite(result.x).all() or residual > 1e-8
            or relative_difference > 1e-8 or beta_root.imag <= 0):
        raise RuntimeError("independent planar TM boundary root failed verification")
    return {
        "wavelength_um": wavelength_um, "analyte_ri": analyte_ri,
        "au_epsilon_real": eps_m.real, "au_epsilon_imag": eps_m.imag,
        "beta_real_um_inv": beta_closed.real,
        "beta_imag_um_inv": beta_closed.imag,
        "root_relative_difference": relative_difference,
        "tm_boundary_residual": residual,
        "power_attenuation_db_cm": (20 / math.log(10)) * beta_closed.imag * 1e4,
        "propagation_length_power_um": 1 / (2 * beta_closed.imag),
        "qualified_pcf_physics": False, "publication_ready": False,
        "comsol_verified": False,
    }


def run_control(out: Path, wavelengths: tuple[float, ...] = (0.75, 0.8, 0.85),
                analyte_ris: tuple[float, ...] = (1.35, 1.36, 1.37)) -> dict:
    if not wavelengths or not analyte_ris:
        raise ValueError("nonempty grid required")
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    results = [planar_tm_control(w, n) for n in analyte_ris for w in wavelengths]
    report = {
        "evidence_class": "independent_analytic_planar_interface_control_only",
        "source": au_reference_manifest(),
        "method": "closed_form_TM_SPP_vs_independent_complex_boundary_root",
        "temporal_convention": "exp(-i omega t)",
        "points": results,
        "pcf_geometry_verified": False, "pml_validated": False,
        "mode_tracking_physically_verified": False,
        "sensor_sensitivity_verified": False, "publication_ready": False,
        "comsol_verified": False,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        handle.write(payload)
    return {"report_sha256": hashlib.sha256(payload).hexdigest(), **report}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = run_control(args.out)
    print(json.dumps({"report_sha256": report["report_sha256"],
                      "points": len(report["points"]),
                      "max_relative_error": max(x["root_relative_difference"] for x in report["points"]),
                      "publication_ready": False}, sort_keys=True))


if __name__ == "__main__":
    main()
