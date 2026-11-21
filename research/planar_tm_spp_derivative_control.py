"""Planar TM-SPP RI derivative consistency; never PCF sensor sensitivity.

At fixed wavelength and measured Au permittivity, compare the analytic
complex propagation-constant derivative with central differences of the
independently root-verified planar interface solution.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from research.planar_tm_spp_reference import planar_tm_control
from sprpcf.simulation.materials_reference import au_reference_manifest


def planar_derivative_control(wavelength_um: float, analyte_ri: float, step: float = 1e-4) -> dict:
    if not math.isfinite(step) or not 0 < step < 0.01:
        raise ValueError("finite positive RI step below 0.01 required")
    if not math.isfinite(analyte_ri) or not 1.0 + step < analyte_ri < 2.0 - step:
        raise ValueError("RI and perturbations must remain inside material domain")
    center = planar_tm_control(wavelength_um, analyte_ri)
    plus = planar_tm_control(wavelength_um, analyte_ri + step)
    minus = planar_tm_control(wavelength_um, analyte_ri - step)
    eps_m = complex(center["au_epsilon_real"], center["au_epsilon_imag"])
    beta = complex(center["beta_real_um_inv"], center["beta_imag_um_inv"])
    derivative = beta * (1 / analyte_ri - analyte_ri / (eps_m + analyte_ri**2))
    beta_plus = complex(plus["beta_real_um_inv"], plus["beta_imag_um_inv"])
    beta_minus = complex(minus["beta_real_um_inv"], minus["beta_imag_um_inv"])
    finite_difference = (beta_plus - beta_minus) / (2 * step)
    relative_error = abs(derivative - finite_difference) / max(abs(derivative), 1e-15)
    if not math.isfinite(relative_error) or relative_error > 1e-5:
        raise RuntimeError("planar TM-SPP derivative consistency failed")
    return {
        "wavelength_um": wavelength_um,
        "analyte_ri": analyte_ri,
        "ri_step": step,
        "analytic_dbeta_dn_real_um_inv_riu_inv": derivative.real,
        "analytic_dbeta_dn_imag_um_inv_riu_inv": derivative.imag,
        "finite_difference_dbeta_dn_real_um_inv_riu_inv": finite_difference.real,
        "finite_difference_dbeta_dn_imag_um_inv_riu_inv": finite_difference.imag,
        "relative_error": relative_error,
        "source_class": "measured_Au_planar_interface_only",
        "pcf_sensor_sensitivity": False,
        "qualified_pcf_physics": False,
        "publication_ready": False,
        "comsol_verified": False,
    }


def run(out: Path) -> dict:
    if out.exists():
        raise FileExistsError("immutable output exists")
    points = [planar_derivative_control(w, n) for n in (1.35, 1.36, 1.37)
              for w in (0.75, 0.80, 0.85)]
    report = {
        "source": au_reference_manifest(),
        "method": "analytic_derivative_vs_central_difference_of_root_checked_planar_beta",
        "temporal_convention": "exp(-i omega t)",
        "qualified_pcf_physics": False,
        "evidence_class": "planar_analytic_derivative_crosscheck_not_PCF",
        "points": points,
        "pcf_sensor_sensitivity": False,
        "publication_ready": False,
        "comsol_verified": False,
    }
    payload = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        f.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), **report}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.out)
    print(json.dumps({"sha256": result["sha256"],
                      "points": len(result["points"]),
                      "max_relative_error": max(p["relative_error"] for p in result["points"]),
                      "publication_ready": False}))
