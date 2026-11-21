"""Three-resolution Richardson convergence audit of planar TM-SPP RI derivatives.

This is a measured-reference planar interface arithmetic benchmark, NOT PCF physics.
No fixture is ever eligible for scientific evidence promotion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from research.planar_tm_spp_derivative_control import planar_derivative_control
from sprpcf.simulation.materials_reference import au_reference_manifest

STEPS = (1e-3, 5e-4, 2.5e-4)


def audit_point(wavelength_um: float, analyte_ri: float) -> dict:
    records = [planar_derivative_control(wavelength_um, analyte_ri, h) for h in STEPS]
    # A finite relative-error scalar alone does not establish a valid derivative.
    # Reject nonfinite/zero analytic roots and inconsistent analytic references
    # before Richardson extrapolation, rather than emitting a misleading report.
    for record in records:
        for key in (
            "finite_difference_dbeta_dn_real_um_inv_riu_inv",
            "finite_difference_dbeta_dn_imag_um_inv_riu_inv",
            "analytic_dbeta_dn_real_um_inv_riu_inv",
            "analytic_dbeta_dn_imag_um_inv_riu_inv",
        ):
            value = float(record[key])
            if not math.isfinite(value):
                raise RuntimeError("nonfinite derivative component")
    analytic_values = [
        complex(r["analytic_dbeta_dn_real_um_inv_riu_inv"],
                r["analytic_dbeta_dn_imag_um_inv_riu_inv"])
        for r in records
    ]
    if abs(analytic_values[0]) == 0:
        raise RuntimeError("zero analytic derivative reference")
    if any(abs(v - analytic_values[0]) > 1e-12 * abs(analytic_values[0])
           for v in analytic_values[1:]):
        raise RuntimeError("analytic derivative reference varies with step")
    errors = [float(r["relative_error"]) for r in records]
    if not all(math.isfinite(e) and e > 0 for e in errors):
        raise RuntimeError("nonfinite or zero convergence error")
    ratios = [errors[i] / errors[i + 1] for i in range(2)]
    # A central finite difference has second-order truncation error. Tolerance
    # accommodates roundoff in complex root evaluation at the smallest step.
    if not all(2.5 < ratio < 5.5 for ratio in ratios):
        raise RuntimeError("expected second-order finite-difference convergence absent")
    deriv = [complex(r["finite_difference_dbeta_dn_real_um_inv_riu_inv"],
                     r["finite_difference_dbeta_dn_imag_um_inv_riu_inv"]) for r in records]
    analytic = complex(records[0]["analytic_dbeta_dn_real_um_inv_riu_inv"],
                       records[0]["analytic_dbeta_dn_imag_um_inv_riu_inv"])
    richardson = (4 * deriv[2] - deriv[1]) / 3
    extrapolated_error = abs(richardson - analytic) / abs(analytic)
    if not math.isfinite(extrapolated_error) or extrapolated_error > 1e-8:
        raise RuntimeError("Richardson extrapolation fails analytic cross-check")
    return {
        "wavelength_um": wavelength_um, "analyte_ri": analyte_ri,
        "steps_riu": list(STEPS), "central_difference_relative_errors": errors,
        "successive_error_ratios": ratios,
        "richardson_relative_error": extrapolated_error,
        "qualified_pcf_physics": False, "publication_ready": False,
        "comsol_verified": False,
    }


def run(out: Path) -> dict:
    if out.exists():
        raise FileExistsError("immutable native report already exists")
    points = [audit_point(w, n) for n in (1.35, 1.36, 1.37)
              for w in (0.75, 0.8, 0.85)]
    report = {
        "evidence_class": "measured_reference_planar_derivative_convergence_only",
        "source": au_reference_manifest(),
        "method": "three_step_central_difference_Richardson_order_two",
        "temporal_convention": "exp(-i omega t)",
        "points": points,
        "qualified_pcf_physics": False, "publication_ready": False,
        "comsol_verified": False, "pml_validated": False,
        "pcf_sensor_sensitivity_verified": False,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as f:
        f.write(payload)
    return {"sha256": hashlib.sha256(payload).hexdigest(), **report}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    r = run(args.out)
    print(json.dumps({"sha256": r["sha256"], "points": len(r["points"]),
                      "max_richardson_relative_error": max(x["richardson_relative_error"] for x in r["points"]),
                      "publication_ready": False}))
