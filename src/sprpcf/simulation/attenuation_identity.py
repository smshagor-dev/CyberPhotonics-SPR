"""Dimensional audit of passive eigenmode attenuation, not sensor validation."""
from __future__ import annotations
import math


def loss_db_per_cm(wavelength_um: float, n_eff_imag: float) -> float:
    """Intensity attenuation in dB/cm for exp(+i beta z - i omega t).

    beta = 2*pi*n_eff/lambda; positive Im(n_eff) is passive decay.
    """
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool)
               and math.isfinite(v) for v in (wavelength_um, n_eff_imag)):
        raise ValueError("wavelength and imaginary effective index must be finite real numbers")
    if wavelength_um <= 0 or n_eff_imag < 0:
        raise ValueError("positive wavelength and passive nonnegative Im(n_eff) required")
    return (20.0 / math.log(10.0)) * (2.0 * math.pi * 1e4 / wavelength_um) * n_eff_imag


def audit_attenuation_rows(rows, *, relative_tolerance: float = 1e-6,
                           absolute_tolerance: float = 1e-8) -> dict:
    """Audit CSV-like row dictionaries, accounting for each invalid row explicitly."""
    if not math.isfinite(relative_tolerance) or relative_tolerance < 0:
        raise ValueError("invalid relative tolerance")
    if not math.isfinite(absolute_tolerance) or absolute_tolerance < 0:
        raise ValueError("invalid absolute tolerance")
    failures = []
    count = 0
    for count, row in enumerate(rows, 1):
        try:
            expected = loss_db_per_cm(float(row["wavelength_um"]), float(row["n_eff_imag"]))
            observed = float(row["raw_attenuation_db_per_cm_unqualified"])
            if not math.isfinite(observed) or observed < 0:
                raise ValueError("nonfinite or negative reported loss")
            if not math.isclose(observed, expected, rel_tol=relative_tolerance,
                                abs_tol=absolute_tolerance):
                raise ValueError(f"loss identity mismatch: reported={observed:.12g}, expected={expected:.12g}")
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            failures.append({"row": count, "reason": str(exc)})
    if count == 0:
        failures.append({"row": 0, "reason": "empty eigenmode rows"})
    return {"rows_checked": count, "failures": failures,
            "dimensional_identity_pass": not failures,
            "physics_qualified": False, "publication_ready": False,
            "comsol_verified": False}
