"""Fail-closed Maxwell attenuation budget, exp(-i omega t).

All alpha values are POWER attenuation in Np/m. Unmeasured leakage cannot be
silently set to zero. Closure is a numerical diagnostic, not publication evidence.
"""
import math


def audit_loss_budget(neff_imag, wavelength_m, absorption_alpha_np_per_m,
                      *, leakage_alpha_np_per_m=None, relative_tolerance=0.05):
    values = (neff_imag, wavelength_m, absorption_alpha_np_per_m, relative_tolerance)
    if any(isinstance(v, bool) or not isinstance(v, (float, int))
           or not math.isfinite(v) for v in values):
        raise ValueError("all inputs must be finite real scalars")
    if (wavelength_m <= 0 or neff_imag < 0 or absorption_alpha_np_per_m < 0
            or not 0 < relative_tolerance < 1):
        raise ValueError("invalid passive attenuation, wavelength or tolerance")
    eigen_alpha = 4 * math.pi * neff_imag / wavelength_m
    residual = eigen_alpha - absorption_alpha_np_per_m
    db_cm = (10 / math.log(10)) / 100
    result = {
        "eigenmode_alpha_np_per_m": eigen_alpha,
        "absorption_alpha_np_per_m": absorption_alpha_np_per_m,
        "unexplained_alpha_np_per_m": residual,
        "eigenmode_loss_db_per_cm": eigen_alpha * db_cm,
        "absorption_loss_db_per_cm": absorption_alpha_np_per_m * db_cm,
        "budget_verified": False, "physics_qualified": False,
        "publication_ready": False,
    }
    if leakage_alpha_np_per_m is None:
        result["reason"] = "leakage_or_PML_boundary_power_not_independently_measured"
        return result
    if (isinstance(leakage_alpha_np_per_m, bool)
            or not isinstance(leakage_alpha_np_per_m, (float, int))
            or not math.isfinite(leakage_alpha_np_per_m)
            or leakage_alpha_np_per_m < 0):
        raise ValueError("leakage must be finite nonnegative power attenuation")
    error = eigen_alpha - absorption_alpha_np_per_m - leakage_alpha_np_per_m
    result["leakage_alpha_np_per_m"] = leakage_alpha_np_per_m
    result["closure_error_alpha_np_per_m"] = error
    result["relative_closure_error"] = abs(error) / max(
        eigen_alpha, absorption_alpha_np_per_m + leakage_alpha_np_per_m, 1e-30)
    result["budget_verified"] = result["relative_closure_error"] <= relative_tolerance
    result["reason"] = ("numerical_budget_closure_only" if result["budget_verified"]
                        else "loss_budget_does_not_close")
    return result
