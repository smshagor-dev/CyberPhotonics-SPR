"""Eigenvalue-to-power attenuation consistency audit; not PML validation.

E(z,t)=E0 exp(i beta z-i omega t), beta=k0*n_eff.
Power attenuation in dB/cm = (20/ln(10))*(2*pi/(lambda_um*1e-4))*Im(n_eff).
"""
from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class LossAudit:
    consistent: bool
    reasons: tuple[str, ...]
    computed_db_cm: float | None = None
    physics_qualified: bool = False
    publication_ready: bool = False
    comsol_verified: bool = False

def audit_neff_loss(wavelength_um: float, neff_real: float, neff_imag: float,
                    reported_db_cm: float, *, relative_tolerance: float = 1e-6,
                    absolute_tolerance_db_cm: float = 1e-9) -> LossAudit:
    """Fail closed on nonfinite, sign-invalid or unit-inconsistent loss records."""
    try:
        values = (wavelength_um, neff_real, neff_imag, reported_db_cm,
                  relative_tolerance, absolute_tolerance_db_cm)
        if any(isinstance(v, bool) for v in values):
            return LossAudit(False, ("invalid_numeric_type",))
        wl, nr, ni, observed, rtol, atol = map(float, values)
    except (TypeError, ValueError, OverflowError):
        return LossAudit(False, ("invalid_numeric_type",))
    if not all(map(math.isfinite, (wl, nr, ni, observed, rtol, atol))):
        return LossAudit(False, ("nonfinite_input",))
    if wl <= 0 or nr <= 0 or ni < 0 or observed < 0 or rtol < 0 or atol < 0:
        return LossAudit(False, ("invalid_physical_range_or_sign",))
    expected = (20.0 / math.log(10.0)) * (2.0 * math.pi / (wl * 1e-4)) * ni
    if not math.isfinite(expected):
        return LossAudit(False, ("overflow_loss",))
    if not math.isclose(expected, observed, rel_tol=rtol, abs_tol=atol):
        return LossAudit(False, ("loss_neff_mismatch",), expected)
    return LossAudit(True, (), expected)
