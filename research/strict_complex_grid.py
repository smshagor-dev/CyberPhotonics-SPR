"""Fail-closed complex eigenmode mesh-drift diagnostic; NOT physics qualification."""
import math
import re

HEX = re.compile(r"^[0-9a-f]{64}$")

def audit(rows):
    result = {"diagnostic_passed": False, "physics_qualified": False,
              "publication_ready": False, "errors": []}
    errors = result["errors"]
    if not isinstance(rows, list) or len(rows) != 3 or not all(isinstance(r, dict) for r in rows):
        errors.append("exactly three mesh records required")
        return result
    required = ("h_um", "neff_real", "neff_imag", "geometry_sha256",
                "material_sha256", "pml_sha256", "archive_sha256", "mode_tracking_verified")
    if any(any(k not in r for k in required) for r in rows):
        errors.append("missing required field")
        return result
    for field in ("geometry_sha256", "material_sha256", "pml_sha256", "archive_sha256"):
        values = [r[field] for r in rows]
        if any(not isinstance(v, str) or not HEX.fullmatch(v) for v in values):
            errors.append("invalid " + field)
        elif len(set(values)) != (3 if field == "archive_sha256" else 1):
            errors.append("inconsistent " + field)
    if any(r["mode_tracking_verified"] is not True for r in rows):
        errors.append("unverified mode correspondence")
    if any(any(type(r[k]) not in (float, int) or not math.isfinite(r[k])
               for k in ("h_um", "neff_real", "neff_imag")) for r in rows):
        errors.append("invalid mesh or eigenvalue")
        return result
    h = [r["h_um"] for r in rows]
    if not h[0] > h[1] > h[2] > 0:
        errors.append("nondecreasing mesh widths")
    if any(r["neff_imag"] < 0 for r in rows):
        errors.append("negative imaginary neff in passive convention")
    if errors:
        return result
    z = [complex(r["neff_real"], r["neff_imag"]) for r in rows]
    d1, d2 = z[0] - z[1], z[1] - z[2]
    r1, r2 = h[0] / h[1], h[1] / h[2]
    if abs(r1 - r2) / max(r1, r2) > .01:
        errors.append("nongeometric refinement")
    if abs(d2) == 0 or abs(d1) <= abs(d2):
        errors.append("nondecreasing or zero complex drift")
    if errors:
        return result
    ratio = d1 / d2
    phase = abs(math.degrees(math.atan2(ratio.imag, ratio.real)))
    result.update(coarse_medium_drift=abs(d1), medium_fine_drift=abs(d2),
                  relative_fine_drift=abs(d2)/max(abs(z[2]), 1e-30),
                  complex_drift_phase_deg=phase)
    if phase > 10:
        errors.append("complex drift direction inconsistent")
        return result
    order = math.log(abs(ratio)) / math.log(r2)
    if not math.isfinite(order) or order <= 0:
        errors.append("invalid observed order")
        return result
    result["observed_order_diagnostic"] = order
    result["diagnostic_passed"] = True
    return result
