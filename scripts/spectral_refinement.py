"""Paired nested-grid diagnostic. NEVER qualifies Maxwell physics."""
import math
import re
from scripts.spectral_identifiability import inspect_spectrum

REQUIRED = ("geometry_sha256", "au_dataset_sha256", "analyte_ri",
            "mode_branch", "boundary_condition", "solver")

def _valid_provenance(p):
    if not isinstance(p, dict):
        return False
    for key in REQUIRED:
        if key not in p or p[key] is None:
            return False
    for key in ("geometry_sha256", "au_dataset_sha256"):
        if not isinstance(p[key], str) or re.fullmatch(r"[0-9a-fA-F]{64}", p[key]) is None:
            return False
    if not isinstance(p["analyte_ri"], (int, float)) or isinstance(p["analyte_ri"], bool):
        return False
    if not math.isfinite(p["analyte_ri"]) or p["analyte_ri"] <= 0:
        return False
    return all(isinstance(p[k], str) and bool(p[k].strip())
               for k in ("mode_branch", "boundary_condition", "solver"))


def _valid_nested_samples(coarse_x, coarse_y, fine_x, fine_y):
    # Numeric matching is exact by design: do not silently interpolate or
    # accept a nominally nested mesh with altered coarse-point attenuation.
    try:
        cx, cy, fx, fy = map(list, (coarse_x, coarse_y, fine_x, fine_y))
    except TypeError:
        return False, "invalid_samples"
    if len(cx) != len(cy) or len(fx) != len(fy):
        return False, "invalid_samples"
    if any(isinstance(v, bool) or not isinstance(v, (int, float))
           or not math.isfinite(v) for v in cx + cy + fx + fy):
        return False, "invalid_samples"
    if len(fx) <= len(cx) or len(set(cx)) != len(cx) or len(set(fx)) != len(fx):
        return False, "not_nested_refinement"
    fine = dict(zip(fx, fy))
    if not all(x in fine for x in cx):
        return False, "not_nested_refinement"
    if any(fine[x] != y for x, y in zip(cx, cy)):
        return False, "coarse_point_loss_mismatch"
    return True, None


def compare_spectra(coarse_wavelength_nm, coarse_loss_db_cm,
                    fine_wavelength_nm, fine_loss_db_cm, *,
                    coarse_provenance, fine_provenance,
                    peak_tolerance_nm=0.1, width_relative_tolerance=0.02):
    out = dict(stable=False, physics_qualified=False, publication_ready=False,
               resonance_verified=False, fwhm_verified=False, reason=None)
    if not _valid_provenance(coarse_provenance) or not _valid_provenance(fine_provenance):
        out["reason"] = "missing_provenance"
        return out
    if any(k not in coarse_provenance or k not in fine_provenance or
           coarse_provenance[k] != fine_provenance[k] or
           coarse_provenance[k] is None for k in REQUIRED):
        out["reason"] = "incompatible_or_missing_provenance"
        return out
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and
               math.isfinite(v) and v > 0 for v in
               (peak_tolerance_nm, width_relative_tolerance)):
        out["reason"] = "invalid_tolerance"
        return out
    # Materialize once: generators are otherwise exhausted by inspect_spectrum,
    # causing a false nested-grid failure or TypeError at len(...).
    try:
        coarse_wavelength_nm = list(coarse_wavelength_nm)
        coarse_loss_db_cm = list(coarse_loss_db_cm)
        fine_wavelength_nm = list(fine_wavelength_nm)
        fine_loss_db_cm = list(fine_loss_db_cm)
    except (TypeError, ValueError, OverflowError):
        out["reason"] = "invalid_samples"
        return out
    c = inspect_spectrum(coarse_wavelength_nm, coarse_loss_db_cm)
    f = inspect_spectrum(fine_wavelength_nm, fine_loss_db_cm)
    if not c["identified"] or not f["identified"]:
        out.update(reason="unidentified_spectrum", coarse_reason=c["reason"],
                   fine_reason=f["reason"])
        return out
    nested, nested_reason = _valid_nested_samples(coarse_wavelength_nm, coarse_loss_db_cm, fine_wavelength_nm, fine_loss_db_cm)
    if not nested:
        out["reason"] = nested_reason
        return out
    peak_drift = abs(c["peak_nm"] - f["peak_nm"])
    relative_width_drift = abs(c["width_nm"] - f["width_nm"]) / f["width_nm"]
    stable = peak_drift <= peak_tolerance_nm and relative_width_drift <= width_relative_tolerance
    out.update(stable=stable,
               reason="sampling_stable_diagnostic_only" if stable else "sampling_drift_exceeds_tolerance",
               peak_drift_nm=peak_drift, relative_width_drift=relative_width_drift,
               coarse_sample_count=len(coarse_wavelength_nm),
               fine_sample_count=len(fine_wavelength_nm))
    return out
