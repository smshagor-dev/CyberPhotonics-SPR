"""Paired nested-grid diagnostic. NEVER qualifies Maxwell physics."""
import math
from scripts.spectral_identifiability import inspect_spectrum

REQUIRED = ("geometry_sha256", "au_dataset_sha256", "analyte_ri",
            "mode_branch", "boundary_condition", "solver")

def compare_spectra(coarse_wavelength_nm, coarse_loss_db_cm,
                    fine_wavelength_nm, fine_loss_db_cm, *,
                    coarse_provenance, fine_provenance,
                    peak_tolerance_nm=0.1, width_relative_tolerance=0.02):
    out = dict(stable=False, physics_qualified=False, publication_ready=False,
               resonance_verified=False, fwhm_verified=False, reason=None)
    if not isinstance(coarse_provenance, dict) or not isinstance(fine_provenance, dict):
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
    c = inspect_spectrum(coarse_wavelength_nm, coarse_loss_db_cm)
    f = inspect_spectrum(fine_wavelength_nm, fine_loss_db_cm)
    if not c["identified"] or not f["identified"]:
        out.update(reason="unidentified_spectrum", coarse_reason=c["reason"],
                   fine_reason=f["reason"])
        return out
    if len(fine_wavelength_nm) <= len(coarse_wavelength_nm) or not set(coarse_wavelength_nm).issubset(set(fine_wavelength_nm)):
        out["reason"] = "not_nested_refinement"
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
