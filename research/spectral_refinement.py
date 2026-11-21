"""Fail-closed two-grid spectral stability audit; never qualifies Maxwell physics."""
from dataclasses import dataclass
import math
from .spectral_resolution_gate import assess_spectrum

@dataclass(frozen=True)
class SpectralRefinement:
    stable: bool
    reasons: tuple
    resonance_shift_um: float | None = None
    fwhm_relative_change: float | None = None
    physics_qualified: bool = False
    publication_ready: bool = False
    comsol_verified: bool = False

def compare_spectra(coarse, fine, *, resonance_tol_um=0.001,
                    fwhm_relative_tol=0.03, min_refinement=1.5):
    """Require identical geometry/RI, distinct source IDs and refined wavelength grid.

    This tests numerical spectral sampling consistency, NOT independent solvers,
    actual Maxwell mode tracking, PML convergence, or material provenance.
    """
    reasons = []
    for label, record in (("coarse", coarse), ("fine", fine)):
        if not isinstance(record, dict):
            return SpectralRefinement(False, (f"{label}_invalid_record",))
        for field in ("wavelength_um", "loss_db_cm", "geometry_sha256", "ri", "source_id"):
            if field not in record:
                reasons.append(f"{label}_missing_{field}")
    if reasons:
        return SpectralRefinement(False, tuple(reasons))
    for field in ("geometry_sha256", "ri"):
        if coarse[field] != fine[field]:
            reasons.append(f"mismatched_{field}")
    if coarse["source_id"] == fine["source_id"] or not all(
        isinstance(r["source_id"], str) and r["source_id"].strip()
        for r in (coarse, fine)
    ):
        reasons.append("nonindependent_sources")
    if not all(isinstance(r["geometry_sha256"], str) and
               len(r["geometry_sha256"]) == 64 and
               all(c in "0123456789abcdef" for c in r["geometry_sha256"])
               for r in (coarse, fine)):
        reasons.append("invalid_geometry_hash")
    try:
        ri = float(coarse["ri"])
        if not math.isfinite(ri) or ri <= 0:
            reasons.append("invalid_ri")
        c = assess_spectrum(coarse["wavelength_um"], coarse["loss_db_cm"])
        f = assess_spectrum(fine["wavelength_um"], fine["loss_db_cm"])
        if not c.resolved:
            reasons.append("coarse_unresolved:" + ",".join(c.reasons))
        if not f.resolved:
            reasons.append("fine_unresolved:" + ",".join(f.reasons))
        cx, fx = [tuple(map(float, r["wavelength_um"])) for r in (coarse, fine)]
        if cx[0] != fx[0] or cx[-1] != fx[-1]:
            reasons.append("different_spectral_domain")
        ch = max(b-a for a,b in zip(cx,cx[1:]))
        fh = max(b-a for a,b in zip(fx,fx[1:]))
        if fh <= 0 or ch / fh < min_refinement:
            reasons.append("insufficient_refinement")
        if not all(math.isfinite(v) and v > 0 for v in
                   (resonance_tol_um, fwhm_relative_tol, min_refinement)):
            reasons.append("invalid_tolerance")
        if reasons:
            return SpectralRefinement(False, tuple(reasons))
        shift = abs(c.resonance_um - f.resonance_um)
        change = abs(c.fwhm_um - f.fwhm_um) / f.fwhm_um
        if shift > resonance_tol_um:
            reasons.append("resonance_not_stable")
        if change > fwhm_relative_tol:
            reasons.append("fwhm_not_stable")
        return SpectralRefinement(not reasons, tuple(reasons), shift, change)
    except (ValueError, TypeError, ZeroDivisionError, OverflowError):
        return SpectralRefinement(False, ("invalid_numeric_input",))
