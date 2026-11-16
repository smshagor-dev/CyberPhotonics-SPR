"""Fail-closed spectral sampling audit; diagnostics never qualify Maxwell physics."""
import math

def audit_spectrum(wavelength_um, loss_db_cm, *, source_kind, geometry_sha256,
                   pml_verified=False, branch_verified=False, spectral_convergence_verified=False):
    result = dict(physics_qualified=False, publication_ready=False,
                  resonance_verified=False, fwhm_verified=False, errors=[])
    x, y = list(wavelength_um), list(loss_db_cm)
    if source_kind != "solver_native":
        result["errors"].append("source is not solver-native")
    if not isinstance(geometry_sha256, str) or len(geometry_sha256) != 64 or any(c not in "0123456789abcdef" for c in geometry_sha256):
        result["errors"].append("invalid geometry SHA256")
    if len(x) != len(y) or len(x) < 5:
        result["errors"].append("at least five matched wavelength/loss samples required")
        return result
    if any(not math.isfinite(v) for v in x+y) or any(b <= a for a,b in zip(x,x[1:])) or any(v < 0 for v in y):
        result["errors"].append("nonfinite, unordered or nonpassive spectrum")
        return result
    if result["errors"]:
        return result
    peaks = [i for i in range(1,len(y)-1) if y[i]>y[i-1] and y[i]>y[i+1]]
    if len(peaks) != 1:
        result["errors"].append("unique interior loss peak not established")
        return result
    p = peaks[0]
    baseline = min(y[0], y[-1])
    height = y[p]-baseline
    if height <= 0:
        result["errors"].append("nonpositive peak prominence")
        return result
    half = baseline+height/2
    def crossing(indices):
        for a,b in indices:
            if (y[a]-half)*(y[b]-half)<=0 and y[a]!=y[b]:
                return x[a]+(half-y[a])*(x[b]-x[a])/(y[b]-y[a]), abs(x[b]-x[a])
        return None
    left = crossing((i,i+1) for i in range(p-1,-1,-1))
    right = crossing((i,i+1) for i in range(p,len(y)-1))
    if left is None or right is None:
        result["errors"].append("half-maximum not bracketed on both sides")
        return result
    width = right[0]-left[0]
    if width <= 0:
        result["errors"].append("nonpositive interpolated width")
        return result
    step = max(left[1],right[1])
    result.update(candidate_peak_um=x[p], candidate_fwhm_um=width,
                  max_bracket_step_um=step, fwhm_to_bracket_step_ratio=width/step)
    if width/step < 5:
        result["errors"].append("FWHM insufficiently sampled relative to crossing intervals")
    if not pml_verified: result["errors"].append("PML unverified")
    if not branch_verified: result["errors"].append("mode identity unverified")
    if not spectral_convergence_verified: result["errors"].append("independent spectral refinement unverified")
    if not result["errors"]:
        result["resonance_verified"]=True
        result["fwhm_verified"]=True
    return result
