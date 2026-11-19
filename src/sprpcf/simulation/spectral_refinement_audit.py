"""Fail-closed spectral sampling convergence diagnostic (not physics qualification)."""
import math

def _number(x):
    if isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x):
        raise ValueError("nonfinite or nonnumeric optical datum")
    return float(x)

def _metrics(row):
    xs = [_number(x) for x in row["wavelength_nm"]]
    ys = [_number(y) for y in row["loss_db_cm"]]
    if len(xs) != len(ys) or len(xs) < 7 or xs[0] <= 0 or any(b <= a for a,b in zip(xs,xs[1:])):
        raise ValueError("invalid wavelength grid")
    if any(y < 0 for y in ys):
        raise ValueError("negative confinement loss")
    i = max(range(len(ys)), key=ys.__getitem__)
    if i < 2 or i > len(xs)-3 or ys.count(ys[i]) != 1:
        raise ValueError("ambiguous or boundary resonance")
    if any(ys[j] >= ys[j+1] for j in range(i)) or any(ys[j] <= ys[j+1] for j in range(i,len(ys)-1)):
        raise ValueError("not strictly unimodal")
    # Three-point nonuniform-grid parabola: report a sub-grid diagnostic
    # only when the vertex is concave and bracketed by neighboring samples.
    x0, x1, x2 = xs[i-1], xs[i], xs[i+1]
    y0, y1, y2 = ys[i-1], ys[i], ys[i+1]
    left_slope = (y1-y0)/(x1-x0)
    right_slope = (y2-y1)/(x2-x1)
    curvature = (right_slope-left_slope)/(x2-x0)
    if not math.isfinite(curvature) or curvature >= 0:
        raise ValueError("unresolved or nonconcave local peak")
    linear = left_slope-curvature*(x0+x1)
    vertex = -linear/(2*curvature)
    if not math.isfinite(vertex) or not (x0 <= vertex <= x2):
        raise ValueError("unbracketed quadratic peak vertex")
    # FWHM must use the same interpolated peak height as the reported
    # sub-grid peak position. A sampled-bin height biases half-prominence
    # crossings, especially on coarser or offset spectral grids.
    peak_height = y1 - curvature*(x1-vertex)**2
    baseline = max(ys[0],ys[-1])
    if not math.isfinite(peak_height) or peak_height <= baseline:
        raise ValueError("unresolved interpolated peak prominence")
    half = baseline + (peak_height-baseline)/2
    def crossing(indices):
        for j,k in indices:
            if (ys[j]-half)*(ys[k]-half) <= 0 and ys[j] != ys[k]:
                return xs[j]+(half-ys[j])*(xs[k]-xs[j])/(ys[k]-ys[j])
        raise ValueError("unbracketed half-prominence")
    left = crossing((j,j+1) for j in range(i-1,-1,-1))
    right = crossing((j,j+1) for j in range(i,len(xs)-1))
    return {"peak_nm":vertex,"sample_peak_nm":xs[i],"interpolated_peak_loss_db_cm":peak_height,
            "fwhm_nm":right-left,\n            "width_definition":"half-prominence above max endpoint baseline (not zero-baseline FWHM)",
            "max_step_nm":max(b-a for a,b in zip(xs,xs[1:]))}

def audit_spectral_refinement(records, *, peak_tol_nm, fwhm_relative_tol):
    """Require >=3 increasingly dense same-geometry, same-mode spectra.

    This checks sampling stability only; mesh, PML, domain and independent
    solver evidence remain separate mandatory gates.
    """
    peak_tol_nm = _number(peak_tol_nm)
    fwhm_relative_tol = _number(fwhm_relative_tol)
    if peak_tol_nm <= 0 or not 0 < fwhm_relative_tol < 1 or len(records) < 3:
        raise ValueError("invalid audit protocol")
    keys = ("geometry_sha256","material_sha256","mode_branch_id","ri","pml_sha256","solver_name")
    identity = None
    reference_window = None
    metrics = []
    for row in records:
        if any(k not in row or row[k] is None or row[k] == "" for k in keys):
            raise ValueError("missing source identity")
        _number(row["ri"])
        ident = tuple(row[k] for k in keys)
        if identity is not None and ident != identity:
            raise ValueError("fixed-configuration/mode mismatch")
        identity = ident
        # A narrowed or shifted spectral window can suppress half-height tails\n        # and manufacture apparent refinement stability. Require identical\n        # physical wavelength endpoints, independent of interior sampling.\n        wavelengths = row["wavelength_nm"]\n        window = (_number(wavelengths[0]), _number(wavelengths[-1]))\n        if reference_window is not None and any(\n            abs(x - y) > 1e-9 for x, y in zip(window, reference_window)\n        ):\n            raise ValueError("spectral refinement wavelength window changed")\n        reference_window = window\n        metrics.append(_metrics(row))
    if any(b["max_step_nm"] >= a["max_step_nm"] for a,b in zip(metrics,metrics[1:])):
        raise ValueError("grid not strictly refined")
    a,b = metrics[-2:]
    peak_drift = abs(a["peak_nm"]-b["peak_nm"])
    width_drift = abs(a["fwhm_nm"]-b["fwhm_nm"])/b["fwhm_nm"]
    return {"diagnostic_pass":peak_drift <= peak_tol_nm and width_drift <= fwhm_relative_tol,
            "physics_qualified":False,"publication_ready":False,"comsol_verified":False,
            "peak_drift_nm":peak_drift,"fwhm_relative_drift":width_drift,
            "spectra":metrics,
            "limitations":"Sampling diagnostic only; no mesh/PML/domain/solver convergence."}
