"""Fail-closed same-quadrature E/H overlap diagnostic; NOT PCF mode identity proof."""
import numpy as np
from scipy.optimize import linear_sum_assignment


def _normalized(fields, weights):
    f = np.asarray(fields, dtype=np.complex128)
    if f.ndim != 3 or f.shape[0] < 2 or f.shape[2] != 3 or f.shape[1] != len(weights):
        raise ValueError("expected (modes, quadrature points, 3) vectors")
    if not np.isfinite(f).all():
        raise ValueError("nonfinite fields")
    n = np.einsum("mqi,mqi,q->m", f.conj(), f, weights).real
    if np.any(n <= 1e-24):
        raise ValueError("zero field")
    return f / np.sqrt(n)[:, None, None]


def audit_overlap(e0, h0, e1, h1, weights, *, mesh0, mesh1,
                  geometry0, geometry1, material0, material1,
                  wavelength0=None, wavelength1=None, ri0=None, ri1=None,
                  min_overlap=0.9, min_margin=0.05):
    """Only same mesh/geometry; material variation needs explicit physical step.

    Does not prove core/SPP identity, Maxwell convergence, or sensor performance.
    """
    result = dict(overlap_tracking_passed=False, mode_identity_verified=False,
                  physics_qualified=False, publication_ready=False,
                  comsol_verified=False, errors=[], assignments=[])
    errors = result["errors"]
    for name, a, b in (("mesh", mesh0, mesh1), ("geometry", geometry0, geometry1)):
        if not a or a != b:
            errors.append(name + " mismatch")
    if not material0 or not material1:
        errors.append("missing material provenance")
    changed = False
    for name, a, b, lower in (("wavelength", wavelength0, wavelength1, 0.0),
                              ("analyte RI", ri0, ri1, 1.0)):
        if (a is None) != (b is None):
            errors.append("incomplete " + name + " provenance")
        elif a is not None:
            try:
                x, y = float(a), float(b)
                if not (np.isfinite(x) and np.isfinite(y) and x > lower and y > lower):
                    raise ValueError()
                changed |= x != y
            except (ValueError, TypeError):
                errors.append("invalid " + name + " provenance")
    if material0 != material1 and not changed:
        errors.append("unexplained material change")
    if not (np.isfinite(min_overlap) and 0 < min_overlap <= 1
            and np.isfinite(min_margin) and 0 < min_margin < 1):
        errors.append("invalid thresholds")
    try:
        w = np.asarray(weights, dtype=float)
        if w.ndim != 1 or not np.isfinite(w).all() or np.any(w <= 0):
            raise ValueError("invalid quadrature")
        e0, h0, e1, h1 = (_normalized(f, w) for f in (e0, h0, e1, h1))
        if not (e0.shape == h0.shape == e1.shape == h1.shape):
            raise ValueError("field dimensions differ")
        oe = abs(np.einsum("iqk,jqk,q->ij", e0.conj(), e1, w)) ** 2
        oh = abs(np.einsum("iqk,jqk,q->ij", h0.conj(), h1, w)) ** 2
        scores = np.minimum(oe, oh)
        rows, cols = linear_sum_assignment(-scores)
        for i, j in zip(rows, cols):
            other = np.r_[np.delete(scores[i], j), np.delete(scores[:, j], i)]
            margin = float(scores[i, j] - other.max())
            result["assignments"].append(dict(before=int(i), after=int(j),
                                              e_overlap=float(oe[i, j]),
                                              h_overlap=float(oh[i, j]),
                                              margin=margin))
            if scores[i, j] < min_overlap or margin < min_margin:
                errors.append("weak or ambiguous overlap")
    except (ValueError, TypeError) as exc:
        errors.append(str(exc))
    result["overlap_tracking_passed"] = not errors
    return result
