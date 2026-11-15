"""Diagnostic same-quadrature vector-field correspondence; NOT publication evidence.

Positive-definite E/H overlap is not lossy-mode Poynting orthogonality.
Requires externally verified physical quadrature, matching mesh, field ordering,
units, magnetic scaling, and independently identified core/SPP character.
"""
from itertools import permutations
import numpy as np


class AmbiguousCorrespondence(ValueError):
    pass


def normalized_field_overlap(e_ref, h_ref, e_candidate, h_candidate, weights, *, magnetic_scale=1.0):
    """Phase-invariant overlap of (N,3) complex E/H vectors at identical nodes."""
    fields = [np.asarray(a, dtype=np.complex128) for a in
              (e_ref, h_ref, e_candidate, h_candidate)]
    w = np.asarray(weights, dtype=np.float64)
    if any(a.ndim != 2 or a.shape[1] != 3 for a in fields) or len({a.shape for a in fields}) != 1:
        raise ValueError("E/H vectors must share (nodes,3) quadrature")
    if w.shape != (fields[0].shape[0],) or not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("positive finite quadrature weights required")
    if any(not np.all(np.isfinite(a)) for a in fields):
        raise ValueError("nonfinite E/H field")
    if not np.isfinite(magnetic_scale) or magnetic_scale <= 0:
        raise ValueError("positive finite magnetic scaling required")
    er, hr, ec, hc = fields
    scale = float(magnetic_scale)
    dot = np.sum(w[:, None] * (np.conj(er)*ec + scale**2*np.conj(hr)*hc))
    nr = np.sum(w[:, None] * (abs(er)**2 + scale**2*abs(hr)**2)).real
    nc = np.sum(w[:, None] * (abs(ec)**2 + scale**2*abs(hc)**2)).real
    if not np.isfinite(nr) or not np.isfinite(nc) or nr <= 0 or nc <= 0:
        raise ValueError("zero or invalid field norm")
    score = float(abs(dot)/np.sqrt(nr*nc))
    if not np.isfinite(score) or score > 1 + 1e-10:
        raise ValueError("invalid normalized overlap")
    return min(1.0, score)


def correspondence_matrix(reference_modes, candidate_modes, weights, *, magnetic_scale=1.0):
    """Each mode is (E,H); eigenvalue ordering need not match."""
    if not reference_modes or len(reference_modes) != len(candidate_modes) or len(reference_modes) > 8:
        raise ValueError("equal nonempty batches of at most eight modes required")
    return np.array([[normalized_field_overlap(*r, *c, weights, magnetic_scale=magnetic_scale)
                      for c in candidate_modes] for r in reference_modes])


def assign_correspondence(matrix, *, minimum_overlap=0.75, minimum_margin=0.05):
    """Global assignment with weak-match and near-degeneracy rejection."""
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not 1 <= m.shape[0] <= 8:
        raise ValueError("square 1..8 overlap matrix required")
    if not np.all(np.isfinite(m)) or np.any(m < 0) or np.any(m > 1):
        raise ValueError("invalid normalized overlaps")
    if not (0 < minimum_overlap <= 1 and 0 < minimum_margin <= 1):
        raise ValueError("invalid thresholds")
    ranked = sorted(((sum(m[i,p[i]] for i in range(len(m))), p)
                     for p in permutations(range(len(m)))), reverse=True)
    score, assignment = ranked[0]
    if any(m[i,assignment[i]] < minimum_overlap for i in range(len(m))):
        raise AmbiguousCorrespondence("weak matched field overlap")
    if len(ranked) > 1 and score-ranked[1][0] < minimum_margin:
        raise AmbiguousCorrespondence("near-degenerate global correspondence")
    return assignment
