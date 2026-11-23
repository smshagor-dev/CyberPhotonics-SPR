"""Fail-closed same-DOF Maxwell eigenfield mode-cycle diagnostic.

This is software validation, not proof of physical core/SPP identity.
"""
import itertools
import numpy as np


def overlap_matrix(reference, candidate, weights):
    a, b = np.asarray(reference, complex), np.asarray(candidate, complex)
    w = np.asarray(weights, float)
    if a.ndim != 2 or b.shape != a.shape or w.shape != (a.shape[1],):
        raise ValueError("same mesh and DOF ordering required")
    if not (np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(w).all()) or np.any(w <= 0):
        raise ValueError("invalid fields or quadrature weights")
    w = w / w.max()

    def normalize(x):
        scale = np.max(np.abs(x), axis=1)
        if np.any(scale == 0):
            raise ValueError("zero eigenfield")
        y = x / scale[:, None]
        norm = np.sqrt(np.sum(w[None, :] * np.abs(y)**2, axis=1))
        if np.any(norm == 0) or not np.isfinite(norm).all():
            raise ValueError("invalid field norm")
        return y / norm[:, None]

    a, b = normalize(a), normalize(b)
    result = np.abs((a.conj() * w[None, :]) @ b.T)**2
    if not np.isfinite(result).all():
        raise ValueError("nonfinite overlap")
    return np.clip(result, 0, 1)


def assign(overlap, minimum=0.6, gap=0.05):
    m = np.asarray(overlap, float)
    if m.ndim != 2 or m.shape[0] != m.shape[1] or not 1 <= len(m) <= 8:
        raise ValueError("1..8 square mode matrix required")
    if not np.isfinite(m).all() or np.any((m < 0) | (m > 1)):
        raise ValueError("invalid overlap")
    if not np.isfinite(minimum) or not np.isfinite(gap) or not 0 <= minimum <= 1 or gap < 0:
        raise ValueError("invalid thresholds")
    scores = sorted(((sum(m[i, p[i]] for i in range(len(p))), p)
                     for p in itertools.permutations(range(len(m)))), reverse=True)
    best, perm = scores[0]
    if any(m[i, perm[i]] < minimum for i in range(len(perm))):
        raise ValueError("low-confidence correspondence")
    if len(scores) > 1 and best - scores[1][0] <= gap:
        raise ValueError("globally ambiguous correspondence")
    return tuple(perm)


def cycle_audit(fields, weights, minimum=0.6, gap=0.05):
    """A->B->C must equal A->C; three points must share identical mesh DOFs."""
    if len(fields) != 3:
        raise ValueError("three spectral points required")
    ab = assign(overlap_matrix(fields[0], fields[1], weights), minimum, gap)
    bc = assign(overlap_matrix(fields[1], fields[2], weights), minimum, gap)
    ac = assign(overlap_matrix(fields[0], fields[2], weights), minimum, gap)
    consistent = tuple(bc[j] for j in ab) == ac
    return {"ab": ab, "bc": bc, "ac": ac, "cycle_consistent": consistent,
            "accepted": consistent, "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False}
