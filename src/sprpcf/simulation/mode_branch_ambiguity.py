"""Audit branch-assignment ambiguity from normalized native E/H overlaps.

Diagnostic only: never establishes core/SPP identity or PCF-SPR qualification.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


def audit_overlap_matrix(overlaps, *, minimum_overlap=0.80, minimum_margin=0.10):
    """Hungarian assignment with bidirectional competing-mode margins.

    Requires same-mesh, consistently normalized absolute overlaps in [0, 1].
    """
    if not (0 < minimum_overlap <= 1 and 0 < minimum_margin <= 1):
        raise ValueError("invalid diagnostic thresholds")
    a = np.asarray(overlaps)
    if a.ndim != 2 or a.shape[0] == 0 or a.shape[0] != a.shape[1]:
        raise ValueError("require nonempty square matrix")
    if a.dtype.kind not in "fi" or not np.isfinite(a).all() or np.any(a < 0) or np.any(a > 1):
        raise ValueError("require finite normalized absolute overlaps in [0,1]")
    row, col = linear_sum_assignment(-a)
    results = []
    for i, j in zip(row, col):
        competitors = [float(a[i, k]) for k in range(len(a)) if k != j]
        competitors += [float(a[k, j]) for k in range(len(a)) if k != i]
        runner_up = max(competitors, default=0.0)
        score = float(a[i, j])
        margin = score - runner_up
        results.append({
            "previous_mode_index": int(i), "current_mode_index": int(j),
            "normalized_overlap": score, "runner_up_overlap": runner_up,
            "bidirectional_margin": margin,
            "diagnostically_unambiguous": bool(score >= minimum_overlap and margin >= minimum_margin),
            "core_spp_identity_verified": False,
        })
    return {
        "assignments": results,
        "all_diagnostically_unambiguous": all(r["diagnostically_unambiguous"] for r in results),
        "physics_qualified": False, "publication_ready": False, "comsol_verified": False,
        "assumptions": "same spatial mesh, consistent field normalization, physical E/H overlap",
    }
