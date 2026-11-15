"""Conservative, globally optimal mode correspondence from independently computed overlaps.

Inputs are *normalized magnitudes* of native vector E/H overlap integrals,
not fabricated identity evidence. This utility never qualifies a physics result.
"""
from __future__ import annotations

from itertools import permutations
from math import isfinite
from typing import Sequence


class AmbiguousModeAssignment(ValueError):
    """Insufficient overlap evidence to establish a unique correspondence."""


def assign_modes(
    overlap: Sequence[Sequence[float]],
    *,
    minimum_overlap: float = 0.75,
    minimum_margin: float = 0.05,
    max_modes: int = 8,
) -> tuple[int, ...]:
    """Return reference-index -> candidate-index correspondence.

    Exhaustive global assignment is intentional: small eigenmode batches only.
    Reject both low absolute overlap and near-tied global assignments; do not
    silently infer mode identity from proximity of complex eigenvalues.
    """
    n = len(overlap)
    if not 1 <= n <= max_modes or any(len(row) != n for row in overlap):
        raise ValueError("square nonempty overlap matrix required (bounded mode count)")
    if not (isfinite(minimum_overlap) and 0 < minimum_overlap <= 1):
        raise ValueError("invalid minimum_overlap")
    if not (isfinite(minimum_margin) and 0 < minimum_margin <= 1):
        raise ValueError("invalid minimum_margin")
    matrix = tuple(tuple(float(v) for v in row) for row in overlap)
    if any(not isfinite(v) or v < 0 or v > 1 for row in matrix for v in row):
        raise ValueError("overlap magnitudes must be finite in [0,1]")
    ranked = sorted(
        ((sum(matrix[i][p[i]] for i in range(n)), p) for p in permutations(range(n))),
        reverse=True,
    )
    score, assignment = ranked[0]
    if any(matrix[i][assignment[i]] < minimum_overlap for i in range(n)):
        raise AmbiguousModeAssignment("matched mode overlap below threshold")
    if len(ranked) > 1 and score - ranked[1][0] < minimum_margin:
        raise AmbiguousModeAssignment("globally ambiguous mode correspondence")
    return assignment


def assign_modes_from_fields(*args, **kwargs):
    raise NotImplementedError(
        "Native E/H quadrature, consistent meshes, phase conventions and "
        "physical core/SPP identity verification are required upstream."
    )
