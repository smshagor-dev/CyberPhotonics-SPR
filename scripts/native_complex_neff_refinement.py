"""Fail-closed numerical convergence diagnostics for complex Maxwell eigenvalues.

Inputs must come from independently archived native solver runs. This module
never certifies mode identity, material validity, or publication readiness.
"""
from __future__ import annotations

import math


class ConvergenceError(ValueError):
    pass


def _complex_mode(row):
    if not isinstance(row, dict):
        raise ConvergenceError("invalid run record")
    for key in ("neff_real", "neff_imag", "resolution_um", "artifact_sha256"):
        if key not in row:
            raise ConvergenceError(f"missing {key}")
    values = [row[k] for k in ("neff_real", "neff_imag", "resolution_um")]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ConvergenceError("nonfinite mode or resolution")
    if row["resolution_um"] <= 0 or row["neff_imag"] < 0:
        raise ConvergenceError("invalid passive mode or mesh size")
    digest = row["artifact_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ConvergenceError("invalid artifact digest")
    return complex(row["neff_real"], row["neff_imag"])


def audit_refinement(runs, *, wavelength_um, max_relative_drift=0.01):
    """Compare three *same-mode* refinements, without assuming mode identity.

    A small neff drift alone cannot establish convergence: branch swaps,
    boundary reflections, PML changes and loss accuracy remain independent.
    """
    if not isinstance(runs, list) or len(runs) < 3:
        raise ConvergenceError("at least three refinement runs required")
    if isinstance(wavelength_um, bool) or not isinstance(wavelength_um, (float, int)) or not math.isfinite(wavelength_um) or wavelength_um <= 0:
        raise ConvergenceError("invalid wavelength")
    if not math.isfinite(max_relative_drift) or max_relative_drift <= 0:
        raise ConvergenceError("invalid drift tolerance")
    modes = [_complex_mode(r) for r in runs]
    sizes = [r["resolution_um"] for r in runs]
    if any(a <= b for a, b in zip(sizes, sizes[1:])):
        raise ConvergenceError("mesh sizes must strictly decrease")
    digests = [r["artifact_sha256"] for r in runs]
    if len(set(digests)) != len(digests):
        raise ConvergenceError("reused native artifact")
    for key in ("geometry_sha256", "material_source_sha256", "solver_config_sha256", "mode_track_id"):
        values = [r.get(key) for r in runs]
        if not all(isinstance(v, str) and v for v in values) or len(set(values)) != 1:
            raise ConvergenceError(f"nonconstant or missing {key}")
    k0 = 2 * math.pi / wavelength_um
    loss = [20 / math.log(10) * k0 * z.imag * 1e4 for z in modes]
    relative = [abs(b-a) / max(abs(b), 1e-15) for a,b in zip(modes, modes[1:])]
    return {
        "neff_complex_drift": relative,
        "loss_db_per_cm": loss,
        "last_pair_relative_drift_below_tolerance": relative[-1] <= max_relative_drift,
        "mode_identity_verified": False,
        "mesh_convergence_verified": False,
        "pml_convergence_verified": False,
        "domain_convergence_verified": False,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "limitations": "Numerical drift is diagnostic only; verify E/H overlap, PML, domain, Au provenance and independent solver."
    }
