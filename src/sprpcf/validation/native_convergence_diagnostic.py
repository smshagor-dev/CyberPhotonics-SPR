"""Conservative native-eigenvalue convergence diagnostics; never a physics qualification.

All inputs must originate from independently archived native solves on the SAME
frozen geometry, material dataset, wavelength, RI and tracked physical mode.
This module cannot establish PML correctness or mode identity by itself.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class NativePoint:
    mesh_um: float
    domain_um: float
    pml_um: float
    neff_real: float
    neff_imag: float
    archive_sha256: str
    mode_id: str


def assess_convergence(
    points: list[NativePoint],
    *,
    max_relative_drift: float = 1e-3,
    min_levels: int = 3,
) -> dict:
    """Fail closed unless independent mesh, domain and PML sweeps exist.

    Diagnostic only: independent field/mode tracking, physical boundary checks,
    mesh quality and experimental/literature geometry are separate requirements.
    """
    result = {"diagnostic_pass": False, "publication_qualified": False, "reasons": [], "axes": {}}
    if not (math.isfinite(max_relative_drift) and 0 < max_relative_drift < 1):
        raise ValueError("relative drift tolerance must be finite and between 0 and 1")
    if min_levels < 3:
        raise ValueError("at least three distinct levels are required")
    if not points:
        result["reasons"].append("no native solver records")
        return result
    seen = set()
    hashes = set()
    modes = set()
    for p in points:
        vals = (p.mesh_um, p.domain_um, p.pml_um, p.neff_real, p.neff_imag)
        if not all(math.isfinite(v) for v in vals):
            result["reasons"].append("nonfinite native record")
            return result
        if min(p.mesh_um, p.domain_um, p.pml_um) <= 0:
            result["reasons"].append("nonpositive geometric scale")
            return result
        if not (len(p.archive_sha256) == 64 and all(c in "0123456789abcdef" for c in p.archive_sha256)):
            result["reasons"].append("missing or malformed immutable archive SHA256")
            return result
        if not p.mode_id.strip():
            result["reasons"].append("missing tracked mode identifier")
            return result
        key = (p.mesh_um, p.domain_um, p.pml_um)
        if key in seen or p.archive_sha256 in hashes:
            result["reasons"].append("duplicate configuration or archive")
            return result
        seen.add(key)
        hashes.add(p.archive_sha256)
        modes.add(p.mode_id)
    if len(modes) != 1:
        result["reasons"].append("mode identity differs between runs")
        return result
    # Only vary one axis at a time. A diagonal sweep cannot isolate errors.
    for axis in ("mesh_um", "domain_um", "pml_um"):
        groups = defaultdict(list)
        other = tuple(a for a in ("mesh_um", "domain_um", "pml_um") if a != axis)
        for p in points:
            groups[tuple(getattr(p, a) for a in other)].append(p)
        candidates = [g for g in groups.values() if len(g) >= min_levels]
        if not candidates:
            result["reasons"].append(f"missing independent {axis} sweep")
            continue
        best = None
        for group in candidates:
            # Finest mesh first; largest domain/PML first.
            ordered = sorted(group, key=lambda p: getattr(p, axis), reverse=axis != "mesh_um")
            first = complex(ordered[0].neff_real, ordered[0].neff_imag)
            second = complex(ordered[1].neff_real, ordered[1].neff_imag)
            drift = abs(first - second) / max(abs(first), 1e-15)
            if best is None or drift < best:
                best = drift
        result["axes"][axis] = {"relative_drift": best, "diagnostic_pass": best <= max_relative_drift}
        if best > max_relative_drift:
            result["reasons"].append(f"{axis} eigenvalue drift exceeds tolerance")
    result["diagnostic_pass"] = len(result["axes"]) == 3 and all(
        v["diagnostic_pass"] for v in result["axes"].values()
    )
    if result["diagnostic_pass"]:
        result["reasons"].append("numerical drift only; PML physics and mode identity unverified")
    return result
