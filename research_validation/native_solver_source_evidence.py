"""Fail-closed source-specific evidence registry for native PCF-SPR Maxwell solves.

This checks provenance/completeness, NOT numerical Maxwell correctness.
No COMSOL claim is inferred from a Femwell or other open-source run.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path

REQUIRED = ("mesh", "electric_field", "magnetic_field", "eigenvalues",
            "solver_log", "geometry", "materials", "configuration")
REQUIRED_CHECKS = ("mesh_convergence", "pml_convergence", "domain_convergence",
                   "spectral_convergence", "mode_identity", "independent_solver")
ALLOWED_SOURCES = ("femwell", "ngsolve", "getdp", "fenicsx", "comsol")

class EvidenceError(ValueError):
    pass

def audit(root: Path, registry: dict) -> dict:
    """Audit an immutable manifest without ever promoting physics readiness."""
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise EvidenceError("Evidence root must be a directory")
    source = registry.get("source")
    if source not in ALLOWED_SOURCES:
        raise EvidenceError("Unknown numerical source")
    if source != "comsol" and (registry.get("comsol_verified") or registry.get("mph_model")):
        raise EvidenceError("Open-source results cannot claim COMSOL provenance")
    artifacts = registry.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != set(REQUIRED):
        raise EvidenceError("Exactly eight native artifact roles required")
    seen = set()
    digests = {}
    for role in REQUIRED:
        record = artifacts[role]
        if not isinstance(record, dict):
            raise EvidenceError("Malformed artifact record")
        name, expected = record.get("path"), record.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(expected, str) or len(expected) != 64:
            raise EvidenceError("Missing artifact path or SHA256")
        try:
            int(expected, 16)
        except ValueError as exc:
            raise EvidenceError("Nonhex SHA256") from exc
        path = root / name
        if path.is_symlink() or not path.is_file() or path.resolve() == root or not path.resolve().is_relative_to(root):
            raise EvidenceError("Missing, symlinked, or escaping artifact")
        if path.resolve() in seen:
            raise EvidenceError("Reused artifact across roles")
        seen.add(path.resolve())
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected.lower():
            raise EvidenceError("Artifact digest mismatch")
        digests[role] = actual
    checks = registry.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS):
        raise EvidenceError("All six numerical checks must be declared")
    if any(type(checks[k]) is not bool for k in REQUIRED_CHECKS):
        raise EvidenceError("Numerical checks must be booleans")
    if registry.get("physics_qualified") is not False or registry.get("publication_ready") is not False:
        raise EvidenceError("Publication flags must remain explicitly false pending independent verification")
    if not isinstance(registry.get("solver_version"), str) or not registry["solver_version"].strip():
        raise EvidenceError("Solver version missing")
    return {"source": source, "integrity_verified": True,
            "numerical_qualification_verified": False,
            "physics_qualified": False, "publication_ready": False,
            "checks": checks, "digests": digests}

def audit_manifest(root: Path, manifest: Path) -> dict:
    return audit(root, json.loads(Path(manifest).read_text(encoding="utf-8")))
