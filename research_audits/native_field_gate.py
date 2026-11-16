"""Fail-closed native E/H archive integrity gate. Structural checks are NOT physics validation."""
import hashlib
import json
import re
from pathlib import Path

REQUIRED = ("mesh", "electric_field", "magnetic_field", "eigenvalues", "solver_log", "material_data", "geometry")


def audit_native_fields(root):
    root = Path(root).resolve()
    failures = []
    try:
        manifest = json.loads((root / "native_fields_manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        manifest = {}
        failures.append("missing or malformed manifest: " + type(exc).__name__)
    if manifest.get("source_type") != "open_source_fem_native":
        failures.append("invalid source type")
    if manifest.get("comsol_model_verified") is not False:
        failures.append("COMSOL provenance must explicitly be false")
    files = manifest.get("files")
    if not isinstance(files, dict):
        files = {}
        failures.append("files must be an object")
    seen = set()
    for role in REQUIRED:
        entry = files.get(role)
        if not isinstance(entry, dict):
            failures.append("missing native role: " + role)
            continue
        name, digest = entry.get("path"), entry.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            failures.append("invalid path or digest: " + role)
            continue
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in seen:
            failures.append("unsafe or reused path: " + role)
            continue
        seen.add(name)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            failures.append("missing or escaping native file: " + role)
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append("native SHA256 mismatch: " + role)
    return {
        "schema_version": 1,
        "native_archive_integrity_passed": not failures,
        "failures": failures,
        "maxwell_residual_verified": False,
        "mesh_pml_domain_converged": False,
        "mode_identity_verified": False,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_evidence": False,
    }
