"""Audit archive integrity only; scientific qualification requires external review."""
import hashlib
import json
import re
from pathlib import Path

REQUIRED = ("geometry", "material", "mesh", "pml", "domain", "solver_log", "e_field", "h_field", "eigenvalues")
_SHA = re.compile(r"[0-9a-fA-F]{64}\Z")


def audit(manifest_path):
    path = Path(manifest_path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    if not isinstance(record, dict):
        return {"integrity_verified": False, "qualified": False, "errors": ["invalid_manifest"], "publication_ready": False}
    if record.get("source") not in ("femwell-native", "ngsolve-native", "fenicsx-native"):
        errors.append("unsupported_source")
    if record.get("evidence_class") != "native_maxwell" or record.get("fixture") is not False:
        errors.append("unqualified_origin")
    if record.get("geometry_frozen") is not True or not record.get("material_citation") or record.get("material_citation") == "doi:placeholder":
        errors.append("missing_physics_provenance")
    if not all(record.get(k) is True for k in ("mesh_converged", "pml_converged", "domain_converged")):
        errors.append("unverified_convergence")
    if not all(record.get(k) is True for k in ("mode_overlap_verified", "independent_solver_verified")):
        errors.append("unverified_mode_or_solver")
    assets = record.get("assets")
    if not isinstance(assets, dict):
        assets = {}
        errors.append("invalid_assets")
    for key in REQUIRED:
        entry = assets.get(key)
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str) or not isinstance(entry.get("sha256"), str):
            errors.append(f"missing_asset:{key}")
            continue
        rel = Path(entry["path"])
        if rel.is_absolute() or ".." in rel.parts or not _SHA.fullmatch(entry["sha256"]):
            errors.append(f"unsafe_or_invalid_asset:{key}")
            continue
        target = (path.parent / rel).resolve()
        if not target.is_relative_to(path.parent) or not target.is_file():
            errors.append(f"unsafe_or_missing_asset:{key}")
            continue
        if hashlib.sha256(target.read_bytes()).hexdigest() != entry["sha256"].lower():
            errors.append(f"asset_digest_mismatch:{key}")
    return {"integrity_verified": not errors, "qualified": False, "errors": errors,
            "publication_ready": False}
