"""Fail-closed integrity gate for solver-native Maxwell artifacts.

This checks archive integrity, not physical correctness or publication readiness.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

REQUIRED = ("geometry", "mesh", "fields_e", "fields_h", "eigenvalues", "solver_log", "material_table")
SOURCES = {"native_fem", "native_fdtd", "comsol", "fixture", "synthetic", "surrogate"}


def audit(manifest_path: Path) -> dict:
    path = Path(manifest_path)
    m = json.loads(path.read_text(encoding="utf-8"))
    errors = []
    source = m.get("source_class")
    if source not in SOURCES:
        errors.append("invalid_source_class")
    if source in ("fixture", "synthetic", "surrogate"):
        errors.append("nonphysical_source")
    if source == "comsol" and not m.get("comsol_model_sha256"):
        errors.append("comsol_model_hash_missing")
    if source != "comsol" and m.get("comsol_model_sha256"):
        errors.append("comsol_provenance_mismatch")
    for key in ("solver_name", "solver_version", "git_commit", "geometry_id",
                "boundary_conditions", "mode_tracking_method"):
        if not isinstance(m.get(key), str) or not m[key].strip():
            errors.append("missing_" + key)
    if m.get("geometry_frozen") is not True:
        errors.append("geometry_not_frozen")
    if m.get("physics_qualified") is not False or m.get("publication_ready") is not False:
        errors.append("unsafe_publication_flags")
    for key in ("analyte_ri", "wavelength_nm"):
        val = m.get(key)
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            errors.append("invalid_" + key)
        elif key == "wavelength_nm" and val <= 0:
            errors.append("invalid_wavelength_nm")
    artifacts = m.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts_missing")
        artifacts = {}
    root = path.parent.resolve()
    for key in REQUIRED:
        item = artifacts.get(key)
        if not isinstance(item, dict):
            errors.append(key + "_record_missing")
            continue
        rel, digest = item.get("path"), item.get("sha256")
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            errors.append(key + "_unsafe_path")
            continue
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            errors.append(key + "_invalid_digest")
            continue
        target = (root / rel).resolve()
        if not target.is_relative_to(root) or not target.is_file():
            errors.append(key + "_file_missing_or_escaped")
            continue
        h = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1048576), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            errors.append(key + "_digest_mismatch")
    return {"integrity_traceability_pass": not errors, "physics_qualified": False,
            "publication_ready": False, "source_class": source,
            "failures": sorted(set(errors))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    result = audit(parser.parse_args().manifest)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["integrity_traceability_pass"] else 2)


if __name__ == "__main__":
    main()
