"""Fail-closed source-specific artifact integrity; NOT numerical physics qualification."""
import hashlib
from pathlib import Path

SOURCES = frozenset(("open_femwell", "open_cross_solver", "comsol"))
REQUIRED = frozenset(("geometry", "materials", "mesh", "boundary", "eigenfields", "solver_log"))

def audit_source_bundle(root, manifest):
    root = Path(root).resolve()
    if not isinstance(manifest, dict):
        return {"integrity_passed": False, "publication_qualified": False, "failures": ["invalid_manifest"]}
    failures = []
    source = manifest.get("source")
    if source not in SOURCES:
        failures.append("unknown_source")
    if manifest.get("synthetic") is not False or manifest.get("fixture") is not False:
        failures.append("non_native_or_unverified_origin")
    if source == "comsol" and not manifest.get("mph_model_sha256"):
        failures.append("missing_comsol_model_hash")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        artifacts = {}
        failures.append("invalid_artifact_map")
    for role in sorted(REQUIRED - artifacts.keys()):
        failures.append("missing_" + role)
    seen = set()
    for role, entry in artifacts.items():
        if not isinstance(entry, dict):
            failures.append("invalid_entry_" + str(role))
            continue
        name, digest = entry.get("path"), entry.get("sha256")
        if (not isinstance(name, str) or not name or not isinstance(digest, str)
                or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            failures.append("invalid_reference_" + str(role))
            continue
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            failures.append("missing_or_unsafe_" + str(role))
            continue
        if path in seen:
            failures.append("duplicate_artifact_path")
        seen.add(path)
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1048576), b""):
                h.update(chunk)
        if h.hexdigest() != digest:
            failures.append("hash_mismatch_" + str(role))
    return {"source": source, "integrity_passed": not failures,
            "publication_qualified": False, "failures": sorted(set(failures)),
            "qualification_scope": "artifact_integrity_only"}
