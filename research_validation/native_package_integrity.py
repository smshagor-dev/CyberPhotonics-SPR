"""Structural integrity audit only; NEVER qualifies PCF-SPR physics or COMSOL."""
import hashlib
import json
from pathlib import Path

REQUIRED = ("geometry", "au_dispersion", "mesh", "pml", "domain", "spectral", "field_tracking", "solver_crosscheck")

def audit(manifest, root):
    failures = []
    if manifest.get("source") not in ("femwell", "ngsolve", "getdp", "fenicsx"):
        failures.append("unsupported_solver")
    if manifest.get("comsol_verified") is not False:
        failures.append("comsol_claim_not_explicitly_false")
    if manifest.get("fixture") is not False:
        failures.append("fixture_not_explicitly_false")
    if manifest.get("geometry_frozen") is not True or not str(manifest.get("geometry_reference") or "").strip():
        failures.append("geometry_not_frozen_or_uncited")
    checks = manifest.get("checks")
    if not isinstance(checks, dict):
        checks = {}
    for key in REQUIRED:
        if checks.get(key) is not True:
            failures.append("missing_" + key)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        failures.append("no_artifacts")
        artifacts = []
    root = Path(root).resolve()
    seen = set()
    for index, item in enumerate(artifacts):
        if not isinstance(item, dict):
            failures.append(f"artifact_{index}_invalid")
            continue
        name, digest = item.get("path"), item.get("sha256")
        if not isinstance(name, str) or not name or not isinstance(digest, str) or len(digest) != 64:
            failures.append(f"artifact_{index}_metadata")
            continue
        path = (root / name).resolve()
        if path in seen or not path.is_relative_to(root) or not path.is_file() or path.is_symlink():
            failures.append(f"artifact_{index}_missing_unsafe_duplicate")
            continue
        seen.add(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest.lower():
            failures.append(f"artifact_{index}_digest")
    return {
        "evidence_package_structurally_complete": not failures,
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "failures": failures,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    result = audit(json.loads(Path(args.manifest).read_text()), args.root)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["evidence_package_structurally_complete"] else 2)
