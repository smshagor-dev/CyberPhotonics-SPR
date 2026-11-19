"""Byte-level integrity gate. Does not certify Maxwell physics or COMSOL."""
import hashlib
import json
from pathlib import Path


def audit(manifest_path, root):
    root = Path(root).resolve()
    errors = []
    try:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"integrity_verified": False, "physics_qualified": False,
                "publication_ready": False, "comsol_verified": False,
                "errors": [str(exc)]}
    if not isinstance(manifest, dict):
        manifest = {}
        errors.append("manifest must be an object")
    entries = manifest.get("artifacts")
    if not isinstance(entries, list) or not entries:
        entries = []
        errors.append("nonempty artifact list required")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            errors.append("artifact entry must be object")
            continue
        name, digest, size = entry.get("path"), entry.get("sha256"), entry.get("bytes")
        if (not isinstance(name, str) or not name or "\\" in name
                or Path(name).is_absolute()
                or any(x in (".", "..") for x in name.split("/"))
                or name in seen):
            errors.append("invalid or duplicate relative path")
            continue
        seen.add(name)
        if (not isinstance(digest, str) or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)):
            errors.append(f"invalid digest: {name}")
            continue
        if type(size) is not int or size < 0:
            errors.append(f"invalid byte count: {name}")
            continue
        path = root / name
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
            errors.append(f"absent or unsafe file: {name}")
            continue
        h = hashlib.sha256()
        actual_size = 0
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(block)
                actual_size += len(block)
        if h.hexdigest() != digest or actual_size != size:
            errors.append(f"content mismatch: {name}")
    for flag in ("physics_qualified", "publication_ready", "comsol_verified"):
        if manifest.get(flag) is not False:
            errors.append(f"{flag} must explicitly be false")
    return {"integrity_verified": not errors, "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False,
            "artifacts_checked": len(seen), "errors": errors}
