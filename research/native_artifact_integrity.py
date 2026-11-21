"""Byte-level native artifact audit; no physics qualification."""
import hashlib
import json
from pathlib import Path

def audit(registry, artifact_root):
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir() or not isinstance(registry, dict) or not isinstance(registry.get("sources"), list):
        raise ValueError("invalid root or sources")
    records, seen = [], set()
    for src in registry["sources"]:
        if not isinstance(src, dict):
            raise ValueError("invalid source")
        sid, rel, expected = src.get("id"), src.get("local_path"), src.get("sha256")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("duplicate/invalid source ID")
        seen.add(sid)
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("invalid relative path")
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("invalid SHA256")
        path = (root / rel).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("missing/escaping artifact")
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1048576):
                digest.update(chunk)
                size += len(chunk)
        records.append({"id": sid, "bytes": size, "sha256_match": digest.hexdigest() == expected})
    return {"schema": "pcf_spr_native_bytes_v1", "records": records,
            "all_bytes_verified": bool(records) and all(x["sha256_match"] for x in records),
            "physics_qualified": False, "publication_ready": False, "comsol_verified": False}

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("registry")
    p.add_argument("artifact_root")
    p.add_argument("output")
    args = p.parse_args()
    result = audit(json.loads(Path(args.registry).read_text()), args.artifact_root)
    Path(args.output).write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    if not result["all_bytes_verified"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
