"""Fail-closed native-byte integrity audit; not a Maxwell-physics qualification."""
import hashlib
import json
import os
import stat
from pathlib import Path
from research.secure_native_paths import open_native_file

def audit(registry, artifact_root):
    root = Path(artifact_root).resolve(strict=True)
    if not root.is_dir() or not isinstance(registry, dict) or not isinstance(registry.get("sources"), list):
        raise ValueError("invalid root or sources")
    records, seen_ids, seen_paths = [], set(), set()
    for src in registry["sources"]:
        if not isinstance(src, dict):
            raise ValueError("invalid source")
        sid, rel, expected = src.get("id"), src.get("local_path"), src.get("sha256")
        if not isinstance(sid, str) or not sid.strip() or sid in seen_ids:
            raise ValueError("duplicate/invalid source ID")
        seen_ids.add(sid)
        if not isinstance(rel, str) or not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError("invalid relative path")
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
            raise ValueError("invalid SHA256")
        path = root
        for part in Path(rel).parts:
            if part in ("", "."):
                continue
            path = path / part
            if path.is_symlink():
                raise ValueError("symlink artifact path")
        try:
            resolved = path.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError) as exc:
            raise ValueError("missing artifact") from exc
        if not resolved.is_relative_to(root) or resolved in seen_paths:
            raise ValueError("escaping/duplicate artifact path")
        seen_paths.add(resolved)
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = open_native_file(root, rel)
        try:
            metadata = os.fstat(fd)
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError("artifact is not a regular file")
            digest, size = hashlib.sha256(), 0
            with os.fdopen(fd, "rb") as stream:
                fd = -1
                while chunk := stream.read(1048576):
                    digest.update(chunk)
                    size += len(chunk)
                after = os.fstat(stream.fileno())
                if (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns) != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) or size != metadata.st_size:
                    raise ValueError("artifact mutated during audit")
        finally:
            if fd >= 0:
                os.close(fd)
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
