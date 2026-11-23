"""Verify native artifact bytes, not just claimed GitHub metadata. Never certifies physics."""
import hashlib
import os
from pathlib import Path
import re

def verify_native_bytes(path, expected_sha256, *, max_bytes=2_000_000_000):
    result = dict(integrity_verified=False, physics_qualified=False,
                  publication_ready=False, comsol_verified=False)
    if not isinstance(expected_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        return {**result, "reason": "invalid_expected_digest"}
    if type(max_bytes) is not int or max_bytes <= 0:
        return {**result, "reason": "invalid_size_limit"}
    try:
        p = Path(path)
        if p.is_symlink() or not p.is_file():
            return {**result, "reason": "not_regular_file"}
        initial = p.stat()
        if initial.st_size > max_bytes:
            return {**result, "reason": "size_limit_exceeded"}
        digest = hashlib.sha256()
        total = 0
        with p.open("rb") as stream:
            if not os.path.samestat(initial, os.fstat(stream.fileno())):
                return {**result, "reason": "file_changed_during_open"}
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                total += len(chunk)
                if total > max_bytes:
                    return {**result, "reason": "size_limit_exceeded"}
                digest.update(chunk)
            final = os.fstat(stream.fileno())
        if (not os.path.samestat(initial, final) or initial.st_size != total or
                initial.st_mtime_ns != final.st_mtime_ns):
            return {**result, "reason": "file_changed_during_read"}
        actual = digest.hexdigest()
        return {**result, "integrity_verified": actual == expected_sha256,
                "reason": "match" if actual == expected_sha256 else "digest_mismatch",
                "sha256": actual, "size_bytes": total}
    except (OSError, ValueError, TypeError):
        return {**result, "reason": "unreadable_file"}
