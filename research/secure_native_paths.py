"""POSIX descriptor-relative native artifact reader; software integrity only.

Rejects symlinked path components and nonregular targets. The caller owns the
returned file descriptor. No pathname is reopened after validation.
"""
import os
import stat
from pathlib import PurePosixPath


def open_native_file(root, relative):
    if os.name != "posix" or not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
        raise RuntimeError("secure descriptor traversal unavailable on this platform")
    if not isinstance(relative, str) or not relative or "\\" in relative or "\x00" in relative:
        raise ValueError("invalid artifact path")
    p = PurePosixPath(relative)
    if p.is_absolute() or any(x in ("", ".", "..") for x in relative.split("/")):
        raise ValueError("invalid artifact path components")
    dirflags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    current = os.open(root, dirflags)
    try:
        for part in p.parts[:-1]:
            nextfd = os.open(part, dirflags, dir_fd=current)
            os.close(current)
            current = nextfd
        fd = os.open(
            p.parts[-1],
            os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NONBLOCK", 0),
            dir_fd=current,
        )
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("not a regular file")
            return fd
        except BaseException:
            os.close(fd)
            raise
    finally:
        os.close(current)
