from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


def seed_everything(seed: int, include_tensorflow: bool = False) -> None:
    """Seed supported RNGs without making optional dependencies mandatory."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if include_tensorflow:
        try:
            import tensorflow as tf
        except ImportError:
            return
        tf.keras.utils.set_random_seed(seed)


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""
    file_path = Path(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    """Return a stable, normalized snapshot of installed Python distributions."""
    packages: dict[str, str] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name")
        if not name:
            continue
        packages[name.lower().replace("_", "-")] = distribution.version
    return dict(sorted(packages.items()))


def _git_command(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def git_state(repo_root: str | Path = ".") -> dict[str, Any]:
    """Capture commit and dirty state without failing outside a Git checkout."""
    root = Path(repo_root)
    commit = _git_command(root, "rev-parse", "HEAD")
    status = _git_command(root, "status", "--porcelain") if commit else None
    return {
        "available": commit is not None,
        "commit": commit,
        "dirty": bool(status) if status is not None else None,
    }


def environment_snapshot() -> dict[str, Any]:
    """Capture a privacy-conscious software/runtime environment snapshot."""
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "architecture": platform.architecture()[0],
        },
        "packages": package_versions(),
    }


def _portable_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.name


def artifact_metadata(path: str | Path, role: str, repo_root: str | Path = ".") -> dict[str, Any]:
    """Hash an experiment artifact while avoiding machine-specific absolute paths."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Artifact does not exist or is not a file: {file_path}")
    root = Path(repo_root)
    return {
        "role": role,
        "path": _portable_path(file_path, root),
        "size_bytes": file_path.stat().st_size,
        "sha256": sha256_file(file_path),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def create_reproducibility_bundle(
    output_dir: str | Path,
    *,
    experiment_name: str,
    seed: int,
    artifacts: Iterable[tuple[str, str | Path]] = (),
    config: Mapping[str, Any] | None = None,
    repo_root: str | Path = ".",
    notes: str | None = None,
) -> dict[str, Any]:
    """Write a portable provenance bundle for a completed or planned experiment.

    The bundle records hashes and metadata only; it never copies large datasets or
    trained models and therefore cannot silently turn synthetic evidence into a
    claimed physical result.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(repo_root)
    artifact_rows = [artifact_metadata(path, role, root) for role, path in artifacts]
    environment = environment_snapshot()
    try:
        sprpcf_version = metadata.version("sprpcf")
    except metadata.PackageNotFoundError:
        sprpcf_version = None

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": {
            "name": experiment_name,
            "seed": int(seed),
            "notes": notes,
        },
        "software": {
            "sprpcf_version": sprpcf_version,
            "git": git_state(root),
        },
        "config": dict(config or {}),
        "artifacts": artifact_rows,
    }
    _write_json(out / "manifest.json", manifest)
    _write_json(out / "environment.json", environment)

    lock_lines = [f"{name}=={version}" for name, version in environment["packages"].items()]
    (out / "environment.lock.txt").write_text("\n".join(lock_lines) + "\n", encoding="utf-8")

    checksum_lines = [f"{row['sha256']}  {row['role']}:{row['path']}" for row in artifact_rows]
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + ("\n" if checksum_lines else ""), encoding="utf-8")

    reproduce = [
        "# Reproduction Record",
        "",
        f"Experiment: `{experiment_name}`",
        f"Seed: `{seed}`",
        "",
        "This directory is an evidence/provenance bundle. Large datasets and model binaries are referenced by SHA-256 rather than copied.",
        "Recreate the software environment from the repository and compare artifact hashes before using the results in a publication.",
        "",
        "```bash",
        "python -m venv .venv",
        "python -m pip install --upgrade pip",
        "pip install -e \".[dev,onnx]\"",
        "python scripts/verify_release.py",
        "```",
        "",
        "`environment.lock.txt` is a snapshot of the environment that produced this bundle; platform-specific packages may require the repository installation instructions on another machine.",
    ]
    (out / "REPRODUCE.md").write_text("\n".join(reproduce) + "\n", encoding="utf-8")
    return manifest
