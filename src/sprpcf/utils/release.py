from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml


REQUIRED_RELEASE_FILES = (
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "Dockerfile",
    ".dockerignore",
    ".devcontainer/devcontainer.json",
    ".github/release.yml",
    "RELEASE_CANDIDATE.md",
    "configs/comsol_sweep.example.yaml",
    "docs/EVIDENCE_QUALIFICATION.md",
    "docs/REPRODUCIBILITY.md",
    "docs/PUBLICATION_REVIEWER_PACKAGE.md",
    "docs/SUBMISSION_PACKAGE.md",
    "docs/V1_SYSTEM_READINESS.md",
    "scripts/build_reviewer_package.py",
    "scripts/build_submission_package.py",
    "scripts/check_system_readiness.py",
    "scripts/register_evidence.py",
    "scripts/run_publication_demo.py",
    "scripts/smoke_test_wheel.py",
    "pyproject.toml",
)
RUNTIME_PREFIXES = ("data/raw/", "data/processed/", "models/", "outputs/")
_VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?$")


def _read_project_version(pyproject_text: str) -> str | None:
    project_match = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", pyproject_text)
    if not project_match:
        return None
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', project_match.group(1), re.MULTILINE)
    return version_match.group(1) if version_match else None


def _read_package_version(init_text: str) -> str | None:
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', init_text, re.MULTILINE)
    return match.group(1) if match else None


def _tracked_files(repo_root: Path) -> list[str]:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    return [line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()]


def validate_release(repo_root: str | Path = ".", expected_version: str | None = None) -> dict[str, Any]:
    """Validate research-release metadata without publishing anything."""
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    for relative in REQUIRED_RELEASE_FILES:
        if not (root / relative).is_file():
            errors.append(f"Missing required release file: {relative}")

    pyproject_path = root / "pyproject.toml"
    init_path = root / "src/sprpcf/__init__.py"
    citation_path = root / "CITATION.cff"
    candidate_path = root / "RELEASE_CANDIDATE.md"

    project_version = _read_project_version(pyproject_path.read_text(encoding="utf-8")) if pyproject_path.is_file() else None
    package_version = _read_package_version(init_path.read_text(encoding="utf-8")) if init_path.is_file() else None
    citation: dict[str, Any] = {}
    if citation_path.is_file():
        loaded = yaml.safe_load(citation_path.read_text(encoding="utf-8"))
        citation = loaded if isinstance(loaded, dict) else {}

    citation_version = str(citation.get("version")) if citation.get("version") is not None else None
    versions = {"pyproject": project_version, "package": package_version, "citation": citation_version}
    if any(version is None for version in versions.values()):
        errors.append(f"Could not read all release versions: {versions}")
    elif len(set(versions.values())) != 1:
        errors.append(f"Release version mismatch: {versions}")

    if project_version is not None and not _VERSION_PATTERN.fullmatch(project_version):
        errors.append(f"Unsupported project version format: {project_version!r}.")
    if expected_version is not None and project_version != expected_version:
        errors.append(f"Tag/expected version {expected_version!r} does not match project version {project_version!r}.")
    if project_version is not None and "rc" in project_version:
        if not candidate_path.is_file():
            errors.append("Release-candidate version requires RELEASE_CANDIDATE.md.")
        elif project_version not in candidate_path.read_text(encoding="utf-8"):
            errors.append("RELEASE_CANDIDATE.md does not name the active project version.")

    required_citation_fields = (
        "cff-version",
        "message",
        "title",
        "type",
        "authors",
        "repository-code",
        "license",
        "version",
    )
    for field in required_citation_fields:
        if not citation.get(field):
            errors.append(f"CITATION.cff missing required project field: {field}")
    authors = citation.get("authors")
    if authors is not None and not isinstance(authors, list):
        errors.append("CITATION.cff authors must be a list.")

    tracked_runtime = []
    tracked = _tracked_files(root)
    for path in tracked:
        if path.endswith("/.gitkeep"):
            continue
        if any(path.startswith(prefix) for prefix in RUNTIME_PREFIXES):
            tracked_runtime.append(path)
    if tracked_runtime:
        errors.append("Generated/raw runtime artifacts are tracked: " + ", ".join(sorted(tracked_runtime)))

    if not tracked:
        warnings.append("Git tracked-file inspection unavailable; runtime-artifact check was skipped.")

    return {
        "ok": not errors,
        "versions": versions,
        "expected_version": expected_version,
        "errors": errors,
        "warnings": warnings,
    }
