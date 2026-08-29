from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

from sprpcf import __version__
from sprpcf.publication.finalization import validate_finalization_package
from sprpcf.publication.results import validate_paper_results_package
from sprpcf.utils.release import validate_release
from sprpcf.utils.reproducibility import git_state, sha256_file

_STABLE_PATTERN = re.compile(r"^(\d+)\.(\d+)\.(\d+)$")
_PRERELEASE_BASE_PATTERN = re.compile(r"^(\d+\.\d+\.\d+)(?:a|b|rc)\d+$", re.IGNORECASE)
CERTIFICATE_NAME = "STABLE_RELEASE_EVIDENCE.json"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _current_stable_base(version: str) -> str:
    match = _PRERELEASE_BASE_PATTERN.fullmatch(version)
    return match.group(1) if match else version


def _rc_finalization_ready(finalization_dir: Path, validation: Mapping[str, Any]) -> tuple[bool, list[str]]:
    if not validation.get("ok"):
        return False, ["finalization package integrity failed"]
    manifest_path = finalization_dir / "FINALIZATION_MANIFEST.json"
    if not manifest_path.is_file():
        return False, ["finalization manifest is missing"]
    manifest = _read_json(manifest_path)
    blockers: list[str] = []
    if not manifest.get("full_readiness"):
        blockers.append("finalization full physical readiness failed")
    missing = manifest.get("missing_physical_classes", [])
    if isinstance(missing, list) and missing:
        blockers.append("finalization still has missing physical evidence: " + ", ".join(str(value) for value in missing))
    unexpected = [
        str(row.get("gate") or "unknown")
        for row in manifest.get("blockers", [])
        if isinstance(row, Mapping) and str(row.get("gate") or "") != "stable_version"
    ]
    if unexpected:
        blockers.append("finalization has non-version blockers: " + ", ".join(unexpected))
    return not blockers, blockers


def build_stable_release_plan(
    *,
    repo_root: str | Path,
    finalization_dir: str | Path,
    paper_results_dir: str | Path,
    target_version: str = "1.0.0",
    require_clean_git: bool = True,
) -> dict[str, Any]:
    root = Path(repo_root)
    finalization = Path(finalization_dir)
    results = Path(paper_results_dir)
    stable_target = bool(_STABLE_PATTERN.fullmatch(target_version))
    final_validation = validate_finalization_package(finalization)
    results_validation = validate_paper_results_package(results)
    release_validation = validate_release(root, expected_version=__version__)
    git = git_state(root)
    rc_finalization_ready, finalization_blockers = _rc_finalization_ready(finalization, final_validation)

    blockers: list[str] = []
    if not stable_target:
        blockers.append(f"target version {target_version!r} is not a stable semantic version")
    if stable_target and _current_stable_base(__version__) != target_version:
        blockers.append(
            f"target version {target_version!r} does not match prerelease base {_current_stable_base(__version__)!r}"
        )
    blockers.extend(finalization_blockers)
    if not results_validation.get("ok"):
        blockers.append("paper-results package integrity failed")
    if not results_validation.get("ready_for_manuscript_results"):
        blockers.append("paper-results package is not ready for evidence-backed manuscript results")
    if not release_validation.get("ok"):
        blockers.append("current release metadata validation failed")
    if require_clean_git and git.get("available") and git.get("dirty"):
        blockers.append("repository worktree is dirty")

    final_manifest_path = finalization / "FINALIZATION_MANIFEST.json"
    results_manifest_path = results / "PAPER_RESULTS_MANIFEST.json"
    return {
        "schema_version": 1,
        "source_version": __version__,
        "target_version": target_version,
        "stable_target": stable_target,
        "require_clean_git": require_clean_git,
        "repo_root": str(root),
        "git": git,
        "finalization": {
            "path": str(finalization),
            "validation": final_validation,
            "rc_promotion_ready": rc_finalization_ready,
            "manifest_sha256": sha256_file(final_manifest_path) if final_manifest_path.is_file() else None,
        },
        "paper_results": {
            "path": str(results),
            "validation": results_validation,
            "manifest_sha256": sha256_file(results_manifest_path) if results_manifest_path.is_file() else None,
        },
        "release_metadata": release_validation,
        "ready_for_promotion": not blockers,
        "blockers": blockers,
        "scientific_boundary": (
            "A release candidate may be promoted when all physical-evidence and full-readiness gates pass and the only "
            "remaining finalization blocker is the prerelease version itself. Production promotion keeps the clean-Git "
            "gate enabled. The promotion action does not create a Git tag, GitHub Release, DOI, or publication record. "
            "Finalization should be rebuilt after promotion before tagging."
        ),
    }


def stable_release_plan_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Stable Release Promotion Plan",
        "",
        f"- Source version: `{plan['source_version']}`",
        f"- Target version: `{plan['target_version']}`",
        f"- Clean Git required: **{'yes' if plan.get('require_clean_git', True) else 'no'}**",
        f"- RC finalization evidence-ready: **{'yes' if plan['finalization']['rc_promotion_ready'] else 'no'}**",
        f"- Ready for promotion: **{'yes' if plan['ready_for_promotion'] else 'no'}**",
        "",
        "## Blockers",
        "",
    ]
    blockers = plan.get("blockers", [])
    lines.extend(f"- {value}" for value in blockers) if blockers else lines.append("- None.")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            str(plan["scientific_boundary"]),
            "",
        ]
    )
    return "\n".join(lines)


def _replace_project_version(text: str, old: str, new: str) -> str:
    pattern = re.compile(r'(?ms)(^\[project\]\s*.*?^version\s*=\s*")' + re.escape(old) + r'(".*?)(?=^\[|\Z)')
    replaced, count = pattern.subn(r"\g<1>" + new + r"\g<2>", text, count=1)
    if count != 1:
        raise ValueError("Could not replace [project] version in pyproject.toml exactly once.")
    return replaced


def _replace_package_version(text: str, old: str, new: str) -> str:
    pattern = re.compile(r'(?m)^__version__\s*=\s*"' + re.escape(old) + r'"$')
    replaced, count = pattern.subn(f'__version__ = "{new}"', text, count=1)
    if count != 1:
        raise ValueError("Could not replace __version__ exactly once.")
    return replaced


def apply_stable_version(plan: dict[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    if not plan.get("ready_for_promotion"):
        raise ValueError("Stable release plan is blocked; refusing to modify version metadata.")
    if plan.get("require_clean_git") is not True:
        raise ValueError("Stable version application requires a plan created with require_clean_git=True.")
    root = Path(repo_root)
    old = str(plan["source_version"])
    new = str(plan["target_version"])

    pyproject = root / "pyproject.toml"
    package_init = root / "src" / "sprpcf" / "__init__.py"
    citation = root / "CITATION.cff"
    pyproject.write_text(
        _replace_project_version(pyproject.read_text(encoding="utf-8"), old, new),
        encoding="utf-8",
    )
    package_init.write_text(
        _replace_package_version(package_init.read_text(encoding="utf-8"), old, new),
        encoding="utf-8",
    )
    citation_payload = yaml.safe_load(citation.read_text(encoding="utf-8"))
    if not isinstance(citation_payload, dict) or str(citation_payload.get("version")) != old:
        raise ValueError("CITATION.cff version does not match the source version.")
    citation_payload["version"] = new
    citation.write_text(yaml.safe_dump(citation_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    certificate = {
        "schema_version": 1,
        "source_version": old,
        "target_version": new,
        "source_git": plan.get("git", {}),
        "finalization_manifest_sha256": plan["finalization"]["manifest_sha256"],
        "paper_results_manifest_sha256": plan["paper_results"]["manifest_sha256"],
        "qualification_statement": (
            "Generated by the strict stable-release promotion gate after qualified physical evidence, full RC "
            "readiness, paper-results readiness, and a clean Git-worktree check passed. Finalization must be rebuilt at "
            "the stable version before creating the stable tag."
        ),
    }
    certificate_path = root / CERTIFICATE_NAME
    certificate_path.write_text(json.dumps(certificate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return certificate


def validate_stable_release_certificate(
    repo_root: str | Path,
    *,
    expected_version: str | None = None,
) -> dict[str, Any]:
    root = Path(repo_root)
    path = root / CERTIFICATE_NAME
    errors: list[str] = []
    if not path.is_file():
        return {"ok": False, "errors": [f"Missing {CERTIFICATE_NAME}"], "certificate": None}
    try:
        payload = _read_json(path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return {"ok": False, "errors": [str(exc)], "certificate": None}
    target = str(payload.get("target_version") or "")
    if not _STABLE_PATTERN.fullmatch(target):
        errors.append("Certificate target_version must be a stable semantic version.")
    if expected_version is not None and target != expected_version:
        errors.append(f"Certificate target_version {target!r} != expected {expected_version!r}.")
    if str(payload.get("source_version") or "") == target:
        errors.append("Certificate source_version must be a prerelease version distinct from target_version.")
    for key in ("finalization_manifest_sha256", "paper_results_manifest_sha256"):
        digest = str(payload.get(key) or "")
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest.lower()):
            errors.append(f"Certificate {key} is not a SHA-256 digest.")
    return {"ok": not errors, "errors": errors, "certificate": payload}
