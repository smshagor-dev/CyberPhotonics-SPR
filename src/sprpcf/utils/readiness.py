from __future__ import annotations

import importlib.util
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from sprpcf import __version__
from sprpcf.evidence.qualification import validate_evidence_registry
from sprpcf.utils.release import REQUIRED_RELEASE_FILES, validate_release

SUPPORTED_PYTHON = ((3, 10), (3, 11), (3, 12), (3, 13))
CORE_MODULES = (
    "numpy",
    "pandas",
    "pyarrow",
    "scipy",
    "sklearn",
    "matplotlib",
    "torch",
    "tqdm",
    "yaml",
)
CRITICAL_PACKAGE_MODULES = (
    "sprpcf.simulation.comsol_sweep",
    "sprpcf.ml.multiobjective",
    "sprpcf.validation.closed_loop",
    "sprpcf.validation.campaign",
    "sprpcf.edge.hardware",
    "sprpcf.dashboard.core",
    "sprpcf.evidence.qualification",
    "sprpcf.publication.evidence",
    "sprpcf.publication.submission",
    "sprpcf.utils.reproducibility",
)
OPTIONAL_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "dashboard": ("streamlit",),
    "onnx": ("onnx", "onnxruntime"),
    "hardware_serial": ("serial",),
    "comsol": ("mph",),
    "xai": ("shap",),
    "edge_litert": ("ai_edge_litert",),
    "edge_tensorflow": ("tensorflow",),
}
REQUIRED_RUNTIME_DIRS = ("data/raw", "data/processed", "models", "outputs")
REQUIRED_SYSTEM_FILES = (
    "configs/comsol_sweep.example.yaml",
    "configs/real_validation_campaign.example.yaml",
    "docs/EVIDENCE_QUALIFICATION.md",
    "docs/REAL_VALIDATION_CAMPAIGN.md",
    "docs/V1_SYSTEM_READINESS.md",
    "scripts/check_system_readiness.py",
    "scripts/register_evidence.py",
    "scripts/smoke_test_wheel.py",
    "scripts/validation_campaign.py",
)
FULL_EVIDENCE_CLASSES = ("comsol_physics", "experimental_sensor", "device_benchmark")


@dataclass(frozen=True)
class ReadinessCheck:
    name: str
    status: str
    required: bool
    detail: str


def _check(name: str, ok: bool, detail: str, *, required: bool = True) -> ReadinessCheck:
    return ReadinessCheck(name=name, status="pass" if ok else "fail", required=required, detail=detail)


def _info(name: str, available: bool, detail: str) -> ReadinessCheck:
    return ReadinessCheck(name=name, status="available" if available else "unavailable", required=False, detail=detail)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, AttributeError, ValueError):
        return False


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "dirty": None}
    return {"available": True, "commit": commit or None, "dirty": bool(status)}


def _resolve_manifest(path: str | Path | None, filename: str) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_dir():
        candidate = candidate / filename
    return candidate if candidate.is_file() else None


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _evidence_state(
    reviewer_package: str | Path | None,
    submission_package: str | Path | None,
    evidence_registry: str | Path | None,
) -> dict[str, Any]:
    """Collect metadata while treating the qualified registry as the only physical truth source."""
    reviewer_manifest = _resolve_manifest(reviewer_package, "manifest.json")
    submission_manifest = _resolve_manifest(submission_package, "submission_manifest.json")
    reviewer = _load_json(reviewer_manifest)
    submission = _load_json(submission_manifest)

    reviewer_classes = {str(value) for value in reviewer.get("evidence_classes", []) if value}
    submission_readiness = submission.get("readiness", {})
    if not isinstance(submission_readiness, dict):
        submission_readiness = {}

    registry_report: dict[str, Any] | None = None
    registry_path: Path | None = None
    qualified_classes: set[str] = set()
    if evidence_registry is not None:
        registry_path = Path(evidence_registry)
        registry_report = validate_evidence_registry(registry_path, verify_files=True)
        if registry_report.get("ok"):
            qualified_classes.update(str(value) for value in registry_report.get("evidence_classes", []) if value)

    return {
        "reviewer_manifest": str(reviewer_manifest) if reviewer_manifest else None,
        "submission_manifest": str(submission_manifest) if submission_manifest else None,
        "evidence_registry": str(registry_path) if registry_path else None,
        "evidence_registry_validation": registry_report,
        "evidence_classes": sorted(qualified_classes),
        "presentation_evidence_classes": sorted(reviewer_classes),
        "submission_readiness": submission_readiness,
        "physical_truth_source": "qualified_evidence_registry",
    }


def _runtime_dir_check(root: Path, relative: str) -> ReadinessCheck:
    path = root / relative
    exists = path.is_dir()
    writable = exists and os.access(path, os.W_OK)
    return _check(
        f"runtime_dir:{relative}",
        exists and writable,
        f"{path} exists={exists}, writable={writable}",
    )


def _comsol_example_check(root: Path) -> ReadinessCheck:
    path = root / "configs/comsol_sweep.example.yaml"
    if not path.is_file():
        return _check("comsol_example_contract", False, f"missing {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return _check("comsol_example_contract", False, f"invalid YAML: {exc}")
    if not isinstance(payload, dict):
        return _check("comsol_example_contract", False, "top-level YAML must be a mapping")
    sweep = payload.get("sweep")
    required_top = {"study", "wavelength_expression", "loss_expression", "expected_wavelength_nm", "sweep"}
    required_sweep = {"d_over_lambda", "pitch_um", "metal_thickness_nm", "analyte_ri"}
    missing_top = sorted(required_top - set(payload))
    missing_sweep = sorted(required_sweep - set(sweep)) if isinstance(sweep, dict) else sorted(required_sweep)
    ok = not missing_top and isinstance(sweep, dict) and not missing_sweep
    return _check("comsol_example_contract", ok, f"missing_top={missing_top}, missing_sweep={missing_sweep}")


def _campaign_example_check(root: Path) -> ReadinessCheck:
    path = root / "configs/real_validation_campaign.example.yaml"
    if not path.is_file():
        return _check("real_validation_campaign_contract", False, f"missing {path}")
    try:
        from sprpcf.validation.campaign import load_campaign_config

        payload = load_campaign_config(path)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        return _check("real_validation_campaign_contract", False, f"invalid campaign config: {exc}")
    return _check(
        "real_validation_campaign_contract",
        bool(payload.get("campaign_id")),
        f"campaign_id={payload.get('campaign_id')}",
    )


def build_readiness_report(
    repo_root: str | Path = ".",
    *,
    profile: str = "release",
    expected_version: str | None = None,
    reviewer_package: str | Path | None = None,
    submission_package: str | Path | None = None,
    evidence_registry: str | Path | None = None,
) -> dict[str, Any]:
    """Audit repository/runtime completeness without fabricating unavailable evidence."""
    if profile not in {"software", "release", "full"}:
        raise ValueError("profile must be one of: software, release, full")

    root = Path(repo_root).resolve()
    checks: list[ReadinessCheck] = []

    py = sys.version_info[:2]
    checks.append(
        _check(
            "python_version",
            py in SUPPORTED_PYTHON,
            f"running {platform.python_version()}; supported 3.10-3.13",
        )
    )
    checks.append(_check("package_version", bool(__version__), f"sprpcf {__version__}"))

    for module in CORE_MODULES:
        checks.append(_check(f"core_dependency:{module}", _module_available(module), f"module {module}"))
    for module in CRITICAL_PACKAGE_MODULES:
        checks.append(_check(f"package_module:{module}", _module_available(module), f"module {module}"))

    for relative in REQUIRED_RUNTIME_DIRS:
        checks.append(_runtime_dir_check(root, relative))

    system_files = REQUIRED_SYSTEM_FILES if profile in {"release", "full"} else ()
    for relative in system_files:
        checks.append(_check(f"system_file:{relative}", (root / relative).is_file(), relative))
    if profile in {"release", "full"}:
        checks.append(_comsol_example_check(root))
        checks.append(_campaign_example_check(root))

    release_validation: dict[str, Any] | None = None
    if profile in {"release", "full"}:
        release_validation = validate_release(root, expected_version=expected_version)
        checks.append(
            _check(
                "release_metadata",
                bool(release_validation.get("ok")),
                "; ".join(release_validation.get("errors", [])) or "version/citation/release metadata consistent",
            )
        )
        for relative in REQUIRED_RELEASE_FILES:
            checks.append(_check(f"release_file:{relative}", (root / relative).is_file(), relative))

    capability_status: dict[str, bool] = {}
    for capability, modules in OPTIONAL_CAPABILITIES.items():
        available = all(_module_available(module) for module in modules)
        capability_status[capability] = available
        checks.append(_info(f"optional_capability:{capability}", available, "modules=" + ",".join(modules)))

    git_state = _git_state(root)
    if git_state["available"]:
        checks.append(
            ReadinessCheck(
                name="git_worktree",
                status="warning" if git_state["dirty"] else "pass",
                required=False,
                detail=f"commit={git_state['commit']}, dirty={git_state['dirty']}",
            )
        )
    else:
        checks.append(ReadinessCheck("git_worktree", "unavailable", False, "git state unavailable"))

    evidence = _evidence_state(reviewer_package, submission_package, evidence_registry)
    registry_report = evidence.get("evidence_registry_validation")
    if evidence_registry is not None:
        registry_ok = bool(isinstance(registry_report, dict) and registry_report.get("ok"))
        registry_errors = registry_report.get("errors", []) if isinstance(registry_report, dict) else []
        checks.append(
            _check(
                "evidence_registry",
                registry_ok,
                "; ".join(str(value) for value in registry_errors) or "registry structure and artifact hashes valid",
                required=profile == "full",
            )
        )
    elif profile == "full":
        checks.append(
            _check(
                "evidence_registry",
                False,
                "full readiness requires an explicit qualified evidence registry",
            )
        )

    classes = set(evidence["evidence_classes"])
    missing_evidence = [name for name in FULL_EVIDENCE_CLASSES if name not in classes]
    if profile == "full":
        for evidence_class in FULL_EVIDENCE_CLASSES:
            checks.append(
                _check(
                    f"evidence:{evidence_class}",
                    evidence_class in classes,
                    (
                        f"qualified registry contains {evidence_class}"
                        if evidence_class in classes
                        else f"qualified registry does not contain {evidence_class}"
                    ),
                )
            )

    required_failures = [asdict(item) for item in checks if item.required and item.status == "fail"]
    warnings = [asdict(item) for item in checks if item.status in {"warning", "unavailable"} and not item.required]
    ready = not required_failures

    return {
        "schema_version": 3,
        "profile": profile,
        "ready": ready,
        "sprpcf_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "repo_root": str(root),
        "git": git_state,
        "release_validation": release_validation,
        "optional_capabilities": capability_status,
        "evidence": evidence,
        "missing_full_evidence": missing_evidence,
        "checks": [asdict(item) for item in checks],
        "required_failures": required_failures,
        "warnings": warnings,
    }


def readiness_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CyberPhotonics-SPR System Readiness",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Version: `{report['sprpcf_version']}`",
        f"- Ready: **{'YES' if report['ready'] else 'NO'}**",
        f"- Python: `{report['python']}`",
        "",
        "## Checks",
        "",
        "| Check | Status | Required | Detail |",
        "|---|---|---:|---|",
    ]
    for item in report["checks"]:
        detail = str(item["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(f"| `{item['name']}` | {item['status']} | {'yes' if item['required'] else 'no'} | {detail} |")

    lines.extend(["", "## Evidence gate", ""])
    classes = report["evidence"]["evidence_classes"]
    lines.append(
        "Qualified physical evidence classes: "
        + (", ".join(f"`{value}`" for value in classes) if classes else "none")
    )
    presentation = report["evidence"].get("presentation_evidence_classes", [])
    if presentation:
        lines.append(
            "Reviewer-package presentation classes (non-authoritative for the physical gate): "
            + ", ".join(f"`{value}`" for value in presentation)
        )
    registry = report["evidence"].get("evidence_registry")
    if registry:
        lines.append(f"Qualified evidence registry: `{registry}`")
    missing = report.get("missing_full_evidence", [])
    if missing:
        lines.append("")
        lines.append(
            "Full physical/experimental evidence still missing: "
            + ", ".join(f"`{value}`" for value in missing)
        )
    lines.extend(
        [
            "",
            "The readiness audit never upgrades synthetic, replay, surrogate, reviewer-package flags, or submission flags into COMSOL, experimental, or exact-device evidence.",
            "Physical readiness is satisfied only by a qualified evidence registry whose artifact hashes validate.",
            "Registry qualification validates provenance structure and artifact identity; it does not replace scientific review of the underlying experiment or simulation.",
            "",
        ]
    )
    return "\n".join(lines)
