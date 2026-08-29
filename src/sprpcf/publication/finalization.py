from __future__ import annotations

import csv
import json
import re
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from sprpcf import __version__
from sprpcf.evidence.qualification import PHYSICAL_EVIDENCE_CLASSES, validate_evidence_registry
from sprpcf.publication.evidence import EvidenceSource, build_reviewer_package
from sprpcf.publication.submission import build_submission_package, validate_submission_package
from sprpcf.utils.readiness import build_readiness_report
from sprpcf.utils.reproducibility import sha256_file

_REQUIRED_PHYSICAL_CLASSES = tuple(PHYSICAL_EVIDENCE_CLASSES)
_PRERELEASE_PATTERN = re.compile(r"(?:a|b|rc)\d+$", re.IGNORECASE)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _artifact_path(registry_path: Path, stored: str) -> Path:
    candidate = Path(stored)
    return candidate if candidate.is_absolute() else (registry_path.resolve().parent / candidate).resolve()


def _qualified_registry_sources(registry_path: Path) -> tuple[list[EvidenceSource], dict[str, Any]]:
    validation = validate_evidence_registry(registry_path, verify_files=True)
    if not validation.get("ok"):
        errors = "; ".join(str(value) for value in validation.get("errors", []))
        raise ValueError(f"Qualified evidence registry is invalid: {errors}")

    payload = _read_json(registry_path)
    sources = [
        EvidenceSource(
            role="qualified_evidence_registry",
            path=registry_path,
            evidence_class="reproducibility",
            label="Qualified physical-evidence registry",
        )
    ]
    for record in payload.get("records", []):
        if not isinstance(record, Mapping) or record.get("qualified") is not True:
            continue
        evidence_class = str(record.get("evidence_class") or "")
        if evidence_class not in _REQUIRED_PHYSICAL_CLASSES:
            continue
        record_id = str(record.get("record_id") or "unknown")[:12]
        record_label = str(record.get("label") or evidence_class.replace("_", " ").title())
        artifacts = record.get("artifacts", [])
        if not isinstance(artifacts, list):
            continue
        for artifact in artifacts:
            if not isinstance(artifact, Mapping):
                continue
            stored = str(artifact.get("path") or "")
            source = _artifact_path(registry_path, stored)
            if not source.is_file():
                continue
            artifact_role = str(artifact.get("role") or "artifact").replace("-", "_")
            sources.append(
                EvidenceSource(
                    role=f"qualified_{evidence_class}_{record_id}_{artifact_role}",
                    path=source,
                    evidence_class=evidence_class,
                    label=f"{record_label}: {artifact_role}",
                )
            )
    return sources, validation


def _append_optional_source(
    sources: list[EvidenceSource],
    *,
    role: str,
    path: str | Path | None,
    evidence_class: str,
    label: str,
) -> None:
    if path is None:
        return
    sources.append(EvidenceSource(role=role, path=Path(path), evidence_class=evidence_class, label=label))


def _safe_prepare_output(output_dir: Path, *, replace: bool) -> None:
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return
    if not replace:
        raise FileExistsError(f"Finalization output is not empty: {output_dir}")
    marker = output_dir / "FINALIZATION_MANIFEST.json"
    if not marker.is_file():
        raise ValueError(
            "Refusing to replace a non-empty directory that was not created by the finalization pipeline."
        )
    shutil.rmtree(output_dir)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_blocker_markdown(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# Finalization Blocker Matrix",
        "",
        "| Gate | Status | Required | Detail |",
        "|---|---|---:|---|",
    ]
    for row in rows:
        detail = str(row["detail"]).replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{row['gate']}` | {row['status']} | {'yes' if row['required'] else 'no'} | {detail} |"
        )
    lines.extend(
        [
            "",
            "A passing packaging gate proves package integrity and provenance structure only. It does not replace scientific review of COMSOL setup, laboratory protocol, calibration, fabrication, or target-device measurements.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_delta_markdown(path: Path, delta: Mapping[str, Any]) -> None:
    lines = [
        "# Evidence Delta Report",
        "",
        f"- Qualified physical classes present: {', '.join(f'`{value}`' for value in delta['present_physical_classes']) or 'none'}",
        f"- Qualified physical classes missing: {', '.join(f'`{value}`' for value in delta['missing_physical_classes']) or 'none'}",
        "",
        "## Claim families",
        "",
        "| Claim | Status | Evidence class(es) |",
        "|---|---|---|",
    ]
    for row in delta["claims"]:
        claim = str(row["claim"]).replace("|", "\\|")
        classes = str(row.get("evidence_classes", "")).replace("|", "\\|")
        lines.append(f"| {claim} | {row['status']} | {classes} |")
    lines.extend(
        [
            "",
            "`not supplied` is an evidence gap. The finalization pipeline does not substitute synthetic, surrogate, replay, or workstation outputs for missing physical evidence.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _package_checksums(root: Path) -> Path:
    checksum_path = root / "checksums.sha256"
    targets = sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file() and candidate != checksum_path),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    )
    checksum_path.write_text(
        "\n".join(f"{sha256_file(candidate)}  {candidate.relative_to(root).as_posix()}" for candidate in targets)
        + "\n",
        encoding="utf-8",
    )
    return checksum_path


def build_evidence_finalization_package(
    output_dir: str | Path,
    *,
    evidence_registry: str | Path,
    repo_root: str | Path = ".",
    version: str = __version__,
    journal: str | None = None,
    manuscript: str | Path | None = None,
    validation_dir: str | Path | None = None,
    ablation_dir: str | Path | None = None,
    design_dir: str | Path | None = None,
    closed_loop_dir: str | Path | None = None,
    hardware_dir: str | Path | None = None,
    reproducibility_dir: str | Path | None = None,
    release_validation: str | Path | None = None,
    replace: bool = False,
) -> dict[str, Any]:
    """Build a deterministic reviewer/submission finalization view from qualified evidence.

    Physical claim support is derived only from a validated evidence registry. Optional software,
    surrogate, reproducibility, or presentation artifacts retain their non-physical classes.
    """
    out = Path(output_dir)
    registry = Path(evidence_registry)
    root = Path(repo_root)
    if not registry.is_file():
        raise FileNotFoundError(f"Qualified evidence registry not found: {registry}")
    _safe_prepare_output(out, replace=replace)
    out.mkdir(parents=True, exist_ok=True)

    sources, registry_validation = _qualified_registry_sources(registry)
    _append_optional_source(
        sources,
        role="validation",
        path=validation_dir,
        evidence_class="software_only",
        label="Scientific validation pack",
    )
    _append_optional_source(
        sources,
        role="ablation",
        path=ablation_dir,
        evidence_class="software_only",
        label="Physics-loss ablation evidence",
    )
    _append_optional_source(
        sources,
        role="design",
        path=design_dir,
        evidence_class="surrogate_model",
        label="Pareto/inverse-design evidence",
    )
    _append_optional_source(
        sources,
        role="closed_loop",
        path=closed_loop_dir,
        evidence_class="software_only",
        label="Additional closed-loop software artifacts",
    )
    _append_optional_source(
        sources,
        role="hardware",
        path=hardware_dir,
        evidence_class="software_only",
        label="Additional hardware/replay software artifacts",
    )
    _append_optional_source(
        sources,
        role="reproducibility",
        path=reproducibility_dir,
        evidence_class="reproducibility",
        label="Reproducibility/provenance bundle",
    )
    _append_optional_source(
        sources,
        role="release_validation",
        path=release_validation,
        evidence_class="release",
        label="Release validation",
    )

    reviewer_dir = out / "reviewer_package"
    reviewer = build_reviewer_package(
        reviewer_dir,
        sources=sources,
        title="CyberPhotonics-SPR Final Reviewer Evidence Package",
        version=version,
        repo_root=root,
    )
    submission_dir = out / "submission_package"
    submission = build_submission_package(
        submission_dir,
        reviewer_package_dir=reviewer_dir,
        title="CyberPhotonics-SPR Final Manuscript Supplementary Package",
        version=version,
        repo_root=root,
        journal=journal,
        manuscript_file=manuscript,
        validation_dir=validation_dir,
        ablation_dir=ablation_dir,
        design_dir=design_dir,
        closed_loop_dir=closed_loop_dir,
        hardware_dir=hardware_dir,
    )
    submission_validation = validate_submission_package(submission_dir)
    if not submission_validation.get("ok"):
        raise RuntimeError("Submission package validation failed: " + "; ".join(submission_validation.get("errors", [])))

    registry_classes = set(str(value) for value in registry_validation.get("evidence_classes", []))
    present_physical = sorted(registry_classes & set(_REQUIRED_PHYSICAL_CLASSES))
    missing_physical = [value for value in _REQUIRED_PHYSICAL_CLASSES if value not in registry_classes]
    claims = [dict(row) for row in reviewer.get("claims", []) if isinstance(row, Mapping)]
    delta = {
        "schema_version": 1,
        "present_physical_classes": present_physical,
        "missing_physical_classes": missing_physical,
        "claims": claims,
        "scientific_boundary": (
            "Physical claim support is derived only from the validated qualified evidence registry. "
            "Packaging does not convert synthetic, surrogate, replay, or workstation evidence into physical evidence."
        ),
    }
    (out / "EVIDENCE_DELTA.json").write_text(json.dumps(delta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_delta_markdown(out / "EVIDENCE_DELTA.md", delta)

    full_readiness = build_readiness_report(
        root,
        profile="full",
        expected_version=version,
        reviewer_package=reviewer_dir,
        submission_package=submission_dir,
        evidence_registry=registry,
    )
    stable_version = bool(version) and not _PRERELEASE_PATTERN.search(version)
    blocker_rows: list[dict[str, Any]] = [
        {
            "gate": "qualified_registry",
            "status": "pass" if registry_validation.get("ok") else "fail",
            "required": True,
            "detail": "registry structure and all referenced artifact hashes validate",
        },
    ]
    for evidence_class in _REQUIRED_PHYSICAL_CLASSES:
        present = evidence_class in registry_classes
        blocker_rows.append(
            {
                "gate": f"evidence:{evidence_class}",
                "status": "pass" if present else "fail",
                "required": True,
                "detail": "qualified in registry" if present else "qualified evidence not supplied",
            }
        )
    blocker_rows.extend(
        [
            {
                "gate": "submission_package_integrity",
                "status": "pass" if submission_validation.get("ok") else "fail",
                "required": True,
                "detail": "; ".join(submission_validation.get("errors", [])) or "submission checksums and required files validate",
            },
            {
                "gate": "whole_system_full_readiness",
                "status": "pass" if full_readiness.get("ready") else "fail",
                "required": True,
                "detail": (
                    "full readiness satisfied"
                    if full_readiness.get("ready")
                    else "; ".join(item.get("name", "unknown") for item in full_readiness.get("required_failures", []))
                ),
            },
            {
                "gate": "stable_version",
                "status": "pass" if stable_version else "fail",
                "required": True,
                "detail": f"version={version}; prerelease versions cannot pass the stable-release decision",
            },
        ]
    )
    ready_for_stable_release = all(
        row["status"] == "pass" for row in blocker_rows if row["required"]
    )
    _write_csv(out / "BLOCKER_MATRIX.csv", ("gate", "status", "required", "detail"), blocker_rows)
    _write_blocker_markdown(out / "BLOCKER_MATRIX.md", blocker_rows)

    manifest = {
        "schema_version": 1,
        "version": version,
        "journal": journal,
        "evidence_registry": str(registry),
        "evidence_registry_sha256": sha256_file(registry),
        "reviewer_manifest_sha256": sha256_file(reviewer_dir / "manifest.json"),
        "submission_manifest_sha256": sha256_file(submission_dir / "submission_manifest.json"),
        "present_physical_classes": present_physical,
        "missing_physical_classes": missing_physical,
        "claim_gaps": submission.get("claim_gaps", []),
        "full_readiness": bool(full_readiness.get("ready")),
        "stable_version": stable_version,
        "ready_for_stable_release": ready_for_stable_release,
        "blockers": [dict(row) for row in blocker_rows if row["required"] and row["status"] == "fail"],
        "scientific_boundary": delta["scientific_boundary"],
    }
    (out / "FINALIZATION_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _package_checksums(out)
    return manifest


def validate_finalization_package(output_dir: str | Path) -> dict[str, Any]:
    root = Path(output_dir)
    required = (
        "FINALIZATION_MANIFEST.json",
        "EVIDENCE_DELTA.json",
        "EVIDENCE_DELTA.md",
        "BLOCKER_MATRIX.csv",
        "BLOCKER_MATRIX.md",
        "reviewer_package/manifest.json",
        "reviewer_package/checksums.sha256",
        "submission_package/submission_manifest.json",
        "submission_package/submission_checksums.sha256",
        "checksums.sha256",
    )
    errors = [f"Missing finalization file: {relative}" for relative in required if not (root / relative).is_file()]
    checksum_path = root / "checksums.sha256"
    if checksum_path.is_file():
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                errors.append(f"Malformed finalization checksum line {line_number}.")
                continue
            candidate = root / relative
            if not candidate.is_file():
                errors.append(f"Finalization checksum target missing: {relative}")
            elif sha256_file(candidate) != expected:
                errors.append(f"Finalization checksum mismatch: {relative}")
    manifest = _read_json(root / "FINALIZATION_MANIFEST.json") if (root / "FINALIZATION_MANIFEST.json").is_file() else {}
    return {
        "ok": not errors,
        "ready_for_stable_release": bool(manifest.get("ready_for_stable_release")),
        "missing_physical_classes": manifest.get("missing_physical_classes", []),
        "errors": errors,
    }
