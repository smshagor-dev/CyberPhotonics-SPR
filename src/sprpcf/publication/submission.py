from __future__ import annotations

import csv
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from sprpcf.utils.reproducibility import git_state, sha256_file

REQUIRED_SUBMISSION_FILES = (
    "README_FIRST.md",
    "SUPPLEMENTARY_INFORMATION.md",
    "MANUSCRIPT_CHECKLIST.md",
    "FIGURE_INDEX.csv",
    "TABLE_S1_VALIDATION_METRICS.csv",
    "TABLE_S1_VALIDATION_METRICS.md",
    "TABLE_S2_CLAIMS_TO_EVIDENCE.csv",
    "TABLE_S2_CLAIMS_TO_EVIDENCE.md",
    "TABLE_S3_ARTIFACT_PROVENANCE.csv",
    "submission_manifest.json",
    "submission_checksums.sha256",
    "reviewer_evidence/manifest.json",
    "reviewer_evidence/CLAIMS_MATRIX.md",
)

_FIGURE_SUFFIXES = {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
_MANUSCRIPT_SUFFIXES = {".pdf", ".docx", ".tex", ".md"}
_RELEASE_FILES = (
    "CITATION.cff",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "RELEASE_CANDIDATE.md",
    "docs/REPRODUCIBILITY.md",
    "docs/PUBLICATION_REVIEWER_PACKAGE.md",
    "docs/SUBMISSION_PACKAGE.md",
)


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _write_markdown_table(
    path: Path,
    title: str,
    columns: Sequence[tuple[str, str]],
    rows: Sequence[Mapping[str, Any]],
    note: str | None = None,
) -> None:
    headers = [label for _, label in columns]
    lines = [f"# {title}", ""]
    if note:
        lines.extend([note, ""])
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        lines.append("| " + " | ".join(_markdown_escape(row.get(key, "")) for key, _ in columns) + " |")
    if not rows:
        lines.append("| " + " | ".join("Not supplied" if index == 0 else "" for index in range(len(headers))) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _metric_unit(metric_path: str) -> str:
    name = metric_path.lower()
    if "sensitivity" in name:
        return "nm/RIU"
    if "fom" in name:
        return "1/RIU"
    if "lambda" in name or "wavelength" in name or "fwhm" in name:
        return "nm"
    if "latency" in name:
        return "ms"
    if "throughput" in name:
        return "frames/s"
    if "bytes" in name or "memory" in name or "rss" in name:
        return "bytes"
    if "r2" in name or "rate" in name or "confidence" in name or "coverage" in name:
        return "dimensionless"
    if any(token in name for token in ("rows", "count", "sweeps", "samples")):
        return "count"
    return "reported unit"


def _flatten_validation_metrics(payload: Mapping[str, Any], evidence_class: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, Mapping):
            for key in sorted(value):
                visit(value[key], f"{path}.{key}" if path else str(key))
            return
        if isinstance(value, bool):
            rows.append(
                {
                    "metric": path,
                    "value": "true" if value else "false",
                    "unit": "boolean",
                    "evidence_class": evidence_class,
                }
            )
            return
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            numeric = float(value)
            if math.isfinite(numeric):
                rendered = str(int(value)) if isinstance(value, int) else f"{numeric:.10g}"
                rows.append(
                    {
                        "metric": path,
                        "value": rendered,
                        "unit": _metric_unit(path),
                        "evidence_class": evidence_class,
                    }
                )

    visit(payload, "")
    return rows


def _role_class_map(reviewer_manifest: Mapping[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    sources = reviewer_manifest.get("sources", [])
    if not isinstance(sources, list):
        return result
    for row in sources:
        if not isinstance(row, Mapping):
            continue
        role = row.get("role")
        evidence_class = row.get("evidence_class")
        if isinstance(role, str) and isinstance(evidence_class, str):
            result[role] = evidence_class
    return result


def _humanize_stem(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


def _safe_stem(path: Path) -> str:
    text = "_".join(path.stem.replace("-", "_").split())
    clean = "".join(char for char in text if char.isalnum() or char == "_").strip("_")
    return clean or "figure"


def _copy_figures(
    output_dir: Path,
    sources: Sequence[tuple[str, Path | None]],
    role_classes: Mapping[str, str],
) -> list[dict[str, Any]]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for role, source_dir in sources:
        if source_dir is None or not source_dir.exists():
            continue
        paths = [source_dir] if source_dir.is_file() else sorted(source_dir.rglob("*"))
        for candidate in paths:
            if not candidate.is_file() or candidate.suffix.lower() not in _FIGURE_SUFFIXES:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            candidates.append((role, candidate))

    rows: list[dict[str, Any]] = []
    for index, (role, source) in enumerate(candidates, start=1):
        figure_id = f"Figure S{index:02d}"
        destination = figure_dir / f"Figure_S{index:02d}_{_safe_stem(source)}{source.suffix.lower()}"
        shutil.copy2(source, destination)
        evidence_class = role_classes.get(role, "software_only")
        rows.append(
            {
                "figure_id": figure_id,
                "role": role,
                "evidence_class": evidence_class,
                "source_name": source.name,
                "package_path": destination.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(destination),
                "caption": (
                    f"{figure_id}. {_humanize_stem(source)} from the {role.replace('_', ' ')} artifact set. "
                    f"Evidence class: {evidence_class}."
                ),
                "quality_note": "Copied without resampling; verify final dimensions/DPI against the target journal requirements.",
            }
        )
    return rows


def _copy_release_metadata(output_dir: Path, repo_root: Path) -> list[str]:
    destination_root = output_dir / "release_metadata"
    copied: list[str] = []
    for relative in _RELEASE_FILES:
        source = repo_root / relative
        if not source.is_file():
            continue
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied.append(destination.relative_to(output_dir).as_posix())
    return copied


def _write_readme(
    path: Path,
    *,
    title: str,
    version: str,
    journal: str | None,
    evidence_classes: Sequence[str],
    claim_gaps: Sequence[str],
) -> None:
    lines = [
        f"# {title}",
        "",
        f"Package version: `{version}`",
        f"Target journal/venue: `{journal or 'not specified'}`",
        "",
        "## Start here",
        "",
        "1. Read `SUPPLEMENTARY_INFORMATION.md` for the submission-oriented evidence map.",
        "2. Read `MANUSCRIPT_CHECKLIST.md` before uploading files to a journal portal.",
        "3. Inspect `TABLE_S2_CLAIMS_TO_EVIDENCE.csv` before using physical, experimental, or device-specific claims.",
        "4. Use `submission_checksums.sha256` to verify package integrity.",
        "5. The complete reviewer evidence package is mirrored under `reviewer_evidence/`.",
        "",
        "## Evidence classes present",
        "",
    ]
    lines.extend(f"- `{item}`" for item in evidence_classes) if evidence_classes else lines.append("- None supplied.")
    lines.extend(["", "## Claim-family gaps", ""])
    lines.extend(f"- {item}" for item in claim_gaps) if claim_gaps else lines.append("- No gaps reported by the claims matrix.")
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            (
                "This package organizes evidence; it does not create missing COMSOL, experimental, fabricated-sensor, "
                "or target-device results. Software-only and surrogate evidence must remain labeled as such."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_supplementary_information(
    path: Path,
    *,
    title: str,
    version: str,
    evidence_classes: Sequence[str],
    metric_count: int,
    figure_count: int,
    claim_gaps: Sequence[str],
) -> None:
    lines = [
        f"# Supplementary Information — {title}",
        "",
        f"Software/research package version: `{version}`",
        "",
        "## S1. Scope",
        "",
        (
            "This supplementary package is generated from the artifacts explicitly supplied to the submission builder. "
            "It is intended to make computational evidence, figures, tables, provenance, and claim boundaries easy to audit."
        ),
        "",
        "## S2. Evidence classes represented",
        "",
    ]
    lines.extend(f"- `{item}`" for item in evidence_classes) if evidence_classes else lines.append("- None supplied.")
    lines.extend(
        [
            "",
            "## S3. Publication tables",
            "",
            f"- `TABLE_S1_VALIDATION_METRICS.*` contains {metric_count} scalar validation metric row(s) extracted from supplied validation JSON.",
            "- `TABLE_S2_CLAIMS_TO_EVIDENCE.*` mirrors the reviewer claims-to-evidence status.",
            "- `TABLE_S3_ARTIFACT_PROVENANCE.csv` maps packaged evidence artifacts to their SHA-256 digests.",
            "",
            "## S4. Supplementary figures",
            "",
            f"The package contains {figure_count} figure file(s), indexed in `FIGURE_INDEX.csv` and copied under `figures/` without resampling.",
            "",
            "## S5. Reproducibility and provenance",
            "",
            (
                "The mirrored `reviewer_evidence/` directory retains the reviewer guide, evidence classes, artifact index, "
                "manifest, and checksums. Repository citation/reproducibility metadata is copied under `release_metadata/`."
            ),
            "",
            "## S6. Claim-family gaps",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in claim_gaps) if claim_gaps else lines.append("- None reported by the supplied claims matrix.")
    lines.extend(
        [
            "",
            "## S7. Interpretation boundary",
            "",
            (
                "Synthetic spectra and software-only validation demonstrate computational behavior, not measured sensor performance. "
                "Surrogate predictions require independent physics verification. COMSOL output is numerical evidence, not laboratory measurement. "
                "Experimental and device-specific claims require genuinely measured artifacts from the reported protocol/device."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_manuscript_checklist(path: Path, *, version: str) -> None:
    lines = [
        "# Manuscript Submission Checklist",
        "",
        f"Evidence package version: `{version}`",
        "",
        "- [ ] Manuscript title, authors, affiliations, and corresponding-author details are final.",
        "- [ ] Abstract, keywords, methods, results, and conclusions match the supplied evidence.",
        "- [ ] Every table and figure is cited in the manuscript and uses the journal-required format/resolution.",
        "- [ ] Synthetic/software-only values are not described as COMSOL, experimental, or fabricated-sensor measurements.",
        "- [ ] Surrogate/inverse-design results are not presented as independent physical verification.",
        "- [ ] COMSOL claims, if any, are backed by the exact model/configuration/units/mesh/boundary evidence.",
        "- [ ] Experimental claims, if any, include measurement protocol, calibration provenance, and raw/processed data traceability.",
        "- [ ] Device latency/memory/throughput claims, if any, were measured on the exact named device.",
        "- [ ] Data/code availability wording points to real accessible artifacts; no DOI is claimed until actually minted.",
        "- [ ] Funding, competing-interest, ethics, author-contribution, and AI-assistance statements follow the target journal policy.",
        "- [ ] `TABLE_S2_CLAIMS_TO_EVIDENCE.csv` agrees with manuscript wording.",
        "- [ ] Git commit, software version, seeds, dataset/model/configuration hashes, and environment records are retained.",
        "- [ ] `submission_checksums.sha256` verifies before final upload/archive.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_submission_package(
    output_dir: str | Path,
    *,
    reviewer_package_dir: str | Path,
    title: str = "CyberPhotonics-SPR Manuscript Supplementary Package",
    version: str,
    repo_root: str | Path = ".",
    journal: str | None = None,
    manuscript_file: str | Path | None = None,
    validation_dir: str | Path | None = None,
    ablation_dir: str | Path | None = None,
    design_dir: str | Path | None = None,
    closed_loop_dir: str | Path | None = None,
    hardware_dir: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir)
    reviewer = Path(reviewer_package_dir)
    root = Path(repo_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty or absent: {out}")
    if not reviewer.is_dir():
        raise FileNotFoundError(f"Reviewer package directory not found: {reviewer}")
    reviewer_manifest_path = reviewer / "manifest.json"
    if not reviewer_manifest_path.is_file():
        raise FileNotFoundError(f"Reviewer manifest not found: {reviewer_manifest_path}")

    reviewer_manifest = _read_json_object(reviewer_manifest_path)
    claims = reviewer_manifest.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("Reviewer manifest claims must be a list.")
    artifacts = reviewer_manifest.get("artifacts", [])
    if not isinstance(artifacts, list):
        raise ValueError("Reviewer manifest artifacts must be a list.")
    evidence_classes = sorted(
        item for item in reviewer_manifest.get("evidence_classes", []) if isinstance(item, str)
    )
    claim_gaps = [
        str(row.get("claim"))
        for row in claims
        if isinstance(row, Mapping) and row.get("status") == "not supplied"
    ]
    role_classes = _role_class_map(reviewer_manifest)

    out.mkdir(parents=True, exist_ok=True)
    shutil.copytree(reviewer, out / "reviewer_evidence")
    release_metadata = _copy_release_metadata(out, root)

    manuscript_entry: dict[str, Any] | None = None
    if manuscript_file is not None:
        source = Path(manuscript_file)
        if not source.is_file():
            raise FileNotFoundError(f"Manuscript file not found: {source}")
        if source.suffix.lower() not in _MANUSCRIPT_SUFFIXES:
            raise ValueError(f"Unsupported manuscript suffix: {source.suffix}")
        destination = out / "manuscript" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        manuscript_entry = {
            "package_path": destination.relative_to(out).as_posix(),
            "sha256": sha256_file(destination),
            "size_bytes": destination.stat().st_size,
        }

    validation_path = Path(validation_dir) if validation_dir is not None else None
    validation_class = role_classes.get("validation", "software_only")
    metric_rows: list[dict[str, str]] = []
    if validation_path is not None and (validation_path / "summary.json").is_file():
        metric_rows = _flatten_validation_metrics(
            _read_json_object(validation_path / "summary.json"),
            validation_class,
        )

    _write_csv(
        out / "TABLE_S1_VALIDATION_METRICS.csv",
        ("metric", "value", "unit", "evidence_class"),
        metric_rows,
    )
    _write_markdown_table(
        out / "TABLE_S1_VALIDATION_METRICS.md",
        "Table S1 — Validation Metrics",
        (
            ("metric", "Metric"),
            ("value", "Value"),
            ("unit", "Unit"),
            ("evidence_class", "Evidence class"),
        ),
        metric_rows,
        note="Values are extracted verbatim from supplied scalar validation-summary fields; the evidence class controls interpretation.",
    )

    claim_rows = [
        {
            "claim": str(row.get("claim", "")),
            "status": str(row.get("status", "")),
            "evidence_classes": str(row.get("evidence_classes", "")),
            "interpretation": str(row.get("interpretation", "")),
        }
        for row in claims
        if isinstance(row, Mapping)
    ]
    _write_csv(
        out / "TABLE_S2_CLAIMS_TO_EVIDENCE.csv",
        ("claim", "status", "evidence_classes", "interpretation"),
        claim_rows,
    )
    _write_markdown_table(
        out / "TABLE_S2_CLAIMS_TO_EVIDENCE.md",
        "Table S2 — Claims to Evidence",
        (
            ("claim", "Claim"),
            ("status", "Status"),
            ("evidence_classes", "Evidence class(es)"),
            ("interpretation", "Interpretation"),
        ),
        claim_rows,
        note="`not supplied` means the package does not contain the evidence class required for that claim family.",
    )

    provenance_rows = [
        {
            "role": str(row.get("role", "")),
            "evidence_class": str(row.get("evidence_class", "")),
            "package_path": str(row.get("package_path", "")),
            "size_bytes": row.get("size_bytes", ""),
            "sha256": str(row.get("sha256", "")),
        }
        for row in artifacts
        if isinstance(row, Mapping)
    ]
    _write_csv(
        out / "TABLE_S3_ARTIFACT_PROVENANCE.csv",
        ("role", "evidence_class", "package_path", "size_bytes", "sha256"),
        provenance_rows,
    )

    figure_rows = _copy_figures(
        out,
        (
            ("validation", validation_path),
            ("ablation", Path(ablation_dir) if ablation_dir is not None else None),
            ("design", Path(design_dir) if design_dir is not None else None),
            ("closed_loop", Path(closed_loop_dir) if closed_loop_dir is not None else None),
            ("hardware", Path(hardware_dir) if hardware_dir is not None else None),
        ),
        role_classes,
    )
    _write_csv(
        out / "FIGURE_INDEX.csv",
        ("figure_id", "role", "evidence_class", "source_name", "package_path", "sha256", "caption", "quality_note"),
        figure_rows,
    )

    _write_readme(
        out / "README_FIRST.md",
        title=title,
        version=version,
        journal=journal,
        evidence_classes=evidence_classes,
        claim_gaps=claim_gaps,
    )
    _write_supplementary_information(
        out / "SUPPLEMENTARY_INFORMATION.md",
        title=title,
        version=version,
        evidence_classes=evidence_classes,
        metric_count=len(metric_rows),
        figure_count=len(figure_rows),
        claim_gaps=claim_gaps,
    )
    _write_manuscript_checklist(out / "MANUSCRIPT_CHECKLIST.md", version=version)

    readiness = {
        "computational_reproducibility_evidence": "reproducibility" in evidence_classes,
        "software_validation_evidence": "software_only" in evidence_classes,
        "surrogate_model_evidence": "surrogate_model" in evidence_classes,
        "numerical_physics_evidence": "comsol_physics" in evidence_classes,
        "experimental_sensor_evidence": "experimental_sensor" in evidence_classes,
        "target_device_benchmark_evidence": "device_benchmark" in evidence_classes,
    }
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "title": title,
        "version": version,
        "journal": journal,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_state(root),
        "evidence_classes": evidence_classes,
        "claim_gaps": claim_gaps,
        "readiness": readiness,
        "tables": {
            "validation_metrics_rows": len(metric_rows),
            "claims_rows": len(claim_rows),
            "artifact_provenance_rows": len(provenance_rows),
        },
        "figures": figure_rows,
        "release_metadata": release_metadata,
        "manuscript": manuscript_entry,
        "reviewer_manifest_sha256": sha256_file(out / "reviewer_evidence" / "manifest.json"),
        "scientific_boundary": (
            "The submission package does not upgrade software-only, synthetic, surrogate, replay, or workstation evidence "
            "into COMSOL, experimental-sensor, fabricated-device, or target-device evidence."
        ),
    }
    (out / "submission_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    checksum_path = out / "submission_checksums.sha256"
    checksum_targets = sorted(
        candidate
        for candidate in out.rglob("*")
        if candidate.is_file() and candidate != checksum_path
    )
    checksum_lines = [
        f"{sha256_file(candidate)}  {candidate.relative_to(out).as_posix()}"
        for candidate in checksum_targets
    ]
    checksum_path.write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest


def validate_submission_package(package_dir: str | Path) -> dict[str, Any]:
    root = Path(package_dir)
    errors: list[str] = []
    warnings: list[str] = []
    for relative in REQUIRED_SUBMISSION_FILES:
        if not (root / relative).is_file():
            errors.append(f"Missing submission file: {relative}")

    manifest: dict[str, Any] = {}
    manifest_path = root / "submission_manifest.json"
    if manifest_path.is_file():
        try:
            manifest = _read_json_object(manifest_path)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            errors.append(f"Invalid submission manifest: {exc}")

    checksum_path = root / "submission_checksums.sha256"
    if checksum_path.is_file():
        for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                expected, relative = line.split("  ", 1)
            except ValueError:
                errors.append(f"Malformed checksum line {line_number}.")
                continue
            candidate = root / relative
            if not candidate.is_file():
                errors.append(f"Checksum target missing: {relative}")
                continue
            actual = sha256_file(candidate)
            if actual != expected:
                errors.append(f"Checksum mismatch: {relative}")
    elif root.exists():
        errors.append("Missing submission_checksums.sha256")

    evidence_classes = manifest.get("evidence_classes", []) if manifest else []
    if isinstance(evidence_classes, list):
        if "comsol_physics" not in evidence_classes:
            warnings.append("COMSOL physics evidence is not supplied in this submission package.")
        if "experimental_sensor" not in evidence_classes:
            warnings.append("Experimental sensor evidence is not supplied in this submission package.")
    return {
        "ok": not errors,
        "version": manifest.get("version") if manifest else None,
        "evidence_classes": evidence_classes if isinstance(evidence_classes, list) else [],
        "errors": errors,
        "warnings": warnings,
    }
