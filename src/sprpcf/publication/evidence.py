from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from sprpcf.utils.reproducibility import git_state, sha256_file

EVIDENCE_CLASSES = (
    "software_only",
    "surrogate_model",
    "comsol_physics",
    "experimental_sensor",
    "device_benchmark",
    "reproducibility",
    "release",
)

_ALLOWED_SUFFIXES = {
    ".csv",
    ".html",
    ".jpeg",
    ".jpg",
    ".json",
    ".md",
    ".parquet",
    ".pdf",
    ".png",
    ".svg",
    ".txt",
    ".yaml",
    ".yml",
}
_ALLOWED_NAMES = {"checksums.sha256", "environment.lock.txt"}
_RELEASE_METADATA_FILES = (
    "CITATION.cff",
    "MODEL_CARD.md",
    "DATASET_CARD.md",
    "LICENSE",
    "pyproject.toml",
    "docs/REPRODUCIBILITY.md",
    "docs/PUBLICATION_REVIEWER_PACKAGE.md",
)

_CLASS_DESCRIPTIONS = {
    "software_only": "Software, synthetic-data, replay, or pipeline evidence; not physical sensor validation.",
    "surrogate_model": "Learned-model or inverse-design evidence; requires independent physics/experimental verification.",
    "comsol_physics": "Numerical-physics evidence from a COMSOL-backed workflow and its recorded configuration/provenance.",
    "experimental_sensor": "Measured sensor evidence. Valid only when the supplied artifacts originate from a documented experiment.",
    "device_benchmark": "Latency/memory/throughput measured on the named target device, not inferred from another platform.",
    "reproducibility": "Seeds, environment, Git state, configuration, and hashes supporting computational reproducibility.",
    "release": "Release metadata, citation, licensing, validation, and packaging evidence.",
}


@dataclass(frozen=True)
class EvidenceSource:
    role: str
    path: Path
    evidence_class: str | None = None
    label: str | None = None

    def validate(self) -> None:
        if not self.role or any(char in self.role for char in "/\\"):
            raise ValueError("Evidence role must be a non-empty path-safe label without slashes.")
        if not self.path.exists():
            raise FileNotFoundError(f"Evidence source does not exist: {self.path}")
        if self.evidence_class is not None and self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(
                f"Unsupported evidence class {self.evidence_class!r}; expected one of {EVIDENCE_CLASSES}."
            )


def _read_json_object(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _normalize_evidence_class(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    aliases = {
        "software": "software_only",
        "synthetic": "software_only",
        "calibration_data_dependent": "software_only",
        "comsol": "comsol_physics",
        "experimental": "experimental_sensor",
        "hardware": "experimental_sensor",
        "benchmark": "device_benchmark",
    }
    normalized = aliases.get(value.strip().lower(), value.strip().lower())
    return normalized if normalized in EVIDENCE_CLASSES else None


def _infer_evidence_class(source: EvidenceSource) -> str:
    if source.evidence_class is not None:
        return source.evidence_class

    candidates: list[Path] = []
    if source.path.is_file():
        candidates.append(source.path)
    else:
        preferred = (
            "iteration_manifest.json",
            "manifest.json",
            "provenance.json",
            "summary.json",
        )
        for name in preferred:
            candidate = source.path / name
            if candidate.is_file():
                candidates.append(candidate)
        candidates.extend(sorted(source.path.glob("*.meta.json"))[:5])

    for candidate in candidates:
        payload = _read_json_object(candidate)
        if not payload:
            continue
        direct = _normalize_evidence_class(payload.get("evidence_class"))
        if direct:
            return direct
        backend = str(payload.get("backend", "")).strip().lower()
        if backend == "comsol":
            return "comsol_physics"
        source_name = str(payload.get("source", "")).strip().lower()
        if source_name == "comsol":
            return "comsol_physics"

    defaults = {
        "validation": "software_only",
        "ablation": "software_only",
        "design": "surrogate_model",
        "multiobjective": "surrogate_model",
        "closed_loop": "software_only",
        "hardware": "software_only",
        "reproducibility": "reproducibility",
        "release_validation": "release",
    }
    return defaults.get(source.role, "software_only")


def _iter_packable_files(path: Path, max_file_size_bytes: int) -> Iterable[tuple[Path, Path]]:
    if path.is_file():
        candidates = [(path, Path(path.name))]
        sidecar = path.with_suffix(path.suffix + ".meta.json")
        if sidecar.is_file():
            candidates.append((sidecar, Path(sidecar.name)))
    else:
        candidates = [
            (candidate, candidate.relative_to(path))
            for candidate in sorted(path.rglob("*"))
            if candidate.is_file()
        ]

    for candidate, relative in candidates:
        if any(part.startswith(".") for part in relative.parts):
            continue
        if candidate.name not in _ALLOWED_NAMES and candidate.suffix.lower() not in _ALLOWED_SUFFIXES:
            continue
        if candidate.stat().st_size > max_file_size_bytes:
            continue
        yield candidate, relative


def _copy_entry(
    source: Path,
    destination: Path,
    *,
    role: str,
    evidence_class: str,
    source_root: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    try:
        source_display = source.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        source_display = source.name
    return {
        "role": role,
        "evidence_class": evidence_class,
        "source_path": source_display,
        "package_path": destination.relative_to(bundle_root).as_posix(),
        "size_bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }


def _claims_matrix(classes: set[str]) -> list[dict[str, str]]:
    rules = [
        (
            "Computational pipeline and reporting are reproducible",
            {"reproducibility"},
            "Supported when provenance/environment/hash artifacts are supplied.",
        ),
        (
            "Synthetic/software validation pipeline executes as reported",
            {"software_only"},
            "Supports software and methodology validation only.",
        ),
        (
            "Surrogate or inverse-design model performance is documented",
            {"surrogate_model"},
            "Model/data-dependent evidence; independent physics validation is still required.",
        ),
        (
            "Numerical PCF-SPR physics performance is independently verified",
            {"comsol_physics"},
            "Requires COMSOL-backed evidence and independently checked model/configuration/units.",
        ),
        (
            "Experimental sensor performance is measured",
            {"experimental_sensor"},
            "Requires measured spectra and an experimental protocol; simulation is insufficient.",
        ),
        (
            "Target-device latency/memory/throughput is measured",
            {"device_benchmark"},
            "Requires measurements on the exact named deployment device.",
        ),
    ]
    rows: list[dict[str, str]] = []
    for claim, required, interpretation in rules:
        matched = sorted(required & classes)
        rows.append(
            {
                "claim": claim,
                "status": "supported" if matched else "not supplied",
                "evidence_classes": ", ".join(matched),
                "interpretation": interpretation,
            }
        )
    rows.append(
        {
            "claim": "Confidence/ranking score is a calibrated probability that a fabricated sensor will succeed",
            "status": "not claimed",
            "evidence_classes": "",
            "interpretation": (
                "The project treats confidence/OOD values as ranking or calibration aids, not fabricated-device success probabilities."
            ),
        }
    )
    return rows


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)


def _write_claims_markdown(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    lines = [
        "# Claims-to-Evidence Matrix",
        "",
        "This matrix is generated from the evidence classes actually supplied to the package.",
        "",
        "| Claim | Status | Evidence class(es) | Interpretation |",
        "|---|---|---|---|",
    ]
    for row in rows:
        values = [
            str(row["claim"]).replace("|", "\\|"),
            str(row["status"]).replace("|", "\\|"),
            str(row["evidence_classes"]).replace("|", "\\|"),
            str(row["interpretation"]).replace("|", "\\|"),
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reviewer_guide(
    path: Path,
    *,
    title: str,
    claims: Sequence[Mapping[str, str]],
    source_rows: Sequence[Mapping[str, str]],
) -> None:
    supported = [row["claim"] for row in claims if row["status"] == "supported"]
    missing = [row["claim"] for row in claims if row["status"] == "not supplied"]
    lines = [
        f"# {title}",
        "",
        "## Reviewer quick start",
        "",
        "1. Read `CLAIMS_MATRIX.md` to see which claim families are supported by supplied evidence.",
        "2. Read `artifact_index.csv` or `manifest.json` to locate each evidence artifact and verify its SHA-256.",
        "3. Use `checksums.sha256` to detect any package modification.",
        "4. Treat evidence according to its class; synthetic/surrogate outputs are not experimental measurements.",
        "",
        "## Evidence classes in this package",
        "",
    ]
    classes = sorted({str(row["evidence_class"]) for row in source_rows})
    for evidence_class in classes:
        lines.append(f"- **{evidence_class}** — {_CLASS_DESCRIPTIONS[evidence_class]}")
    lines.extend(["", "## Supplied evidence sources", ""])
    for row in source_rows:
        lines.append(
            f"- `{row['role']}` — {row['label']} — class `{row['evidence_class']}` — {row['file_count']} packaged file(s)"
        )
    lines.extend(["", "## Claim families currently supported", ""])
    if supported:
        lines.extend(f"- {claim}" for claim in supported)
    else:
        lines.append("- None; this package contains metadata only.")
    lines.extend(["", "## Evidence not supplied", ""])
    if missing:
        lines.extend(f"- {claim}" for claim in missing)
    else:
        lines.append("- No claim-family gaps detected by the package rules.")
    lines.extend(
        [
            "",
            "## Scientific boundary",
            "",
            (
                "Absence from the missing-evidence list does not by itself prove scientific correctness. "
                "Reviewers must still inspect experimental protocols, COMSOL model setup, units, mesh/boundary conditions, "
                "dataset construction, statistical assumptions, and the underlying source artifacts."
            ),
            "",
            (
                "This package never upgrades synthetic, replay, surrogate, or workstation evidence into COMSOL, experimental, "
                "or target-device evidence automatically."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_release_notes(
    path: Path,
    *,
    title: str,
    version: str | None,
    source_rows: Sequence[Mapping[str, str]],
    claims: Sequence[Mapping[str, str]],
) -> None:
    supported = [row["claim"] for row in claims if row["status"] == "supported"]
    version_text = version or "unversioned build"
    lines = [
        f"# {title} — {version_text}",
        "",
        "## Evidence package contents",
        "",
    ]
    for row in source_rows:
        lines.append(
            f"- {row['label']} (`{row['role']}`, `{row['evidence_class']}`): {row['file_count']} file(s)"
        )
    lines.extend(["", "## Supported claim families", ""])
    lines.extend(f"- {claim}" for claim in supported) if supported else lines.append("- None.")
    lines.extend(
        [
            "",
            "## Release interpretation",
            "",
            (
                "The archive is reviewer-facing evidence and provenance, not a substitute for peer review. "
                "Physical and experimental claims remain limited to the evidence classes actually present."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_reviewer_package(
    output_dir: str | Path,
    *,
    sources: Iterable[EvidenceSource],
    title: str = "CyberPhotonics-SPR Reviewer Evidence Package",
    version: str | None = None,
    repo_root: str | Path = ".",
    max_file_size_bytes: int = 25 * 1024 * 1024,
    include_release_metadata: bool = True,
) -> dict[str, Any]:
    if max_file_size_bytes < 1:
        raise ValueError("max_file_size_bytes must be positive.")

    out = Path(output_dir)
    root = Path(repo_root)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory must be empty or absent: {out}")
    out.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    source_rows: list[dict[str, str]] = []

    for source in sources:
        source.validate()
        evidence_class = _infer_evidence_class(source)
        copied = 0
        for candidate, relative in _iter_packable_files(source.path, max_file_size_bytes):
            destination = out / "evidence" / source.role / relative
            entries.append(
                _copy_entry(
                    candidate,
                    destination,
                    role=source.role,
                    evidence_class=evidence_class,
                    source_root=root,
                    bundle_root=out,
                )
            )
            copied += 1
        source_rows.append(
            {
                "role": source.role,
                "label": source.label or source.role.replace("_", " ").title(),
                "evidence_class": evidence_class,
                "file_count": str(copied),
            }
        )

    if include_release_metadata:
        copied = 0
        for relative_name in _RELEASE_METADATA_FILES:
            candidate = root / relative_name
            if not candidate.is_file():
                continue
            destination = out / "release" / relative_name
            entries.append(
                _copy_entry(
                    candidate,
                    destination,
                    role="release_metadata",
                    evidence_class="release",
                    source_root=root,
                    bundle_root=out,
                )
            )
            copied += 1
        source_rows.append(
            {
                "role": "release_metadata",
                "label": "Repository release metadata",
                "evidence_class": "release",
                "file_count": str(copied),
            }
        )

    classes = {str(row["evidence_class"]) for row in source_rows if int(row["file_count"]) > 0}
    claims = _claims_matrix(classes)
    created_utc = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": "1.0",
        "title": title,
        "version": version,
        "created_utc": created_utc,
        "git": git_state(root),
        "evidence_classes": sorted(classes),
        "sources": source_rows,
        "artifacts": sorted(entries, key=lambda row: row["package_path"]),
        "claims": claims,
        "scientific_boundary": (
            "Synthetic, surrogate, replay, and workstation outputs are not automatically promoted to COMSOL, "
            "experimental-sensor, or target-device evidence."
        ),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    artifact_rows = [
        {
            "role": row["role"],
            "evidence_class": row["evidence_class"],
            "package_path": row["package_path"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
        }
        for row in manifest["artifacts"]
    ]
    _write_csv(
        out / "artifact_index.csv",
        ("role", "evidence_class", "package_path", "size_bytes", "sha256"),
        artifact_rows,
    )
    _write_csv(
        out / "CLAIMS_MATRIX.csv",
        ("claim", "status", "evidence_classes", "interpretation"),
        claims,
    )
    _write_claims_markdown(out / "CLAIMS_MATRIX.md", claims)
    _write_reviewer_guide(
        out / "REVIEWER_GUIDE.md",
        title=title,
        claims=claims,
        source_rows=source_rows,
    )
    _write_release_notes(
        out / "RELEASE_NOTES.md",
        title=title,
        version=version,
        source_rows=source_rows,
        claims=claims,
    )

    generated = [
        out / "manifest.json",
        out / "artifact_index.csv",
        out / "CLAIMS_MATRIX.csv",
        out / "CLAIMS_MATRIX.md",
        out / "REVIEWER_GUIDE.md",
        out / "RELEASE_NOTES.md",
    ]
    checksum_targets = [out / row["package_path"] for row in manifest["artifacts"]] + generated
    checksum_lines = [
        f"{sha256_file(path)}  {path.relative_to(out).as_posix()}"
        for path in sorted(checksum_targets, key=lambda item: item.relative_to(out).as_posix())
    ]
    (out / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    return manifest
