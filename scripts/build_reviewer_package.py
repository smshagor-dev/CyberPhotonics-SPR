from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf import __version__
from sprpcf.evidence.qualification import validate_evidence_registry
from sprpcf.publication.evidence import EVIDENCE_CLASSES, EvidenceSource, build_reviewer_package


def _optional_source(
    sources: list[EvidenceSource],
    role: str,
    path: Path | None,
    evidence_class: str | None,
    label: str,
) -> None:
    if path is not None:
        sources.append(EvidenceSource(role=role, path=path, evidence_class=evidence_class, label=label))


def _registry_sources(sources: list[EvidenceSource], registry_path: Path | None) -> None:
    if registry_path is None:
        return
    report = validate_evidence_registry(registry_path, verify_files=True)
    if not report["ok"]:
        raise SystemExit("Evidence registry validation failed: " + "; ".join(report["errors"]))
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    _optional_source(
        sources,
        "qualified_evidence_registry",
        registry_path,
        "reproducibility",
        "Qualified physical-evidence registry",
    )
    for record in payload.get("records", []):
        if not isinstance(record, dict) or record.get("qualified") is not True:
            continue
        evidence_class = str(record.get("evidence_class") or "")
        record_id = str(record.get("record_id") or "unknown")[:12]
        label = str(record.get("label") or evidence_class.replace("_", " ").title())
        for artifact in record.get("artifacts", []):
            if not isinstance(artifact, dict):
                continue
            stored = Path(str(artifact.get("path") or ""))
            source_path = stored if stored.is_absolute() else (registry_path.resolve().parent / stored).resolve()
            if not source_path.is_file():
                continue
            artifact_role = str(artifact.get("role") or "artifact").replace("-", "_")
            role = f"qualified_{evidence_class}_{record_id}_{artifact_role}"
            sources.append(
                EvidenceSource(
                    role=role,
                    path=source_path,
                    evidence_class=evidence_class,
                    label=f"{label}: {artifact_role}",
                )
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a reviewer-facing, hash-bound evidence package without inventing missing evidence."
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="CyberPhotonics-SPR Reviewer Evidence Package")
    parser.add_argument("--version", default=__version__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))

    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--validation-class", choices=EVIDENCE_CLASSES, default="software_only")
    parser.add_argument("--ablation-dir", type=Path)
    parser.add_argument("--design-dir", type=Path)
    parser.add_argument("--closed-loop-dir", type=Path)
    parser.add_argument(
        "--closed-loop-class",
        choices=("auto", *EVIDENCE_CLASSES),
        default="auto",
        help="Use auto to read evidence_class/backend from iteration_manifest.json when available.",
    )
    parser.add_argument("--hardware-dir", type=Path)
    parser.add_argument(
        "--hardware-class",
        choices=EVIDENCE_CLASSES,
        default="software_only",
        help=(
            "Keep software_only for replay/synthetic runs. Use experimental_sensor or device_benchmark only "
            "for genuinely measured evidence."
        ),
    )
    parser.add_argument("--reproducibility-dir", type=Path)
    parser.add_argument("--release-validation", type=Path)
    parser.add_argument(
        "--evidence-registry",
        type=Path,
        help="Validated registry created by scripts/register_evidence.py; registered artifacts are packaged by class.",
    )
    parser.add_argument("--max-file-size-mib", type=float, default=25.0)
    parser.add_argument("--no-release-metadata", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_file_size_mib <= 0:
        raise SystemExit("--max-file-size-mib must be positive.")

    sources: list[EvidenceSource] = []
    _optional_source(
        sources,
        "validation",
        args.validation_dir,
        args.validation_class,
        "Scientific validation pack",
    )
    _optional_source(
        sources,
        "ablation",
        args.ablation_dir,
        "software_only",
        "Physics-loss ablation evidence",
    )
    _optional_source(
        sources,
        "design",
        args.design_dir,
        "surrogate_model",
        "Pareto/inverse-design evidence",
    )
    _optional_source(
        sources,
        "closed_loop",
        args.closed_loop_dir,
        None if args.closed_loop_class == "auto" else args.closed_loop_class,
        "Closed-loop physics verification",
    )
    _optional_source(
        sources,
        "hardware",
        args.hardware_dir,
        args.hardware_class,
        "Sensor/runtime evidence",
    )
    _optional_source(
        sources,
        "reproducibility",
        args.reproducibility_dir,
        "reproducibility",
        "Reproducibility/provenance bundle",
    )
    _optional_source(
        sources,
        "release_validation",
        args.release_validation,
        "release",
        "Release validation",
    )
    _registry_sources(sources, args.evidence_registry)

    manifest = build_reviewer_package(
        args.out,
        sources=sources,
        title=args.title,
        version=args.version,
        repo_root=args.repo_root,
        max_file_size_bytes=int(args.max_file_size_mib * 1024 * 1024),
        include_release_metadata=not args.no_release_metadata,
    )
    print(
        json.dumps(
            {
                "output": str(args.out),
                "version": manifest["version"],
                "evidence_classes": manifest["evidence_classes"],
                "artifact_count": len(manifest["artifacts"]),
                "claims": {row["claim"]: row["status"] for row in manifest["claims"]},
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
