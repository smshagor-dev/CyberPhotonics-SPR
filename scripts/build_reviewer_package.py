from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf import __version__
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
                "claims": {
                    row["claim"]: row["status"]
                    for row in manifest["claims"]
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
