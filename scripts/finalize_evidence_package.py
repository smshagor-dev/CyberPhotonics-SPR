from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf import __version__
from sprpcf.publication.finalization import (
    build_evidence_finalization_package,
    validate_finalization_package,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build reviewer/submission finalization artifacts, claim/evidence delta, blocker matrix, "
            "and stable-release decision from a qualified evidence registry."
        )
    )
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--version", default=__version__)
    parser.add_argument("--journal")
    parser.add_argument("--manuscript", type=Path)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument("--ablation-dir", type=Path)
    parser.add_argument("--design-dir", type=Path)
    parser.add_argument("--closed-loop-dir", type=Path)
    parser.add_argument("--hardware-dir", type=Path)
    parser.add_argument("--reproducibility-dir", type=Path)
    parser.add_argument("--release-validation", type=Path)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace only an existing directory previously created by this finalization pipeline.",
    )
    parser.add_argument(
        "--strict-stable",
        action="store_true",
        help="Exit non-zero unless the qualified evidence, full readiness, and stable-version gates all pass.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_evidence_finalization_package(
        args.out,
        evidence_registry=args.evidence_registry,
        repo_root=args.repo_root,
        version=args.version,
        journal=args.journal,
        manuscript=args.manuscript,
        validation_dir=args.validation_dir,
        ablation_dir=args.ablation_dir,
        design_dir=args.design_dir,
        closed_loop_dir=args.closed_loop_dir,
        hardware_dir=args.hardware_dir,
        reproducibility_dir=args.reproducibility_dir,
        release_validation=args.release_validation,
        replace=args.replace,
    )
    validation = validate_finalization_package(args.out)
    if not validation["ok"]:
        raise SystemExit(json.dumps(validation, indent=2, sort_keys=True))

    result = {
        "output": str(args.out),
        "version": manifest["version"],
        "present_physical_classes": manifest["present_physical_classes"],
        "missing_physical_classes": manifest["missing_physical_classes"],
        "claim_gaps": manifest["claim_gaps"],
        "full_readiness": manifest["full_readiness"],
        "stable_version": manifest["stable_version"],
        "ready_for_stable_release": manifest["ready_for_stable_release"],
        "blockers": manifest["blockers"],
        "validation": validation,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.strict_stable and not manifest["ready_for_stable_release"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
