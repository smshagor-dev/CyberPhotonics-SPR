from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf import __version__
from sprpcf.publication.submission import build_submission_package, validate_submission_package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a manuscript supplementary/submission package from an existing reviewer evidence package."
    )
    parser.add_argument("--reviewer-package", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="CyberPhotonics-SPR Manuscript Supplementary Package")
    parser.add_argument("--version", default=__version__)
    parser.add_argument("--journal", default=None)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--manuscript", type=Path, default=None)
    parser.add_argument("--validation-dir", type=Path, default=None)
    parser.add_argument("--ablation-dir", type=Path, default=None)
    parser.add_argument("--design-dir", type=Path, default=None)
    parser.add_argument("--closed-loop-dir", type=Path, default=None)
    parser.add_argument("--hardware-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = build_submission_package(
        args.out,
        reviewer_package_dir=args.reviewer_package,
        title=args.title,
        version=args.version,
        repo_root=args.repo_root,
        journal=args.journal,
        manuscript_file=args.manuscript,
        validation_dir=args.validation_dir,
        ablation_dir=args.ablation_dir,
        design_dir=args.design_dir,
        closed_loop_dir=args.closed_loop_dir,
        hardware_dir=args.hardware_dir,
    )
    validation = validate_submission_package(args.out)
    if not validation["ok"]:
        raise SystemExit(json.dumps(validation, indent=2))
    print(
        json.dumps(
            {
                "output": str(args.out),
                "version": manifest["version"],
                "evidence_classes": manifest["evidence_classes"],
                "readiness": manifest["readiness"],
                "claim_gaps": manifest["claim_gaps"],
                "figures": len(manifest["figures"]),
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
