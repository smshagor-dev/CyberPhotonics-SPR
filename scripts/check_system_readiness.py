from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

from sprpcf.utils.readiness import build_readiness_report, readiness_markdown


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit CyberPhotonics-SPR whole-system readiness.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--profile", choices=("software", "release", "full"), default="release")
    parser.add_argument("--expected-version")
    parser.add_argument("--reviewer-package", type=Path)
    parser.add_argument("--submission-package", type=Path)
    parser.add_argument("--evidence-registry", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when a required readiness check fails.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = build_readiness_report(
        args.repo_root,
        profile=args.profile,
        expected_version=args.expected_version,
        reviewer_package=args.reviewer_package,
        submission_package=args.submission_package,
        evidence_registry=args.evidence_registry,
    )

    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(readiness_markdown(report), encoding="utf-8")

    if args.strict and not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
