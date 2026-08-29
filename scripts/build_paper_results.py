from __future__ import annotations

import argparse
import json
from pathlib import Path

from sprpcf.publication.results import build_paper_results_package, validate_paper_results_package


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evidence-backed manuscript Results package from qualified physical evidence."
    )
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--experimental-analysis-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--replace", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    manifest = build_paper_results_package(
        args.out,
        evidence_registry=args.evidence_registry,
        experimental_analysis_dir=args.experimental_analysis_dir,
        replace=args.replace,
    )
    validation = validate_paper_results_package(args.out)
    report = {"output": str(args.out), "manifest": manifest, "validation": validation}
    print(json.dumps(report, indent=2, sort_keys=True))
    if args.strict and not manifest["ready_for_manuscript_results"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
