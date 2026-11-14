"""Build and verify the Stage 6 journal-claim evidence draft."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.publication.claim_trace import (
    build_claim_traceable_package,
    validate_claim_traceable_package,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draft claim-level journal results with strict frozen-source provenance."
    )
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Immutable Stage 4 output directory")
    parser.add_argument("--stage5", type=Path,
                        help="Recorded Stage 5 audit directory; optional for blocker-only report")
    parser.add_argument("--solver-archive", type=Path,
                        help="Authentic external COMSOL source archive for independent re-audit")
    parser.add_argument("--refinement-archive", type=Path,
                        help="Archived real COMSOL mesh and wavelength refinements")
    parser.add_argument("--evidence-registry", type=Path,
                        help="Hash-validated source registry; not a physical authenticity certificate")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--strict-numerical-candidates", action="store_true",
                        help="Exit 2 if no independently recomputed numerical candidates were available")
    args = parser.parse_args()
    report = build_claim_traceable_package(
        args.benchmark, args.out, stage5_dir=args.stage5,
        solver_archive=args.solver_archive, refinement_archive=args.refinement_archive,
        evidence_registry=args.evidence_registry,
        bootstrap_resamples=args.bootstrap_resamples, repo_root=args.repo_root,
    )
    verification = validate_claim_traceable_package(args.out)
    if not verification["ok"]:
        raise SystemExit("Package integrity check failed: " + "; ".join(verification["errors"]))
    print(json.dumps({
        "output": str(args.out),
        "integrity_ok": verification["ok"],
        "scientific_submission_ready": report["scientific_submission_ready"],
        "numerical_candidates_available": report["numerical_candidates_available"],
        "candidate_number_count": report["candidate_number_count"],
        "blockers": report["blockers"],
    }, indent=2, sort_keys=True))
    if args.strict_numerical_candidates and not report["numerical_candidates_available"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
