"""Fail-closed comparison of frozen inverse-design requests with archived COMSOL spectra.

Never declares source authenticity or independent electromagnetic correctness.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.validation.independent_comsol import audit_independent_comsol_comparison


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile each Stage 4 selected design with external fixed-geometry COMSOL sweeps."
    )
    parser.add_argument("--benchmark", type=Path, required=True,
                        help="Frozen Stage 4 output directory (manifest and requests.csv)")
    parser.add_argument("--solver-archive", type=Path, required=True,
                        help="Read-only actual COMSOL model/config/log/spectra archive")
    parser.add_argument("--refinement-archive", type=Path, required=True,
                        help="Independent mesh and spectral convergence archive")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spectrum-tolerance-nm", type=float, default=0.05)
    parser.add_argument("--bootstrap-resamples", type=int, default=500)
    parser.add_argument("--strict", action="store_true",
                        help="Exit 2 unless all recorded solver sweeps and convergence checks are consistent")
    args = parser.parse_args()
    report = audit_independent_comsol_comparison(
        args.benchmark, args.solver_archive, args.refinement_archive, args.out,
        spectrum_tolerance_nm=args.spectrum_tolerance_nm,
        bootstrap_resamples=args.bootstrap_resamples,
    )
    print(json.dumps({
        "numerical_archive_consistent": report["numerical_archive_consistent"],
        "all_requested_sweeps_successful": report["all_requested_sweeps_successful"],
        "source_authenticity_verified": report["source_authenticity_verified"],
        "scientific_submission_ready": report["scientific_submission_ready"],
        "counts": report["counts"], "errors": report["errors"],
        "report": str(args.out / "PHYSICS_VERIFICATION_AUDIT.json"),
    }, indent=2, sort_keys=True))
    if args.strict and not (
        report["numerical_archive_consistent"] and report["all_requested_sweeps_successful"]
    ):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
