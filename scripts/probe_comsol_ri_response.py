"""Run a source-locked COMSOL RI response triad plus repeated-center sanity check."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sprpcf.simulation.ri_response_gate import probe_comsol_ri_response


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check frozen low/center/high RI and repeat-center COMSOL outputs before a full campaign."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-readiness-report", type=Path, required=True,
                        help="Successful Stage10 single-center real COMSOL pilot JSON")
    parser.add_argument("--min-shift-nm", type=float, required=True,
                        help="Preregistered minimum resolvable low-to-high RI shift in nm")
    parser.add_argument("--max-repeat-nm", type=float, required=True,
                        help="Preregistered maximum repeated-center resonance/FWHM change in nm")
    parser.add_argument("--min-diagnostic-r2", type=float, default=0.90,
                        help="Diagnostic three-point trend R2 (not publication linearity)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true",
                        help="Exit status 2 if no source-bound real COMSOL RI diagnostic passes")
    args = parser.parse_args()
    result = probe_comsol_ri_response(
        args.plan, args.benchmark, args.model, args.config,
        args.pilot_readiness_report, args.out,
        min_shift_nm=args.min_shift_nm, max_repeat_nm=args.max_repeat_nm,
        min_diagnostic_r2=args.min_diagnostic_r2,
    )
    print(json.dumps({
        "result_dir": str(args.out),
        "observed_backend": result["observed_backend"],
        "solver_called": result["solver_called"],
        "numerical_ri_response_consistent": result["numerical_ri_response_consistent"],
        "ready_for_full_solver_run": result["ready_for_full_solver_run"],
        "source_authenticity_verified": result["source_authenticity_verified"],
        "physical_correctness_verified": result["physical_correctness_verified"],
        "scientific_submission_ready": result["scientific_submission_ready"],
        "errors": result["errors"],
    }, indent=2, sort_keys=True))
    if args.strict and not result["ready_for_full_solver_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
