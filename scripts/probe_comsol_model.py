"""Check an actual COMSOL model's required studies/parameters before a campaign."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from sprpcf.simulation.model_gate import audit_comsol_model_readiness


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fail-closed source-bound COMSOL readiness audit; no synthetic solver fallback."
    )
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--live", action="store_true",
                        help="Load real COMSOL model and inspect study, params, material, mesh, physics")
    parser.add_argument("--pilot", action="store_true",
                        help="Requires --live; solve exactly one frozen target-center RI sample")
    parser.add_argument("--strict", action="store_true",
                        help="Exit 2 unless actual live one-sample solver pilot passes")
    args = parser.parse_args()
    result = audit_comsol_model_readiness(
        args.plan, args.benchmark, args.model, args.config, args.out,
        live=args.live, pilot=args.pilot,
    )
    print(json.dumps({
        "static_valid": result["static_valid"],
        "live_model_checked": result["live_model_checked"],
        "pilot_solver_called": result["pilot_solver_called"],
        "pilot_numerically_consistent": result["pilot_numerically_consistent"],
        "ready_for_full_solver_run": result["ready_for_full_solver_run"],
        "physical_correctness_verified": result["physical_correctness_verified"],
        "scientific_submission_ready": result["scientific_submission_ready"],
        "errors": result["errors"],
        "report": str(args.out / "COMSOL_MODEL_PILOT_READINESS.json"),
    }, indent=2, sort_keys=True))
    if args.strict and not result["ready_for_full_solver_run"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
