"""Run genuine exploratory PCF full-vector FEM mesh refinement (NOT paper evidence)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.validation.open_mesh_refinement import run_mesh_refinement


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gold-mesh-steps-um", default="0.03,0.025,0.02")
    parser.add_argument("--wavelength-um", type=float, default=0.8)
    parser.add_argument("--analyte-ri", type=float, default=1.36)
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        mesh_steps = tuple(float(x.strip()) for x in args.gold_mesh_steps_um.split(","))
        result = run_mesh_refinement(
            args.out, gold_mesh_steps_um=mesh_steps,
            wavelength_um=args.wavelength_um,
            analyte_ri=args.analyte_ri, num_modes=args.num_modes,
        )
    except Exception as exc:
        print(json.dumps({
            "error": str(exc),
            "error_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "scientific_submission_ready": False,
        }, indent=2), file=sys.stderr)
        return 2 if args.strict else 1
    print(json.dumps({
        "completed": result["completed"],
        "solver_runs": len(result["runs"]),
        "matched_comparisons": len(result["comparisons"]),
        "max_last_step_complex_neff_relative_delta": result["max_last_step_complex_neff_relative_delta"],
        "scientific_submission_ready": False,
        "mesh_convergence_verified": False,
        "report": str(args.out / "REFINEMENT_READINESS.json"),
    }, indent=2))
    return 0 if result["completed"] else (2 if args.strict else 1)


if __name__ == "__main__":
    raise SystemExit(main())
