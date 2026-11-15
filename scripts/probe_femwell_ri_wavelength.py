"""Run actual finite-box exploratory 2D PCF vectorial wavelength × RI pilot.

Outputs are unqualified solver feasibility evidence, never publication results.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.validation.open_vector_grid import run_vector_grid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wavelengths-um", default="0.75,0.8,0.85")
    parser.add_argument("--analyte-ris", default="1.35,1.36,1.37")
    parser.add_argument("--num-modes", type=int, default=2)
    parser.add_argument("--gold-mesh-resolution-um", type=float, default=0.025)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        wavelengths = tuple(float(s.strip()) for s in args.wavelengths_um.split(","))
        analyte_ris = tuple(float(s.strip()) for s in args.analyte_ris.split(","))
        report = run_vector_grid(
            args.out, wavelengths_um=wavelengths,
            analyte_ris=analyte_ris, num_modes=args.num_modes,
            gold_mesh_resolution_um=args.gold_mesh_resolution_um,
        )
    except Exception as exc:
        print(json.dumps({
            "error": str(exc),
            "exception": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "scientific_submission_ready": False,
        }, indent=2), file=sys.stderr)
        return 2 if args.strict else 1
    print(json.dumps({
        "completed": report["completed"],
        "solver_calls_completed": report["solver_calls_completed"],
        "raw_eigenmode_rows": report["raw_eigenmode_rows"],
        "vector_overlap_matched_edges": report["vector_overlap_matched_edges"],
        "scientific_submission_ready": False,
        "report": str(args.out / "GRID_READINESS.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
