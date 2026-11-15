"""Execute the actual open-source planar SPP numerical sanity check.

This is a 1D TM control, *not* a PCF-SPR full-vector simulation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.simulation.open_physics import run_planar_control


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wavelength-um", type=float, default=0.8)
    parser.add_argument("--dielectric-ri", type=float, default=1.36)
    parser.add_argument("--half-domain-um", type=float, default=6.0)
    parser.add_argument("--mesh-steps-um", default="0.08,0.04,0.02,0.01")
    parser.add_argument("--strict", action="store_true", help="exit 2 unless analytic/mesh checks pass")
    args = parser.parse_args()
    try:
        steps = tuple(float(item.strip()) for item in args.mesh_steps_um.split(","))
        result = run_planar_control(
            args.out, wavelength_um=args.wavelength_um,
            dielectric_ri=args.dielectric_ri,
            half_domain_um=args.half_domain_um, mesh_steps_um=steps,
        )
    except (ValueError, ArithmeticError, RuntimeError) as exc:
        print(json.dumps({
            "pass": False, "error": str(exc), "scientific_submission_ready": False,
            "open_fem_physics_qualified": False,
        }, indent=2), file=sys.stderr)
        return 2
    print(json.dumps({
        "planar_control_pass": result["pass"],
        "relative_error": result["relative_error"],
        "relative_residual": result["relative_residual"],
        "pcf_spr_electromagnetic_model_verified": False,
        "scientific_submission_ready": False,
        "report": str(args.out / "CONTROL_READINESS.json"),
    }, indent=2))
    if args.strict and not result["pass"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
