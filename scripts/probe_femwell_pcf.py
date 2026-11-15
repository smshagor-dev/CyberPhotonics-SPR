"""Run optional exploratory 2D Femwell PCF mode-solver feasibility study.

NOTE: no PML, convergence, published topology, or sensor performance validation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.simulation.pcf_femwell_probe import ExploratoryPcfTopology, run_femwell_pcf_probe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--wavelength-um", type=float, default=0.8)
    parser.add_argument("--analyte-ri", type=float, default=1.36)
    parser.add_argument("--num-modes", type=int, default=3)
    parser.add_argument("--gold-mesh-resolution-um", type=float, default=0.018)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    try:
        report = run_femwell_pcf_probe(
            args.out,
            topology=ExploratoryPcfTopology(),
            wavelength_um=args.wavelength_um,
            analyte_ri=args.analyte_ri,
            num_modes=args.num_modes,
            gold_mesh_resolution_um=args.gold_mesh_resolution_um,
        )
    except Exception as exc:
        args.out.mkdir(parents=True, exist_ok=True)
        report = {
            "schema_version": 1,
            "status": "blocked",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "fem_mode_solver_executed": False,
            "scientific_submission_ready": False,
            "physical_sensor_sensitivity_verified": False,
        }
        (args.out / "FEMWELL_BLOCKER.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2 if args.strict else 0
    print(json.dumps({
        "fem_mode_solver_executed": report["fem_mode_solver_executed"],
        "mesh_elements": report["mesh_elements"],
        "num_modes_computed": report["num_modes_computed"],
        "scientific_submission_ready": False,
        "report": str(args.out / "FEMWELL_PILOT.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
