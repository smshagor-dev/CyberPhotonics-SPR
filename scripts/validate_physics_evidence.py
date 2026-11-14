"""Generate one conservative numerical-physics evidence gate report.

Passing archive checks do not replace independent scientific model review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.validation.convergence import audit_refinement_archive
from sprpcf.validation.physics_audit import audit_comsol_iteration


def build_physics_evidence_gate(iteration: Path, refinement_archive: Path) -> dict:
    spectrum = audit_comsol_iteration(iteration)
    refinement = audit_refinement_archive(refinement_archive)
    checks_pass = bool(spectrum["consistency_ok"] and refinement["convergence_consistent"])
    return {
        "schema_version": 1,
        "audit_type": "combined_numerical_physics_evidence_gate",
        "archived_records_consistent": checks_pass,
        "independent_physics_correctness_verified": False,
        "fabricated_sensor_measured": False,
        "submission_physics_claim_ready": False,
        "spectrum_audit": spectrum,
        "mesh_and_spectral_audit": refinement,
        "pending_independent_review": [
            "Authenticity of real COMSOL solver execution and solver logs",
            "Physical formulation, materials, geometry, mode selection and PML/boundaries",
            "Full fixed-geometry RI sweeps, mesh/spectral convergence and literature comparison",
            "Scientific interpretation and paper claim-to-evidence traceability",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check archived COMSOL consistency without certifying physical science.")
    parser.add_argument("--iteration", type=Path, required=True)
    parser.add_argument("--refinement-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict", action="store_true", help="Exit code 2 on missing or inconsistent numerical evidence")
    args = parser.parse_args()
    report = build_physics_evidence_gate(args.iteration, args.refinement_archive)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    if args.strict and not report["archived_records_consistent"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
