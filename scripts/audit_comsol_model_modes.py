"""Freeze and audit reload-vs-reused-model COMSOL numerical equivalence."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sprpcf.validation.comsol_mode_equivalence import (
    compare_comsol_mode_results, freeze_mode_comparison_protocol,
)


def _source_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare completed COMSOL model-reload vs model-reuse checkpoint spectra."
    )
    cmds = parser.add_subparsers(dest="command", required=True)
    freeze = cmds.add_parser("freeze", help="Fix paired numeric comparison tolerances")
    _source_inputs(freeze)
    freeze.add_argument("--max-grid-nm", type=float, required=True)
    freeze.add_argument("--max-resonance-nm", type=float, required=True)
    freeze.add_argument("--max-fwhm-nm", type=float, required=True)
    freeze.add_argument("--max-loss-db-per-cm", type=float, required=True)

    audit = cmds.add_parser("audit", help="Check both completed, source-matched COMSOL campaigns")
    _source_inputs(audit)
    audit.add_argument("--protocol", type=Path, required=True)
    audit.add_argument("--reload-run", type=Path, required=True)
    audit.add_argument("--reuse-run", type=Path, required=True)
    audit.add_argument("--strict", action="store_true",
                       help="Exit 2 unless all pairs match AND both native-log ledgers are complete")

    args = parser.parse_args()
    if args.command == "freeze":
        result = freeze_mode_comparison_protocol(
            args.plan, args.benchmark, args.model, args.config, args.out,
            tolerances={
                "max_wavelength_grid_delta_nm": args.max_grid_nm,
                "max_resonance_delta_nm": args.max_resonance_nm,
                "max_fwhm_delta_nm": args.max_fwhm_nm,
                "max_loss_delta_db_per_cm": args.max_loss_db_per_cm,
            },
        )
        print(json.dumps({
            "protocol": str(args.out / "COMSOL_MODE_COMPARISON_PROTOCOL.json"),
            "planned_evaluations": result["scheduled_solver_evaluations"],
            "thresholds": result["tolerances"],
            "scientific_submission_ready": False,
        }, indent=2, sort_keys=True))
    else:
        result = compare_comsol_mode_results(
            args.plan, args.benchmark, args.model, args.config,
            args.protocol, args.reload_run, args.reuse_run, args.out,
        )
        print(json.dumps({
            "report": str(args.out / "COMSOL_MODE_EQUIVALENCE_AUDIT.json"),
            "paired_evaluations": result["paired_evaluations"],
            "paired_failed": result["paired_failed"],
            "all_paired_spectra_numerically_equivalent": result[
                "all_paired_spectra_numerically_equivalent"
            ],
            "native_session_logs_complete": result["native_session_logs_complete"],
            "operational_evidence_complete": result["operational_evidence_complete"],
            "scientific_submission_ready": False,
        }, indent=2, sort_keys=True))
        if args.strict and not result["operational_evidence_complete"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
