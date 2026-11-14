"""Plan, run and seal genuine local COMSOL physics verification campaigns."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from sprpcf.simulation.resumable_campaign import (inspect_campaign_checkpoint, run_checkpointed_campaign)
from sprpcf.simulation.session_provenance import (
    audit_native_session_logs, bind_native_session_log, close_crashed_solver_session,
)
from sprpcf.simulation.verification_campaign import (
    execute_comsol_campaign,
    prepare_comsol_campaign,
    seal_comsol_campaign,
)


def _paths(parser: argparse.ArgumentParser, *, include_benchmark: bool = True) -> None:
    if include_benchmark:
        parser.add_argument("--benchmark", type=Path, required=True,
                            help="Frozen Stage 4 inverse benchmark directory")
    parser.add_argument("--model", type=Path, required=True,
                        help="Genuine saved COMSOL .mph model")
    parser.add_argument("--config", type=Path, required=True,
                        help="Matching COMSOL parameter/expression/unit configuration")
    parser.add_argument("--out", type=Path, required=True,
                        help="Fresh empty output directory")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and execute *genuine* COMSOL inverse-design RI sweeps."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Freeze inputs and preregister the solver experiment")
    _paths(plan)
    plan.add_argument("--ri-span", type=float, default=0.04)
    plan.add_argument("--ri-points", type=int, default=5)
    plan.add_argument("--sensitivity-scale", type=float, required=True)
    plan.add_argument("--fom-scale", type=float, required=True)
    plan.add_argument("--resonance-scale", type=float, required=True)
    plan.add_argument("--solver-version", required=True,
                      help="Actual installed COMSOL version (not guessed)")
    run = commands.add_parser("run", help="Execute all planned sweeps using COMSOL mph")
    _paths(run)
    run.add_argument("--plan", type=Path, required=True)
    batched = commands.add_parser("run-batched", help="Hash-checkpoint every COMSOL batch; resume without rerunning completed samples")
    _paths(batched)
    batched.add_argument("--plan", type=Path, required=True)
    batched.add_argument("--batch-size", type=int, default=8)
    batched.add_argument("--resume", action="store_true", help="Verify existing committed batches and resume remaining samples")
    batched.add_argument("--recover-stale-lock", action="store_true",
                         help="ONLY when no other worker is alive; explicitly replace abandoned session lock")
    inspect = commands.add_parser("inspect", help="Read-only verification of committed batch hashes and progress")
    _paths(inspect)
    inspect.add_argument("--plan", type=Path, required=True)
    bind = commands.add_parser("bind-log", help="Bind operator-supplied native COMSOL log to one finished solver session")
    _paths(bind)
    bind.add_argument("--plan", type=Path, required=True)
    bind.add_argument("--session-id", required=True)
    bind.add_argument("--native-solver-log", type=Path, required=True)
    audit = commands.add_parser("audit-logs", help="Verify every session's registered native log and batch ownership")
    _paths(audit)
    audit.add_argument("--plan", type=Path, required=True)
    close = commands.add_parser("close-crashed-session",
                                help="Explicitly close a session after checking its worker is gone")
    close.add_argument("--out", type=Path, required=True)
    close.add_argument("--session-id", required=True)
    seal = commands.add_parser("seal", help="Seal real solver artifacts for Stage 5")
    _paths(seal)
    seal.add_argument("--plan", type=Path, required=True)
    seal.add_argument("--run", type=Path, required=True,
                      help="Output directory from real COMSOL run command")
    seal.add_argument("--native-solver-log", type=Path, required=True,
                      help="Authentic solver-produced log covering this campaign")
    seal.add_argument("--refinement-archive", type=Path, required=True,
                      help="Genuine mesh and wavelength refinement archive for same model")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "plan":
        result = prepare_comsol_campaign(
            args.benchmark, args.model, args.config, args.out,
            ri_span=args.ri_span, ri_points=args.ri_points,
            normalization_scales={
                "sensitivity_nm_per_riu": args.sensitivity_scale,
                "fom_per_riu": args.fom_scale,
                "lambda_res_nm": args.resonance_scale,
            }, solver_version=args.solver_version,
        )
    elif args.command == "run":
        result = execute_comsol_campaign(
            args.plan, args.benchmark, args.model, args.config, args.out,
        )
    elif args.command == "run-batched":
        result = run_checkpointed_campaign(
            args.plan, args.benchmark, args.model, args.config, args.out,
            batch_size=args.batch_size, resume=args.resume,
            recover_stale_lock=args.recover_stale_lock,
        )
    elif args.command == "inspect":
        result = inspect_campaign_checkpoint(
            args.plan, args.benchmark, args.model, args.config, args.out,
        )
    elif args.command in ("bind-log", "audit-logs"):
        inspected = inspect_campaign_checkpoint(
            args.plan, args.benchmark, args.model, args.config, args.out,
        )
        checkpoint = json.loads((args.out / "checkpoint_manifest.json").read_text(encoding="utf-8"))
        if checkpoint["backend_kind"] != "comsol_mph":
            raise SystemExit("Native solver logs may only be bound to a COMSOL backend, not fixtures")
        if args.command == "bind-log":
            result = bind_native_session_log(
                args.out, args.session_id, args.native_solver_log, checkpoint,
                inspected["chunk_evidence"],
            )
        else:
            result = {
                "session_logs": audit_native_session_logs(
                    args.out, checkpoint, inspected["chunk_evidence"],
                ),
                "source_authenticity_verified": False,
            }
    elif args.command == "close-crashed-session":
        result = close_crashed_solver_session(args.out, args.session_id)
    else:
        result = seal_comsol_campaign(
            args.plan, args.run, args.benchmark, args.model, args.config,
            args.native_solver_log, args.refinement_archive, args.out,
        )
    print(json.dumps({
        "operation": args.command,
        "output_dir": str(args.out),
        "real_comsol_backend_invoked": result.get("real_comsol_backend_invoked"),
        "scheduled_solver_evaluations": result.get(
            "scheduled_solver_evaluations", result.get("planned_solver_evaluations")
        ),
        "failed_solver_evaluations": result.get("failed_solver_evaluations"),
        "committed_solver_evaluations": result.get("committed_solver_evaluations"),
        "remaining_solver_evaluations": result.get("remaining_solver_evaluations"),
        "execution_mode": result.get("execution_mode"),
        "session_id": result.get("session_id"),
        "session_log_sha256": result.get("log_sha256"),
        "session_count": len(result["session_logs"]) if "session_logs" in result else None,
        "source_authenticity_verified": result.get("source_authenticity_verified",
                                                    result.get("physical_source_authenticity_verified", False)),
        "scientific_submission_ready": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
