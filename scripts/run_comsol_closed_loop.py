from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json

from sprpcf.simulation.remote_comsol import RemoteComsolSettings, run_remote_comsol_geometries
from sprpcf.validation.closed_loop import AcceptanceThresholds, run_closed_loop_iteration


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one PCF-SPR inverse-design -> physics -> dataset closed-loop iteration."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/closed_loop"))
    parser.add_argument("--backend", choices=["comsol", "remote-comsol", "synthetic"], default="comsol")
    parser.add_argument("--comsol-model", type=Path, default=None)
    parser.add_argument("--comsol-config", type=Path, default=None)
    parser.add_argument(
        "--remote-comsol-url",
        default=None,
        help="Remote API base URL. If omitted, SPR_COMSOL_API_URL is used.",
    )
    parser.add_argument(
        "--remote-comsol-token-env",
        default="SPR_COMSOL_API_TOKEN",
        help="Environment variable containing the bearer token. The token is never written to artifacts.",
    )
    parser.add_argument("--remote-comsol-timeout", type=float, default=300.0)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--uncertainty-threshold", type=float, default=None)
    parser.add_argument("--ri-span", type=float, default=0.04)
    parser.add_argument("--ri-points", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-sensitivity-error", type=float, default=150.0)
    parser.add_argument("--max-fom-error", type=float, default=5.0)
    parser.add_argument("--max-lambda-error", type=float, default=30.0)
    parser.add_argument("--min-linearity-r2", type=float, default=0.95)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--retrain-epochs", type=int, default=50)
    parser.add_argument("--retrain-batch-size", type=int, default=64)
    parser.add_argument("--retrain-device", default="cpu")
    return parser


def _annotate_remote_manifest(path: Path, settings: RemoteComsolSettings, token_env: str) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["execution_transport"] = "remote_comsol_api"
    payload["remote_comsol"] = {
        "base_url": settings.validated().base_url,
        "token_env": token_env,
        "token_persisted": False,
        "protocol_version": 1,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.backend == "comsol" and (args.comsol_model is None or args.comsol_config is None):
        parser.error("COMSOL backend requires --comsol-model and --comsol-config.")

    remote_settings: RemoteComsolSettings | None = None
    runner = None
    core_backend = args.backend
    if args.backend == "remote-comsol":
        try:
            remote_settings = RemoteComsolSettings.from_environment(
                base_url=args.remote_comsol_url,
                token_env=args.remote_comsol_token_env,
                timeout_seconds=args.remote_comsol_timeout,
            ).validated()
        except ValueError as exc:
            parser.error(str(exc))
        runner = lambda geometries: run_remote_comsol_geometries(remote_settings, geometries)
        core_backend = "comsol"

    thresholds = AcceptanceThresholds(
        max_sensitivity_error_nm_per_riu=args.max_sensitivity_error,
        max_fom_error_per_riu=args.max_fom_error,
        max_lambda_error_nm=args.max_lambda_error,
        min_linearity_r2=args.min_linearity_r2,
    )
    artifacts = run_closed_loop_iteration(
        checkpoint_path=args.checkpoint,
        target_path=args.targets,
        base_dataset_path=args.base_data,
        output_dir=args.out,
        backend=core_backend,
        model_path=args.comsol_model,
        config_path=args.comsol_config,
        passes=args.passes,
        uncertainty_threshold=args.uncertainty_threshold,
        ri_span=args.ri_span,
        ri_points=args.ri_points,
        thresholds=thresholds,
        device=args.device,
        seed=args.seed,
        runner=runner,
        retrain=args.retrain,
        retrain_epochs=args.retrain_epochs,
        retrain_batch_size=args.retrain_batch_size,
        retrain_device=args.retrain_device,
    )
    if remote_settings is not None:
        _annotate_remote_manifest(artifacts.manifest, remote_settings, args.remote_comsol_token_env)

    print(
        json.dumps(
            {
                "backend": args.backend,
                "evidence_class": "software_only" if args.backend == "synthetic" else "comsol_physics",
                "selected_targets": artifacts.selected_targets,
                "accepted_targets": artifacts.accepted_targets,
                "appended_rows": artifacts.appended_rows,
                "targets_with_geometry": str(artifacts.targets_with_geometry),
                "simulation_results": str(artifacts.simulation_results),
                "verification_results": str(artifacts.verification_results),
                "augmented_dataset": str(artifacts.augmented_dataset),
                "manifest": str(artifacts.manifest),
                "retrained_checkpoint": (
                    str(artifacts.retrained_checkpoint) if artifacts.retrained_checkpoint is not None else None
                ),
                "retrained_onnx": str(artifacts.retrained_onnx) if artifacts.retrained_onnx is not None else None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
