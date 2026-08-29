from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import argparse
import json
from pathlib import Path

import pandas as pd

from sprpcf.ml.multiobjective import optimize_target_table
from sprpcf.validation.closed_loop import AcceptanceThresholds, run_closed_loop_iteration


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run calibrated Pareto inverse design -> COMSOL/synthetic physics -> dataset closed loop."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--base-data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/advanced_closed_loop"))
    parser.add_argument("--backend", choices=["comsol", "synthetic"], default="comsol")
    parser.add_argument("--comsol-model", type=Path, default=None)
    parser.add_argument("--comsol-config", type=Path, default=None)
    parser.add_argument("--candidates-per-target", type=int, default=128)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--latent-scale", type=float, default=0.10)
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
    args = parser.parse_args()

    if args.backend == "comsol" and (args.comsol_model is None or args.comsol_config is None):
        parser.error("COMSOL backend requires --comsol-model and --comsol-config.")

    args.out.mkdir(parents=True, exist_ok=True)
    design_summary: dict[str, object] = {}

    def advanced_designer(targets: pd.DataFrame) -> pd.DataFrame:
        result = optimize_target_table(
            args.checkpoint,
            targets,
            args.base_data,
            candidates_per_target=args.candidates_per_target,
            confidence=args.confidence,
            latent_scale=args.latent_scale,
            seed=args.seed,
            device=args.device,
        )
        result.candidates.to_csv(args.out / "pareto_candidates.csv", index=False)
        result.selected.to_csv(args.out / "pareto_selected_designs.csv", index=False)
        (args.out / "design_calibration.json").write_text(
            json.dumps(result.calibration, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        design_summary.update(result.calibration)
        return result.selected

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
        backend=args.backend,
        model_path=args.comsol_model,
        config_path=args.comsol_config,
        ri_span=args.ri_span,
        ri_points=args.ri_points,
        thresholds=thresholds,
        device=args.device,
        seed=args.seed,
        designer=advanced_designer,
        retrain=args.retrain,
        retrain_epochs=args.retrain_epochs,
        retrain_batch_size=args.retrain_batch_size,
        retrain_device=args.retrain_device,
    )
    print(
        json.dumps(
            {
                "backend": artifacts.backend,
                "selected_targets": artifacts.selected_targets,
                "accepted_targets": artifacts.accepted_targets,
                "appended_rows": artifacts.appended_rows,
                "pareto_candidates": str(args.out / "pareto_candidates.csv"),
                "pareto_selected_designs": str(args.out / "pareto_selected_designs.csv"),
                "design_calibration": str(args.out / "design_calibration.json"),
                "manifest": str(artifacts.manifest),
                "calibration_summary": design_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
