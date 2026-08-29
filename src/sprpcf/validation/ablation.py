from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.ml.train_tandem import train_tandem_pipeline
from sprpcf.validation.benchmark import evaluate_checkpoint


def _aggregate(rows: pd.DataFrame) -> dict[str, dict[str, float]]:
    metrics = [
        "forward_r2",
        "inverse_target_r2",
        "constraint_valid_rate",
        "constraint_violation_rate",
    ]
    result: dict[str, dict[str, float]] = {}
    for variant, group in rows.groupby("variant", sort=True):
        variant_result: dict[str, float] = {}
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            variant_result[f"{metric}_mean"] = float(values.mean())
            variant_result[f"{metric}_std"] = float(values.std(ddof=1)) if values.size > 1 else 0.0
        result[str(variant)] = variant_result
    return result


def run_ablation_study(
    data_path: Path,
    output_dir: Path,
    seeds: tuple[int, ...] = (7, 17, 29),
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device_name: str = "auto",
    physics_alpha: float = 1.0,
    physics_beta: float = 1.0,
    dispersion_weight: float = 0.0,
    mc_samples: int = 16,
) -> dict[str, Any]:
    """Compare the tandem model with and without fabrication/physics penalties across seeds."""
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique.")

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    variants = {
        "physics_informed": {
            "alpha": physics_alpha,
            "beta": physics_beta,
            "dispersion_weight": dispersion_weight,
        },
        "no_physics_penalty": {
            "alpha": 0.0,
            "beta": 0.0,
            "dispersion_weight": 0.0,
        },
    }

    for seed in seeds:
        for variant, config in variants.items():
            model_dir = output_dir / "models" / f"{variant}_seed_{seed}"
            checkpoint = model_dir / "tandem.pt"
            onnx = model_dir / "inverse_pcf_spr.onnx"
            train_tandem_pipeline(
                data_path=data_path,
                checkpoint_out=checkpoint,
                onnx_out=onnx,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device_name=device_name,
                alpha=float(config["alpha"]),
                beta=float(config["beta"]),
                dispersion_weight=float(config["dispersion_weight"]),
                seed=seed,
            )
            evaluation = evaluate_checkpoint(data_path, checkpoint, mc_samples=mc_samples)
            constraints = evaluation["generated_geometry_constraints_pre_projection"]
            rows.append(
                {
                    "variant": variant,
                    "seed": float(seed),
                    "forward_r2": float(evaluation["forward_surrogate"]["r2"]),
                    "inverse_target_r2": float(evaluation["inverse_target_satisfaction"]["r2"]),
                    "constraint_valid_rate": float(constraints["valid_rate"]),
                    "constraint_violation_rate": float(constraints["violation_rate"]),
                }
            )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "ablation_runs.csv", index=False)
    summary = {
        "seeds": [int(seed) for seed in seeds],
        "epochs": int(epochs),
        "runs": int(len(table)),
        "aggregate": _aggregate(table),
    }
    (output_dir / "ablation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
