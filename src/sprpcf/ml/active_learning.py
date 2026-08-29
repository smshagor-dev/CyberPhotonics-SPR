from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint

from sprpcf.ml.dataset import CONDITION_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS, read_table
from sprpcf.ml.tandem import InverseGenerator
from sprpcf.simulation.comsol_sweep import run_comsol_geometries, write_dataset
from sprpcf.simulation.schema import Geometry


ComsolRunner = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ActiveLearningResult:
    """Candidate acquisition output for one active-learning iteration."""

    candidate_metrics: pd.DataFrame
    uncertainty: np.ndarray
    selected: pd.DataFrame
    comsol_results: pd.DataFrame | None = None


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Keep trained dropout modules stochastic while all other modules stay in eval mode."""
    model.eval()
    found = False
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()
            found = True
    if not found:
        raise ValueError("The inverse model has no Dropout layers; MC-dropout uncertainty is unavailable.")


def mc_dropout_inverse_uncertainty(
    inverse: InverseGenerator,
    target_metrics: torch.Tensor,
    conditions: torch.Tensor,
    passes: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate inverse-design epistemic uncertainty with trained MC dropout."""
    if passes < 2:
        raise ValueError("passes must be >= 2 for uncertainty estimation.")
    enable_mc_dropout(inverse)
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(passes):
            predictions.append(inverse(target_metrics, conditions))
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)


def select_uncertain_candidates(
    inverse: InverseGenerator,
    candidate_metrics: pd.DataFrame,
    metric_mean: np.ndarray,
    metric_scale: np.ndarray,
    condition_mean: np.ndarray,
    condition_scale: np.ndarray,
    geometry_mean: np.ndarray,
    geometry_scale: np.ndarray,
    uncertainty_threshold: float,
    passes: int = 32,
    device: str = "cpu",
) -> ActiveLearningResult:
    """Rank targets by MC-dropout uncertainty and attach generated physical geometries."""
    required = METRIC_COLUMNS + CONDITION_COLUMNS
    missing = [column for column in required if column not in candidate_metrics]
    if missing:
        raise ValueError(f"Missing active-learning columns: {missing}")

    torch_device = torch.device(device)
    inverse = inverse.to(torch_device)
    metrics = candidate_metrics[METRIC_COLUMNS].to_numpy(np.float32)
    conditions = candidate_metrics[CONDITION_COLUMNS].to_numpy(np.float32)
    standardized_metrics = (metrics - metric_mean.astype(np.float32)) / metric_scale.astype(np.float32)
    standardized_conditions = (conditions - condition_mean.astype(np.float32)) / condition_scale.astype(np.float32)
    mean_geometry, std_geometry = mc_dropout_inverse_uncertainty(
        inverse,
        torch.tensor(standardized_metrics, dtype=torch.float32, device=torch_device),
        torch.tensor(standardized_conditions, dtype=torch.float32, device=torch_device),
        passes=passes,
    )

    mean_physical = mean_geometry.cpu().numpy() * geometry_scale.astype(np.float32) + geometry_mean.astype(np.float32)
    lower = np.array([0.8, 0.20, 15.0, 0.20], dtype=np.float32)
    upper = np.array([4.0, 0.90, 80.0, 1.50], dtype=np.float32)
    mean_physical = np.clip(mean_physical, lower, upper)
    # Dimensionless uncertainty avoids metal-thickness units dominating the norm.
    uncertainty = std_geometry.norm(dim=1).cpu().numpy()
    enriched = candidate_metrics.copy()
    for index, column in enumerate(GEOMETRY_COLUMNS):
        enriched[column] = mean_physical[:, index]
    enriched["uncertainty"] = uncertainty
    selected = enriched.loc[uncertainty > uncertainty_threshold].copy()
    return ActiveLearningResult(candidate_metrics=enriched, uncertainty=uncertainty, selected=selected)


def _selected_to_geometries(selected: pd.DataFrame) -> list[Geometry]:
    geometries: list[Geometry] = []
    for row in selected.to_dict(orient="records"):
        geometry = Geometry(
            d_over_lambda=float(row["d_over_lambda"]),
            pitch_um=float(row["pitch_um"]),
            metal_thickness_nm=float(row["metal_thickness_nm"]),
            analyte_ri=float(row["analyte_ri"]),
            channel_radius_um=float(row["channel_radius_um"]),
        )
        geometry.validate()
        geometries.append(geometry)
    return geometries


def trigger_comsol_for_uncertain_candidates(
    result: ActiveLearningResult,
    model_path: Path,
    config_path: Path,
    output_path: Path,
    runner: ComsolRunner | None = None,
) -> ActiveLearningResult:
    """Run COMSOL only for the geometries selected by uncertainty acquisition."""
    if result.selected.empty:
        return result
    if runner is not None:
        comsol_results = runner(result.selected)
    else:
        comsol_results = run_comsol_geometries(model_path, config_path, _selected_to_geometries(result.selected))
    write_dataset(comsol_results, output_path)
    return ActiveLearningResult(
        candidate_metrics=result.candidate_metrics,
        uncertainty=result.uncertainty,
        selected=result.selected,
        comsol_results=comsol_results,
    )


def load_checkpoint_inverse(
    checkpoint_path: Path,
) -> tuple[InverseGenerator, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    checkpoint = load_tandem_checkpoint(checkpoint_path)
    inverse = InverseGenerator()
    inverse.load_state_dict(checkpoint["inverse_state_dict"])
    return (
        inverse,
        np.asarray(checkpoint["metric_mean"], dtype=np.float32),
        np.asarray(checkpoint["metric_scale"], dtype=np.float32),
        np.asarray(checkpoint["condition_mean"], dtype=np.float32),
        np.asarray(checkpoint["condition_scale"], dtype=np.float32),
        np.asarray(checkpoint["geometry_mean"], dtype=np.float32),
        np.asarray(checkpoint["geometry_scale"], dtype=np.float32),
    )


def run_active_learning_iteration(
    checkpoint_path: Path,
    candidate_path: Path,
    uncertainty_threshold: float,
    passes: int = 32,
) -> ActiveLearningResult:
    values = load_checkpoint_inverse(checkpoint_path)
    inverse, metric_mean, metric_scale, condition_mean, condition_scale, geometry_mean, geometry_scale = values
    candidates = read_table(candidate_path)
    return select_uncertain_candidates(
        inverse=inverse,
        candidate_metrics=candidates,
        metric_mean=metric_mean,
        metric_scale=metric_scale,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        geometry_mean=geometry_mean,
        geometry_scale=geometry_scale,
        uncertainty_threshold=uncertainty_threshold,
        passes=passes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MC-dropout active-learning acquisition for PCF-SPR targets.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.05)
    parser.add_argument("--passes", type=int, default=32)
    parser.add_argument("--out", type=Path, default=Path("outputs/uncertain_candidates.csv"))
    parser.add_argument("--comsol-model", type=Path, default=None)
    parser.add_argument("--comsol-config", type=Path, default=None)
    parser.add_argument("--comsol-out", type=Path, default=None)
    args = parser.parse_args()

    result = run_active_learning_iteration(args.checkpoint, args.candidates, args.threshold, args.passes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.selected.to_csv(args.out, index=False)

    comsol_args = [args.comsol_model, args.comsol_config, args.comsol_out]
    if any(value is not None for value in comsol_args):
        if not all(value is not None for value in comsol_args):
            parser.error("--comsol-model, --comsol-config, and --comsol-out must be supplied together.")
        result = trigger_comsol_for_uncertain_candidates(
            result,
            args.comsol_model,
            args.comsol_config,
            args.comsol_out,
        )

    print(f"Selected {len(result.selected)} uncertain candidates out of {len(result.candidate_metrics)}.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
