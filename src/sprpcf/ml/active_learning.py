from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch

from sprpcf.ml.dataset import METRIC_COLUMNS, read_table
from sprpcf.ml.tandem import InverseGenerator
from sprpcf.simulation.comsol_sweep import run_comsol_sweep, write_dataset


ComsolRunner = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass(frozen=True)
class ActiveLearningResult:
    """Candidate acquisition output for one active-learning iteration."""

    candidate_metrics: pd.DataFrame
    uncertainty: np.ndarray
    selected: pd.DataFrame
    comsol_results: pd.DataFrame | None = None


def enable_mc_dropout(model: torch.nn.Module) -> None:
    """Keep dropout modules stochastic while other layers remain in eval mode."""
    model.eval()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def mc_dropout_inverse_uncertainty(
    inverse: InverseGenerator,
    target_metrics: torch.Tensor,
    passes: int = 32,
    latent_std: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate inverse-design uncertainty from stochastic latent/dropout samples."""
    if passes < 2:
        raise ValueError("passes must be >= 2 for uncertainty estimation.")
    enable_mc_dropout(inverse)
    predictions: list[torch.Tensor] = []
    with torch.no_grad():
        for _ in range(passes):
            latent = torch.randn(target_metrics.shape[0], inverse.latent_dim, device=target_metrics.device) * latent_std
            predictions.append(inverse(target_metrics, latent))
    stacked = torch.stack(predictions, dim=0)
    return stacked.mean(dim=0), stacked.std(dim=0)


def select_uncertain_candidates(
    inverse: InverseGenerator,
    candidate_metrics: pd.DataFrame,
    metric_mean: np.ndarray,
    metric_scale: np.ndarray,
    uncertainty_threshold: float,
    passes: int = 32,
    latent_std: float = 0.1,
    device: str = "cpu",
) -> ActiveLearningResult:
    """Rank target metrics by inverse-model uncertainty and return selected rows."""
    missing = [column for column in METRIC_COLUMNS if column not in candidate_metrics]
    if missing:
        raise ValueError(f"Missing metric columns: {missing}")
    torch_device = torch.device(device)
    inverse = inverse.to(torch_device)
    metrics = candidate_metrics[METRIC_COLUMNS].to_numpy(np.float32)
    standardized = (metrics - metric_mean.astype(np.float32)) / metric_scale.astype(np.float32)
    _, std_geometry = mc_dropout_inverse_uncertainty(
        inverse,
        torch.tensor(standardized, dtype=torch.float32, device=torch_device),
        passes=passes,
        latent_std=latent_std,
    )
    uncertainty = std_geometry.norm(dim=1).cpu().numpy()
    selected = candidate_metrics.loc[uncertainty > uncertainty_threshold].copy()
    selected["uncertainty"] = uncertainty[uncertainty > uncertainty_threshold]
    return ActiveLearningResult(candidate_metrics=candidate_metrics, uncertainty=uncertainty, selected=selected)


def trigger_comsol_for_uncertain_candidates(
    result: ActiveLearningResult,
    model_path: Path,
    config_path: Path,
    output_path: Path,
    runner: ComsolRunner | None = None,
) -> ActiveLearningResult:
    """Run COMSOL for uncertain candidates.

    A custom runner can be injected for tests or schedulers. Without one, this
    calls the configured COMSOL sweep adapter.
    """
    if result.selected.empty:
        return result
    if runner is not None:
        comsol_results = runner(result.selected)
    else:
        comsol_results = run_comsol_sweep(model_path, config_path)
    write_dataset(comsol_results, output_path)
    return ActiveLearningResult(
        candidate_metrics=result.candidate_metrics,
        uncertainty=result.uncertainty,
        selected=result.selected,
        comsol_results=comsol_results,
    )


def load_checkpoint_inverse(checkpoint_path: Path) -> tuple[InverseGenerator, np.ndarray, np.ndarray]:
    """Load inverse generator and metric scaler values from a tandem checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    inverse = InverseGenerator()
    inverse.load_state_dict(checkpoint["inverse_state_dict"])
    return inverse, np.asarray(checkpoint["metric_mean"], dtype=np.float32), np.asarray(checkpoint["metric_scale"], dtype=np.float32)


def run_active_learning_iteration(
    checkpoint_path: Path,
    candidate_path: Path,
    uncertainty_threshold: float,
    passes: int = 32,
) -> ActiveLearningResult:
    """Convenience helper for file-based active-learning acquisition."""
    inverse, metric_mean, metric_scale = load_checkpoint_inverse(checkpoint_path)
    candidates = read_table(candidate_path)
    return select_uncertain_candidates(
        inverse=inverse,
        candidate_metrics=candidates,
        metric_mean=metric_mean,
        metric_scale=metric_scale,
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
    args = parser.parse_args()

    result = run_active_learning_iteration(args.checkpoint, args.candidates, args.threshold, args.passes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.selected.to_csv(args.out, index=False)
    print(f"Selected {len(result.selected)} uncertain candidates out of {len(result.candidate_metrics)}.")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
