from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from sprpcf.ml.dataset import (
    CONDITION_COLUMNS,
    GEOMETRY_COLUMNS,
    METRIC_COLUMNS,
    DesignDataModule,
    read_table,
)
from sprpcf.ml.losses import GEOMETRY_MAX, GEOMETRY_MIN, clamp_physical_geometry
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator


@dataclass(frozen=True)
class CalibrationProfile:
    confidence: float
    metric_half_width: np.ndarray
    ood_center: np.ndarray
    ood_precision: np.ndarray
    ood_distance_threshold: float
    calibration_rows: int
    training_reference_rows: int

    def summary(self) -> dict[str, Any]:
        return {
            "confidence": float(self.confidence),
            "calibration_rows": int(self.calibration_rows),
            "training_reference_rows": int(self.training_reference_rows),
            "metric_half_width": {
                column: float(self.metric_half_width[index])
                for index, column in enumerate(METRIC_COLUMNS)
            },
            "ood_distance_threshold": float(self.ood_distance_threshold),
        }


@dataclass(frozen=True)
class MultiObjectiveResult:
    candidates: pd.DataFrame
    selected: pd.DataFrame
    calibration: dict[str, Any]


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    required = [
        "forward_state_dict",
        "inverse_state_dict",
        "geometry_mean",
        "geometry_scale",
        "condition_mean",
        "condition_scale",
        "metric_mean",
        "metric_scale",
    ]
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Checkpoint is missing advanced-design fields: {missing}")
    return checkpoint


def _array(checkpoint: dict[str, Any], key: str) -> np.ndarray:
    value = np.asarray(checkpoint[key], dtype=np.float32)
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Checkpoint field {key!r} contains non-finite values.")
    return value


def _load_models(
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[list[ForwardNetwork], InverseGenerator]:
    states = checkpoint.get("forward_ensemble_state_dicts")
    if not states:
        states = [checkpoint["forward_state_dict"]]
    forward_models: list[ForwardNetwork] = []
    for state in states:
        model = ForwardNetwork().to(device)
        model.load_state_dict(state)
        model.eval()
        forward_models.append(model)

    inverse = InverseGenerator().to(device)
    inverse.load_state_dict(checkpoint["inverse_state_dict"])
    inverse.eval()
    return forward_models, inverse


def _standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    safe_scale = np.where(np.abs(scale) > 1e-12, scale, 1.0)
    return (values - mean) / safe_scale


def _conformal_quantile(residuals: np.ndarray, confidence: float) -> np.ndarray:
    if residuals.ndim != 2 or residuals.shape[0] < 1:
        raise ValueError("Conformal calibration requires a non-empty [rows, metrics] residual matrix.")
    n = residuals.shape[0]
    level = min(1.0, math.ceil((n + 1) * confidence) / n)
    return np.quantile(residuals, level, axis=0, method="higher")


def _ensemble_prediction_physical(
    models: list[ForwardNetwork],
    forward_input: torch.Tensor,
    metric_mean: np.ndarray,
    metric_scale: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    outputs: list[np.ndarray] = []
    with torch.no_grad():
        for model in models:
            standardized = model(forward_input).cpu().numpy()
            outputs.append(standardized * metric_scale + metric_mean)
    stacked = np.stack(outputs, axis=0)
    mean = stacked.mean(axis=0)
    if stacked.shape[0] == 1:
        std = np.zeros_like(mean)
    else:
        std = stacked.std(axis=0, ddof=1)
    return mean, std


def fit_calibration_profile(
    checkpoint_path: Path,
    reference_data_path: Path,
    confidence: float = 0.95,
    device: str = "cpu",
) -> CalibrationProfile:
    """Fit residual conformal radii and a training-domain Mahalanobis reference."""
    if not 0.5 <= confidence < 1.0:
        raise ValueError("confidence must be in [0.5, 1.0).")

    checkpoint = _load_checkpoint(checkpoint_path)
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    forward_models, _ = _load_models(checkpoint, torch_device)

    geometry_mean = _array(checkpoint, "geometry_mean")
    geometry_scale = _array(checkpoint, "geometry_scale")
    condition_mean = _array(checkpoint, "condition_mean")
    condition_scale = _array(checkpoint, "condition_scale")
    metric_mean = _array(checkpoint, "metric_mean")
    metric_scale = _array(checkpoint, "metric_scale")
    seed = int(checkpoint.get("seed", 7))

    module = DesignDataModule(reference_data_path, batch_size=256, seed=seed)
    module.setup()
    required = GEOMETRY_COLUMNS + CONDITION_COLUMNS + METRIC_COLUMNS
    frame = read_table(reference_data_path).dropna(subset=required).reset_index(drop=True)
    calibration_frame = frame.iloc[module.val_indices].copy()
    training_frame = frame.iloc[module.train_indices].copy()
    if calibration_frame.empty or training_frame.empty:
        raise ValueError("Reference data must provide non-empty train and calibration splits.")

    cal_geometry = calibration_frame[GEOMETRY_COLUMNS].to_numpy(np.float32)
    cal_condition = calibration_frame[CONDITION_COLUMNS].to_numpy(np.float32)
    cal_metrics = calibration_frame[METRIC_COLUMNS].to_numpy(np.float32)
    cal_input = np.concatenate(
        [
            _standardize(cal_geometry, geometry_mean, geometry_scale),
            _standardize(cal_condition, condition_mean, condition_scale),
        ],
        axis=1,
    ).astype(np.float32)
    predicted, _ = _ensemble_prediction_physical(
        forward_models,
        torch.tensor(cal_input, dtype=torch.float32, device=torch_device),
        metric_mean,
        metric_scale,
    )
    residuals = np.abs(predicted - cal_metrics)
    metric_half_width = _conformal_quantile(residuals, confidence).astype(np.float64)

    train_geometry = training_frame[GEOMETRY_COLUMNS].to_numpy(np.float64)
    train_condition = training_frame[CONDITION_COLUMNS].to_numpy(np.float64)
    train_input = np.concatenate(
        [
            _standardize(train_geometry, geometry_mean, geometry_scale),
            _standardize(train_condition, condition_mean, condition_scale),
        ],
        axis=1,
    )
    train_input = train_input[np.all(np.isfinite(train_input), axis=1)]
    if train_input.shape[0] < 2:
        raise ValueError("At least two finite training-reference rows are required for OOD calibration.")
    center = train_input.mean(axis=0)
    centered = train_input - center
    covariance = centered.T @ centered / max(train_input.shape[0] - 1, 1)
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float64) * 1e-4
    precision = np.linalg.pinv(covariance, hermitian=True)
    distances = np.sqrt(np.maximum(np.einsum("ni,ij,nj->n", centered, precision, centered), 0.0))
    threshold = float(np.quantile(distances, confidence, method="higher"))
    threshold = max(threshold, 1e-6)

    return CalibrationProfile(
        confidence=float(confidence),
        metric_half_width=metric_half_width,
        ood_center=center.astype(np.float64),
        ood_precision=precision.astype(np.float64),
        ood_distance_threshold=threshold,
        calibration_rows=int(calibration_frame.shape[0]),
        training_reference_rows=int(train_input.shape[0]),
    )


def pareto_ranks(objectives: np.ndarray) -> np.ndarray:
    """Return zero-based non-dominated ranks for minimization objectives."""
    values = np.asarray(objectives, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("objectives must be a non-empty 2D array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pareto objectives must be finite.")

    remaining = list(range(values.shape[0]))
    ranks = np.full(values.shape[0], -1, dtype=int)
    rank = 0
    while remaining:
        front: list[int] = []
        for index in remaining:
            candidate = values[index]
            dominated = False
            for other_index in remaining:
                if other_index == index:
                    continue
                other = values[other_index]
                if np.all(other <= candidate) and np.any(other < candidate):
                    dominated = True
                    break
            if not dominated:
                front.append(index)
        if not front:
            raise RuntimeError("Pareto ranking failed to identify a non-dominated front.")
        ranks[front] = rank
        front_set = set(front)
        remaining = [index for index in remaining if index not in front_set]
        rank += 1
    return ranks


def _ood_scores(points: np.ndarray, profile: CalibrationProfile) -> np.ndarray:
    delta = np.asarray(points, dtype=np.float64) - profile.ood_center
    distances = np.sqrt(
        np.maximum(np.einsum("ni,ij,nj->n", delta, profile.ood_precision, delta), 0.0)
    )
    return distances / profile.ood_distance_threshold


def optimize_target_table(
    checkpoint_path: Path,
    targets: pd.DataFrame,
    reference_data_path: Path,
    *,
    candidates_per_target: int = 64,
    confidence: float = 0.95,
    latent_scale: float = 0.10,
    seed: int = 7,
    device: str = "cpu",
) -> MultiObjectiveResult:
    """Generate and Pareto-rank a latent design pool for each sensing target."""
    if candidates_per_target < 4:
        raise ValueError("candidates_per_target must be >= 4.")
    if latent_scale <= 0:
        raise ValueError("latent_scale must be > 0.")
    required = METRIC_COLUMNS + CONDITION_COLUMNS
    missing = [column for column in required if column not in targets.columns]
    if missing:
        raise ValueError(f"Targets are missing required columns: {missing}")
    clean_targets = targets.dropna(subset=required).reset_index(drop=True)
    if clean_targets.empty:
        raise ValueError("No finite sensing targets are available for multi-objective design.")

    checkpoint = _load_checkpoint(checkpoint_path)
    profile = fit_calibration_profile(
        checkpoint_path,
        reference_data_path,
        confidence=confidence,
        device=device,
    )
    torch_device = torch.device(device)
    forward_models, inverse = _load_models(checkpoint, torch_device)
    geometry_mean = _array(checkpoint, "geometry_mean")
    geometry_scale = _array(checkpoint, "geometry_scale")
    condition_mean = _array(checkpoint, "condition_mean")
    condition_scale = _array(checkpoint, "condition_scale")
    metric_mean = _array(checkpoint, "metric_mean")
    metric_scale = _array(checkpoint, "metric_scale")
    safe_metric_scale = np.where(np.abs(metric_scale) > 1e-12, np.abs(metric_scale), 1.0)
    geometry_range = np.asarray(GEOMETRY_MAX, dtype=np.float64) - np.asarray(GEOMETRY_MIN, dtype=np.float64)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    all_candidates: list[pd.DataFrame] = []
    selected_rows: list[pd.Series] = []

    for target_id, row in clean_targets.iterrows():
        target_metrics = row[METRIC_COLUMNS].to_numpy(dtype=np.float32)
        target_condition = row[CONDITION_COLUMNS].to_numpy(dtype=np.float32)
        standardized_metrics = _standardize(target_metrics, metric_mean, metric_scale).astype(np.float32)
        standardized_condition = _standardize(target_condition, condition_mean, condition_scale).astype(np.float32)
        metrics_tensor = torch.tensor(
            np.repeat(standardized_metrics[None, :], candidates_per_target, axis=0),
            dtype=torch.float32,
            device=torch_device,
        )
        conditions_tensor = torch.tensor(
            np.repeat(standardized_condition[None, :], candidates_per_target, axis=0),
            dtype=torch.float32,
            device=torch_device,
        )
        latent = torch.randn(
            (candidates_per_target, inverse.latent_dim),
            generator=generator,
            dtype=torch.float32,
        ).to(torch_device) * float(latent_scale)

        with torch.no_grad():
            raw_standardized = inverse(metrics_tensor, conditions_tensor, latent)
        raw_physical = raw_standardized.cpu().numpy() * geometry_scale + geometry_mean
        clamped_tensor = clamp_physical_geometry(torch.tensor(raw_physical, dtype=torch.float32))
        physical = clamped_tensor.numpy().astype(np.float32)
        clamped_standardized = _standardize(physical, geometry_mean, geometry_scale).astype(np.float32)
        forward_input = np.concatenate(
            [clamped_standardized, np.repeat(standardized_condition[None, :], candidates_per_target, axis=0)],
            axis=1,
        ).astype(np.float32)
        predicted, ensemble_std = _ensemble_prediction_physical(
            forward_models,
            torch.tensor(forward_input, dtype=torch.float32, device=torch_device),
            metric_mean,
            metric_scale,
        )

        normalized_errors = np.abs(predicted - target_metrics[None, :]) / safe_metric_scale[None, :]
        projection = np.mean(np.abs(raw_physical - physical) / np.maximum(geometry_range, 1e-9), axis=1)
        ood = _ood_scores(forward_input, profile)
        interval_half_width = profile.metric_half_width[None, :] + ensemble_std
        covered = np.abs(predicted - target_metrics[None, :]) <= interval_half_width
        coverage_fraction = covered.mean(axis=1)

        objectives = np.column_stack([normalized_errors, projection, ood])
        ranks = pareto_ranks(objectives)
        composite = (
            normalized_errors.mean(axis=1)
            + 0.25 * projection
            + 0.20 * np.maximum(ood - 1.0, 0.0)
            + 0.10 * (1.0 - coverage_fraction)
        )
        confidence_score = np.exp(-np.clip(composite, 0.0, 50.0)) / (1.0 + np.maximum(ood - 1.0, 0.0))

        payload: dict[str, Any] = {
            "target_id": np.full(candidates_per_target, int(target_id), dtype=int),
            "source_target_id": np.full(
                candidates_per_target,
                int(row["source_target_id"])
                if "source_target_id" in row and pd.notna(row["source_target_id"])
                else int(target_id),
                dtype=int,
            ),
            "candidate_id": np.arange(candidates_per_target, dtype=int),
            "pareto_rank": ranks,
            "composite_score": composite,
            "confidence_score": confidence_score,
            "ood_score": ood,
            "in_calibration_domain": ood <= 1.0,
            "fabrication_projection_distance": projection,
            "target_interval_coverage_fraction": coverage_fraction,
            "ensemble_members": np.full(candidates_per_target, len(forward_models), dtype=int),
        }
        for index, column in enumerate(METRIC_COLUMNS):
            payload[column] = np.full(candidates_per_target, float(target_metrics[index]))
            payload[f"predicted_{column}"] = predicted[:, index]
            payload[f"normalized_error_{column}"] = normalized_errors[:, index]
            payload[f"ensemble_std_{column}"] = ensemble_std[:, index]
            payload[f"interval_half_width_{column}"] = interval_half_width[:, index]
        for index, column in enumerate(CONDITION_COLUMNS):
            payload[column] = np.full(candidates_per_target, float(target_condition[index]))
        for index, column in enumerate(GEOMETRY_COLUMNS):
            payload[column] = physical[:, index]
            payload[f"raw_{column}"] = raw_physical[:, index]

        candidates = pd.DataFrame(payload)
        uncertainty_components = []
        for index, column in enumerate(METRIC_COLUMNS):
            uncertainty_components.append(
                candidates[f"interval_half_width_{column}"].to_numpy(dtype=float) / safe_metric_scale[index]
            )
        candidates["uncertainty"] = np.mean(np.column_stack(uncertainty_components), axis=1)
        best = candidates.sort_values(
            ["pareto_rank", "composite_score", "ood_score", "candidate_id"],
            kind="mergesort",
        ).iloc[0].copy()
        candidates["selected"] = candidates["candidate_id"].eq(int(best["candidate_id"]))
        all_candidates.append(candidates)
        selected_rows.append(candidates.loc[candidates["selected"]].iloc[0])

    candidate_frame = pd.concat(all_candidates, ignore_index=True)
    selected_frame = pd.DataFrame(selected_rows).reset_index(drop=True)
    summary = profile.summary()
    summary.update(
        {
            "ensemble_members": int(len(forward_models)),
            "candidates_per_target": int(candidates_per_target),
            "latent_scale": float(latent_scale),
            "targets": int(len(clean_targets)),
            "selected_in_calibration_domain_rate": float(selected_frame["in_calibration_domain"].mean()),
            "selected_mean_confidence_score": float(selected_frame["confidence_score"].mean()),
        }
    )
    return MultiObjectiveResult(
        candidates=candidate_frame,
        selected=selected_frame,
        calibration=summary,
    )
