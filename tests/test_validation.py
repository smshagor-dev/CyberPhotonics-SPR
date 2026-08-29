from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sprpcf.ml.dataset import DesignDataModule
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator
from sprpcf.simulation.synthetic import build_synthetic_dataset
from sprpcf.validation.benchmark import evaluate_checkpoint, evaluate_ridge_baseline
from sprpcf.validation.scientific import (
    bootstrap_mean_ci,
    fabrication_constraint_report,
    fixed_geometry_sweep_report,
    summarize_sweep_report,
)


def test_bootstrap_mean_ci_is_deterministic() -> None:
    values = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=float)
    first = bootstrap_mean_ci(values, resamples=500, seed=11)
    second = bootstrap_mean_ci(values, resamples=500, seed=11)
    assert first == second
    assert first["ci_low"] <= first["mean"] <= first["ci_high"]


def test_fixed_geometry_report_is_independent_and_linear() -> None:
    frame = build_synthetic_dataset(samples=6, wavelengths=128, seed=3)
    report = fixed_geometry_sweep_report(frame)
    summary = summarize_sweep_report(report, bootstrap_resamples=500, seed=3)

    assert len(report) == 6
    assert np.all(report["ri_points"].to_numpy() == 5)
    assert 700.0 < float(np.median(report["fitted_sensitivity_nm_per_riu"])) < 900.0
    assert float(report["linearity_r2"].min()) > 0.98
    assert summary["geometry_sweeps"] == 6.0


def test_fabrication_constraint_report_detects_invalid_geometry() -> None:
    frame = pd.DataFrame(
        [
            {
                "pitch_um": 2.0,
                "d_over_lambda": 0.50,
                "metal_thickness_nm": 45.0,
                "channel_radius_um": 0.60,
            },
            {
                "pitch_um": 0.5,
                "d_over_lambda": 1.10,
                "metal_thickness_nm": 100.0,
                "channel_radius_um": 2.0,
            },
        ]
    )
    report = fabrication_constraint_report(frame)
    assert report["valid_rate"] == 0.5
    assert report["violation_rate"] == 0.5
    assert report["overlap_violation_rate"] == 0.5


def test_ridge_baseline_uses_grouped_validation(tmp_path: Path) -> None:
    frame = build_synthetic_dataset(samples=10, wavelengths=64, seed=5)
    data_path = tmp_path / "validation.csv"
    frame.to_csv(data_path, index=False)

    metrics = evaluate_ridge_baseline(data_path, seed=5)
    assert np.isfinite(metrics["r2"])
    assert np.isfinite(metrics["mae"])
    assert "lambda_res_nm_r2" in metrics


def test_checkpoint_evaluation_reports_target_satisfaction_and_constraints(tmp_path: Path) -> None:
    frame = build_synthetic_dataset(samples=10, wavelengths=64, seed=7)
    data_path = tmp_path / "validation.csv"
    frame.to_csv(data_path, index=False)

    data = DesignDataModule(data_path, batch_size=16, seed=7)
    data.setup()
    forward = ForwardNetwork()
    inverse = InverseGenerator()
    checkpoint_path = tmp_path / "tandem.pt"
    torch.save(
        {
            "forward_state_dict": forward.state_dict(),
            "inverse_state_dict": inverse.state_dict(),
            "geometry_mean": data.geometry_scaler.mean_,
            "geometry_scale": data.geometry_scaler.scale_,
            "condition_mean": data.condition_scaler.mean_,
            "condition_scale": data.condition_scaler.scale_,
            "metric_mean": data.metric_scaler.mean_,
            "metric_scale": data.metric_scaler.scale_,
            "seed": 7,
        },
        checkpoint_path,
    )

    report = evaluate_checkpoint(data_path, checkpoint_path, mc_samples=4)
    assert report["validation_rows"] >= 2
    assert report["generated_geometry_constraints_post_projection"]["valid_rate"] == 1.0
    assert np.isfinite(report["forward_surrogate"]["mae"])
    assert np.isfinite(report["inverse_target_satisfaction"]["mae"])
    assert all(value >= 0.0 for value in report["mc_dropout_uncertainty"].values())
