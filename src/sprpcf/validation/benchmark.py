from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sprpcf.ml.checkpoint_io import load_tandem_checkpoint
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sprpcf.ml.dataset import (
    CONDITION_COLUMNS,
    FORWARD_INPUT_COLUMNS,
    GEOMETRY_COLUMNS,
    METRIC_COLUMNS,
    DesignDataModule,
    geometry_group_labels,
    read_table,
)
from sprpcf.ml.losses import clamp_physical_geometry
from sprpcf.ml.tandem import ForwardNetwork, InverseGenerator
from sprpcf.validation.scientific import (
    fabrication_constraint_report,
    fixed_geometry_sweep_report,
    summarize_sweep_report,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if truth.shape != pred.shape or truth.ndim != 2:
        raise ValueError("Regression arrays must have matching [N, targets] shapes.")
    if truth.shape[0] < 2:
        raise ValueError("At least two validation rows are required.")

    residual = np.sum(np.square(truth - pred), axis=0)
    centered = truth - truth.mean(axis=0, keepdims=True)
    total = np.sum(np.square(centered), axis=0)
    per_target_r2 = 1.0 - residual / np.maximum(total, 1e-12)
    rmse = np.sqrt(np.mean(np.square(truth - pred), axis=0))
    mae = np.mean(np.abs(truth - pred), axis=0)

    result: dict[str, float] = {
        "r2": float(np.mean(per_target_r2)),
        "rmse": float(np.mean(rmse)),
        "mae": float(np.mean(mae)),
    }
    for index, column in enumerate(METRIC_COLUMNS):
        result[f"{column}_r2"] = float(per_target_r2[index])
        result[f"{column}_rmse"] = float(rmse[index])
        result[f"{column}_mae"] = float(mae[index])
    return result


def evaluate_ridge_baseline(
    data_path: Path,
    test_size: float = 0.2,
    seed: int = 7,
) -> dict[str, float]:
    """Evaluate a simple leakage-resistant linear baseline on physical inputs."""
    required = FORWARD_INPUT_COLUMNS + METRIC_COLUMNS
    frame = read_table(data_path).dropna(subset=required).reset_index(drop=True)
    if len(frame) < 4:
        raise ValueError("At least four valid rows are required for baseline validation.")

    groups = geometry_group_labels(frame)
    if np.unique(groups).size < 2:
        raise ValueError("At least two unique geometries are required for grouped validation.")

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, val_idx = next(splitter.split(frame, groups=groups))
    x = frame[FORWARD_INPUT_COLUMNS].to_numpy(dtype=float)
    y = frame[METRIC_COLUMNS].to_numpy(dtype=float)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ]
    )
    model.fit(x[train_idx], y[train_idx])
    prediction = model.predict(x[val_idx])
    return _regression_metrics(y[val_idx], prediction)


def _checkpoint_scaler(checkpoint: dict[str, Any], prefix: str) -> tuple[np.ndarray, np.ndarray]:
    mean_key = f"{prefix}_mean"
    scale_key = f"{prefix}_scale"
    if mean_key not in checkpoint or scale_key not in checkpoint:
        raise ValueError(f"Checkpoint is missing {mean_key}/{scale_key}.")
    mean = np.asarray(checkpoint[mean_key], dtype=np.float32)
    scale = np.asarray(checkpoint[scale_key], dtype=np.float32)
    if np.any(scale <= 0):
        raise ValueError(f"Checkpoint {scale_key} contains a non-positive scale.")
    return mean, scale


def evaluate_checkpoint(
    data_path: Path,
    checkpoint_path: Path,
    mc_samples: int = 32,
) -> dict[str, Any]:
    """Evaluate target satisfaction, constraints, and MC-dropout uncertainty."""
    if mc_samples < 2:
        raise ValueError("mc_samples must be >= 2.")

    checkpoint = load_tandem_checkpoint(checkpoint_path)
    seed = int(checkpoint.get("seed", 7))
    data = DesignDataModule(data_path, batch_size=256, seed=seed)
    data.setup()

    forward = ForwardNetwork()
    inverse = InverseGenerator()
    forward.load_state_dict(checkpoint["forward_state_dict"])
    inverse.load_state_dict(checkpoint["inverse_state_dict"])
    forward.eval()
    inverse.eval()

    geometry_mean, geometry_scale = _checkpoint_scaler(checkpoint, "geometry")
    metric_mean, metric_scale = _checkpoint_scaler(checkpoint, "metric")
    condition_mean, condition_scale = _checkpoint_scaler(checkpoint, "condition")

    geometry_mean_t = torch.tensor(geometry_mean, dtype=torch.float32)
    geometry_scale_t = torch.tensor(geometry_scale, dtype=torch.float32)

    forward_truth: list[np.ndarray] = []
    forward_pred: list[np.ndarray] = []
    target_truth: list[np.ndarray] = []
    target_pred: list[np.ndarray] = []
    generated_geometry_raw: list[np.ndarray] = []
    generated_geometry_projected: list[np.ndarray] = []
    mc_geometry_std: list[np.ndarray] = []
    mc_metric_std: list[np.ndarray] = []

    with torch.no_grad():
        for geometry, conditions, target_metrics in data.val_loader():
            predicted_standard = forward(torch.cat([geometry, conditions], dim=-1))
            forward_truth.append(target_metrics.numpy() * metric_scale + metric_mean)
            forward_pred.append(predicted_standard.numpy() * metric_scale + metric_mean)

            generated_standard = inverse(target_metrics, conditions)
            generated_physical_raw = generated_standard * geometry_scale_t + geometry_mean_t
            generated_physical = clamp_physical_geometry(generated_physical_raw)
            generated_standard_clamped = (generated_physical - geometry_mean_t) / geometry_scale_t
            satisfied_standard = forward(torch.cat([generated_standard_clamped, conditions], dim=-1))

            target_truth.append(target_metrics.numpy() * metric_scale + metric_mean)
            target_pred.append(satisfied_standard.numpy() * metric_scale + metric_mean)
            generated_geometry_raw.append(generated_physical_raw.numpy())
            generated_geometry_projected.append(generated_physical.numpy())

            inverse.train()
            geometry_samples: list[np.ndarray] = []
            metric_samples: list[np.ndarray] = []
            for _ in range(mc_samples):
                sampled_standard = inverse(target_metrics, conditions)
                sampled_physical = sampled_standard * geometry_scale_t + geometry_mean_t
                sampled_physical = clamp_physical_geometry(sampled_physical)
                sampled_standard_clamped = (sampled_physical - geometry_mean_t) / geometry_scale_t
                sampled_metric = forward(torch.cat([sampled_standard_clamped, conditions], dim=-1))
                geometry_samples.append(sampled_physical.numpy())
                metric_samples.append(sampled_metric.numpy() * metric_scale + metric_mean)
            inverse.eval()

            mc_geometry_std.append(np.std(np.stack(geometry_samples, axis=0), axis=0))
            mc_metric_std.append(np.std(np.stack(metric_samples, axis=0), axis=0))

    forward_truth_array = np.concatenate(forward_truth, axis=0)
    forward_pred_array = np.concatenate(forward_pred, axis=0)
    target_truth_array = np.concatenate(target_truth, axis=0)
    target_pred_array = np.concatenate(target_pred, axis=0)
    generated_geometry_raw_array = np.concatenate(generated_geometry_raw, axis=0)
    generated_geometry_projected_array = np.concatenate(generated_geometry_projected, axis=0)

    raw_geometry_frame = pd.DataFrame(generated_geometry_raw_array, columns=GEOMETRY_COLUMNS)
    projected_geometry_frame = pd.DataFrame(generated_geometry_projected_array, columns=GEOMETRY_COLUMNS)
    geometry_std = np.concatenate(mc_geometry_std, axis=0)
    metric_std = np.concatenate(mc_metric_std, axis=0)

    uncertainty: dict[str, float] = {}
    for index, column in enumerate(GEOMETRY_COLUMNS):
        uncertainty[f"{column}_mc_std_mean"] = float(np.mean(geometry_std[:, index]))
    for index, column in enumerate(METRIC_COLUMNS):
        uncertainty[f"{column}_mc_std_mean"] = float(np.mean(metric_std[:, index]))

    return {
        "validation_rows": float(target_truth_array.shape[0]),
        "forward_surrogate": _regression_metrics(forward_truth_array, forward_pred_array),
        "inverse_target_satisfaction": _regression_metrics(target_truth_array, target_pred_array),
        "generated_geometry_constraints_pre_projection": fabrication_constraint_report(raw_geometry_frame),
        "generated_geometry_constraints_post_projection": fabrication_constraint_report(projected_geometry_frame),
        "mc_dropout_uncertainty": uncertainty,
        "checkpoint_seed": float(seed),
        "condition_scaler": {
            "mean": [float(value) for value in condition_mean],
            "scale": [float(value) for value in condition_scale],
        },
    }


def _plot_resonance_shift(frame: pd.DataFrame, sweep_report: pd.DataFrame, output: Path) -> None:
    if sweep_report.empty:
        return
    first = sweep_report.iloc[0]
    mask = np.ones(len(frame), dtype=bool)
    for column in GEOMETRY_COLUMNS:
        mask &= np.isclose(frame[column].to_numpy(dtype=float), float(first[column]), rtol=0.0, atol=1e-10)
    sweep = frame.loc[mask, ["analyte_ri", "lambda_res_nm"]].dropna().sort_values("analyte_ri")
    if len(sweep) < 2:
        return

    ri = sweep["analyte_ri"].to_numpy(dtype=float)
    wavelength = sweep["lambda_res_nm"].to_numpy(dtype=float)
    slope, intercept = np.polyfit(ri, wavelength, 1)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(ri, wavelength, label="Observed resonance")
    ax.plot(ri, slope * ri + intercept, label=f"Linear fit: {slope:.1f} nm/RIU")
    ax.set_xlabel("Analyte refractive index")
    ax.set_ylabel("Resonance wavelength (nm)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_sensitivity_distribution(sweep_report: pd.DataFrame, output: Path) -> None:
    values = sweep_report["fitted_sensitivity_nm_per_riu"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.hist(values, bins=min(20, max(5, int(np.sqrt(values.size)))))
    ax.set_xlabel("Fitted wavelength sensitivity (nm/RIU)")
    ax.set_ylabel("Geometry count")
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _plot_model_comparison(
    baseline: dict[str, float],
    checkpoint: dict[str, Any],
    output: Path,
) -> None:
    labels = ["Ridge baseline", "Neural surrogate", "Inverse target satisfaction"]
    values = [
        baseline["r2"],
        checkpoint["forward_surrogate"]["r2"],
        checkpoint["inverse_target_satisfaction"]["r2"],
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.bar(labels, values)
    ax.set_ylabel("Mean R² across sensing metrics")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _write_markdown_report(summary: dict[str, Any], output: Path) -> None:
    sweep = summary["fixed_geometry_sweeps"]
    sensitivity = sweep["sensitivity_nm_per_riu"]
    lines = [
        "# Scientific Validation Pack",
        "",
        "This report is generated from fixed-geometry RI sweeps and leakage-resistant model validation.",
        "",
        "## Fixed-geometry optical validation",
        "",
        f"- Geometry sweeps: {int(sweep['geometry_sweeps'])}",
        (
            "- Mean fitted sensitivity: "
            f"{sensitivity['mean']:.3f} nm/RIU "
            f"({100.0 * sensitivity['confidence']:.1f}% bootstrap CI "
            f"{sensitivity['ci_low']:.3f}–{sensitivity['ci_high']:.3f})"
        ),
        f"- Median RI-to-resonance linearity R²: {sweep['median_linearity_r2']:.6f}",
        f"- Minimum RI-to-resonance linearity R²: {sweep['minimum_linearity_r2']:.6f}",
        "",
        "## Leakage-resistant forward baseline",
        "",
        f"- Ridge baseline mean R²: {summary['ridge_baseline']['r2']:.6f}",
        f"- Ridge baseline mean MAE: {summary['ridge_baseline']['mae']:.6f}",
    ]

    if "checkpoint" in summary:
        checkpoint = summary["checkpoint"]
        raw_constraints = checkpoint["generated_geometry_constraints_pre_projection"]
        projected_constraints = checkpoint["generated_geometry_constraints_post_projection"]
        lines.extend(
            [
                "",
                "## Tandem checkpoint validation",
                "",
                f"- Forward surrogate mean R²: {checkpoint['forward_surrogate']['r2']:.6f}",
                (
                    "- Inverse target-satisfaction mean R²: "
                    f"{checkpoint['inverse_target_satisfaction']['r2']:.6f}"
                ),
                f"- Raw generated-geometry valid rate: {raw_constraints['valid_rate']:.6f}",
                f"- Raw generated-geometry violation rate: {raw_constraints['violation_rate']:.6f}",
                f"- Post-projection valid rate: {projected_constraints['valid_rate']:.6f}",
            ]
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Synthetic results validate the software and evaluation pipeline only. "
            "Use verified COMSOL or experimental spectra before making physical performance claims.",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8")


def run_validation_pack(
    data_path: Path,
    output_dir: Path,
    checkpoint_path: Path | None = None,
    bootstrap_resamples: int = 2000,
    mc_samples: int = 32,
    seed: int = 7,
) -> dict[str, Any]:
    """Generate tables, uncertainty statistics, plots, provenance, and a Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = read_table(data_path)
    sweep_report = fixed_geometry_sweep_report(frame)
    sweep_summary = summarize_sweep_report(
        sweep_report,
        bootstrap_resamples=bootstrap_resamples,
        seed=seed,
    )
    baseline = evaluate_ridge_baseline(data_path, seed=seed)

    summary: dict[str, Any] = {
        "fixed_geometry_sweeps": sweep_summary,
        "ridge_baseline": baseline,
        "dataset_constraints": fabrication_constraint_report(frame),
    }
    if checkpoint_path is not None:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        summary["checkpoint"] = evaluate_checkpoint(data_path, checkpoint_path, mc_samples=mc_samples)

    sweep_report.to_csv(output_dir / "fixed_geometry_sweeps.csv", index=False)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    provenance = {
        "dataset": str(data_path),
        "dataset_sha256": _sha256_file(data_path),
        "checkpoint": str(checkpoint_path) if checkpoint_path is not None else None,
        "checkpoint_sha256": _sha256_file(checkpoint_path) if checkpoint_path is not None else None,
        "bootstrap_resamples": int(bootstrap_resamples),
        "mc_samples": int(mc_samples),
        "seed": int(seed),
        "geometry_columns": GEOMETRY_COLUMNS,
        "condition_columns": CONDITION_COLUMNS,
        "metric_columns": METRIC_COLUMNS,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    _plot_resonance_shift(frame, sweep_report, output_dir / "resonance_shift.png")
    _plot_sensitivity_distribution(sweep_report, output_dir / "sensitivity_distribution.png")
    if "checkpoint" in summary:
        _plot_model_comparison(baseline, summary["checkpoint"], output_dir / "model_comparison_r2.png")
    _write_markdown_report(summary, output_dir / "validation_report.md")
    return summary
