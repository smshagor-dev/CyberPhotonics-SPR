from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.ml.active_learning import load_checkpoint_inverse, select_uncertain_candidates
from sprpcf.ml.dataset import CONDITION_COLUMNS, GEOMETRY_COLUMNS, METRIC_COLUMNS, read_table
from sprpcf.simulation.comsol_sweep import (
    SENSITIVITY_GROUP_COLUMNS,
    run_comsol_geometries,
    write_dataset,
)
from sprpcf.simulation.metrics import extract_metrics, grouped_finite_difference_sensitivity
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.synthetic import synthetic_loss_spectrum
from sprpcf.validation.scientific import fixed_geometry_sweep_report

GeometryRunner = Callable[[Sequence[Geometry]], pd.DataFrame]
TargetDesigner = Callable[[pd.DataFrame], pd.DataFrame]
Retrainer = Callable[[Path, Path, Path], dict[str, Any]]


@dataclass(frozen=True)
class AcceptanceThresholds:
    """Reviewer-facing tolerances used to accept one physics-validated target."""

    max_sensitivity_error_nm_per_riu: float = 150.0
    max_fom_error_per_riu: float = 5.0
    max_lambda_error_nm: float = 30.0
    min_linearity_r2: float = 0.95

    def validate(self) -> None:
        if self.max_sensitivity_error_nm_per_riu < 0:
            raise ValueError("max_sensitivity_error_nm_per_riu must be >= 0.")
        if self.max_fom_error_per_riu < 0:
            raise ValueError("max_fom_error_per_riu must be >= 0.")
        if self.max_lambda_error_nm < 0:
            raise ValueError("max_lambda_error_nm must be >= 0.")
        if not 0.0 <= self.min_linearity_r2 <= 1.0:
            raise ValueError("min_linearity_r2 must be within [0, 1].")


@dataclass(frozen=True)
class ClosedLoopArtifacts:
    """Paths and counts produced by one closed-loop iteration."""

    output_dir: Path
    targets_with_geometry: Path
    simulation_results: Path
    verification_results: Path
    augmented_dataset: Path
    manifest: Path
    selected_targets: int
    accepted_targets: int
    appended_rows: int
    backend: str
    retrained_checkpoint: Path | None = None
    retrained_onnx: Path | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_validation_ri_values(target_ri: float, span: float = 0.04, points: int = 5) -> tuple[float, ...]:
    """Build an odd, target-centered RI sweep required to validate sensitivity."""
    target = float(target_ri)
    if not 1.0 < target < 2.0:
        raise ValueError("target_ri must be in the physically plausible interval (1, 2).")
    if span <= 0:
        raise ValueError("span must be > 0.")
    if points < 3 or points % 2 == 0:
        raise ValueError("points must be an odd integer >= 3 so the target RI is sampled exactly.")

    half = 0.5 * float(span)
    lower = max(1.000001, target - half)
    upper = min(1.999999, target + half)
    values = np.linspace(lower, upper, points, dtype=float)
    values[points // 2] = target
    unique = np.unique(np.round(values, 12))
    if unique.size != points:
        raise ValueError("RI sweep collapsed after physical-bound clipping; reduce span or points.")
    return tuple(float(value) for value in values)


def design_targets_from_checkpoint(
    checkpoint_path: Path,
    targets: pd.DataFrame,
    passes: int = 32,
    uncertainty_threshold: float | None = None,
    device: str = "cpu",
) -> pd.DataFrame:
    """Generate bounded physical geometries and MC-dropout uncertainty for sensing targets."""
    values = load_checkpoint_inverse(checkpoint_path)
    inverse, metric_mean, metric_scale, condition_mean, condition_scale, geometry_mean, geometry_scale = values
    threshold = -np.inf if uncertainty_threshold is None else float(uncertainty_threshold)
    result = select_uncertain_candidates(
        inverse=inverse,
        candidate_metrics=targets,
        metric_mean=metric_mean,
        metric_scale=metric_scale,
        condition_mean=condition_mean,
        condition_scale=condition_scale,
        geometry_mean=geometry_mean,
        geometry_scale=geometry_scale,
        uncertainty_threshold=threshold,
        passes=passes,
        device=device,
    )
    if uncertainty_threshold is None:
        return result.candidate_metrics.copy()
    return result.selected.copy()


def _target_requests(
    designed_targets: pd.DataFrame,
    ri_span: float,
    ri_points: int,
) -> tuple[pd.DataFrame, list[Geometry]]:
    required = METRIC_COLUMNS + CONDITION_COLUMNS + GEOMETRY_COLUMNS
    missing = [column for column in required if column not in designed_targets.columns]
    if missing:
        raise ValueError(f"Designed targets are missing required columns: {missing}")

    plans: list[dict[str, Any]] = []
    geometries: list[Geometry] = []
    sample_id = 0
    for target_id, row in designed_targets.reset_index(drop=True).iterrows():
        base = {
            "target_id": int(target_id),
            "source_target_id": int(row["source_target_id"])
            if "source_target_id" in row and pd.notna(row["source_target_id"])
            else int(target_id),
            "target_sensitivity_nm_per_riu": float(row["sensitivity_nm_per_riu"]),
            "target_fom_per_riu": float(row["fom_per_riu"]),
            "target_lambda_res_nm": float(row["lambda_res_nm"]),
            "target_analyte_ri": float(row["analyte_ri"]),
            "uncertainty": float(row["uncertainty"])
            if "uncertainty" in row and pd.notna(row["uncertainty"])
            else float("nan"),
        }
        for analyte_ri in build_validation_ri_values(float(row["analyte_ri"]), span=ri_span, points=ri_points):
            geometry = Geometry(
                pitch_um=float(row["pitch_um"]),
                d_over_lambda=float(row["d_over_lambda"]),
                metal_thickness_nm=float(row["metal_thickness_nm"]),
                channel_radius_um=float(row["channel_radius_um"]),
                analyte_ri=float(analyte_ri),
            )
            geometry.validate()
            plans.append({"sample_id": sample_id, **base})
            geometries.append(geometry)
            sample_id += 1
    return pd.DataFrame(plans), geometries


def run_synthetic_geometry_validation(
    geometries: Sequence[Geometry],
    wavelengths: int = 256,
    seed: int = 7,
) -> pd.DataFrame:
    """CI-safe software-only backend with the same row contract as COMSOL."""
    if wavelengths < 16:
        raise ValueError("wavelengths must be >= 16.")
    wavelength_nm = np.linspace(350.0, 950.0, wavelengths)
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for sample_id, geometry in enumerate(geometries):
        geometry.validate()
        loss = synthetic_loss_spectrum(geometry, wavelength_nm, rng)
        metrics = extract_metrics(wavelength_nm, loss)
        rows.append(
            {
                "sample_id": sample_id,
                "status": "ok",
                **geometry.__dict__,
                **metrics.__dict__,
                "wavelength_nm": ",".join(f"{value:.6f}" for value in wavelength_nm),
                "loss_db_per_cm": ",".join(f"{value:.6f}" for value in loss),
            }
        )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    sensitivity = grouped_finite_difference_sensitivity(frame, SENSITIVITY_GROUP_COLUMNS)
    frame["sensitivity_nm_per_riu"] = sensitivity
    frame["fom_per_riu"] = frame["sensitivity_nm_per_riu"].abs() / frame["fwhm_nm"]
    return frame


def _attach_request_metadata(simulation: pd.DataFrame, request_plan: pd.DataFrame) -> pd.DataFrame:
    if "sample_id" not in simulation.columns:
        raise ValueError("Simulation results must contain sample_id.")
    if simulation["sample_id"].duplicated().any():
        raise ValueError("Simulation results contain duplicate sample_id values.")
    merged = request_plan.merge(simulation, on="sample_id", how="left", validate="one_to_one")
    if "status" not in merged.columns:
        merged["status"] = "missing"
    merged["status"] = merged["status"].fillna("missing")
    for column in ["lambda_res_nm", "fwhm_nm", "sensitivity_nm_per_riu", "fom_per_riu"]:
        if column not in merged.columns:
            merged[column] = np.nan
    return merged


def evaluate_closed_loop_results(
    simulation: pd.DataFrame,
    thresholds: AcceptanceThresholds,
    expected_ri_points: int,
) -> pd.DataFrame:
    """Compare COMSOL/simulator evidence with requested sensing targets."""
    thresholds.validate()
    if expected_ri_points < 3:
        raise ValueError("expected_ri_points must be >= 3.")

    required = [
        "target_id",
        "target_sensitivity_nm_per_riu",
        "target_fom_per_riu",
        "target_lambda_res_nm",
        "target_analyte_ri",
        "status",
        *GEOMETRY_COLUMNS,
        "analyte_ri",
        "lambda_res_nm",
        "fwhm_nm",
    ]
    missing = [column for column in required if column not in simulation.columns]
    if missing:
        raise ValueError(f"Simulation results are missing verification columns: {missing}")

    successful = simulation.loc[simulation["status"].eq("ok")].copy()
    sweep_report = fixed_geometry_sweep_report(
        successful,
        group_columns=["target_id", *GEOMETRY_COLUMNS],
    )
    sweep_lookup = {int(row["target_id"]): row for row in sweep_report.to_dict(orient="records")}

    records: list[dict[str, Any]] = []
    for target_id, group in simulation.groupby("target_id", sort=True):
        first = group.iloc[0]
        reasons: list[str] = []
        ok_group = group.loc[group["status"].eq("ok")].copy()
        if len(ok_group) != expected_ri_points:
            reasons.append(f"expected {expected_ri_points} successful RI points, got {len(ok_group)}")

        report = sweep_lookup.get(int(target_id))
        if report is None:
            actual_sensitivity = float("nan")
            actual_fom = float("nan")
            linearity = float("nan")
            reasons.append("no valid fixed-geometry sensitivity fit")
        else:
            actual_sensitivity = float(report["fitted_sensitivity_nm_per_riu"])
            actual_fom = float(report["fitted_fom_per_riu"])
            linearity = float(report["linearity_r2"])

        target_ri = float(first["target_analyte_ri"])
        if ok_group.empty:
            actual_lambda = float("nan")
            reasons.append("no successful target-RI sample")
        else:
            distances = np.abs(ok_group["analyte_ri"].to_numpy(dtype=float) - target_ri)
            nearest_position = int(np.argmin(distances))
            if float(distances[nearest_position]) > 1e-9:
                actual_lambda = float("nan")
                reasons.append("target RI was not simulated exactly")
            else:
                actual_lambda = float(ok_group.iloc[nearest_position]["lambda_res_nm"])

        target_sensitivity = float(first["target_sensitivity_nm_per_riu"])
        target_fom = float(first["target_fom_per_riu"])
        target_lambda = float(first["target_lambda_res_nm"])

        sensitivity_error = (
            float(abs(actual_sensitivity - target_sensitivity))
            if np.isfinite(actual_sensitivity) and np.isfinite(target_sensitivity)
            else float("nan")
        )
        fom_error = (
            float(abs(actual_fom - target_fom))
            if np.isfinite(actual_fom) and np.isfinite(target_fom)
            else float("nan")
        )
        lambda_error = (
            float(abs(actual_lambda - target_lambda))
            if np.isfinite(actual_lambda) and np.isfinite(target_lambda)
            else float("nan")
        )

        if not np.isfinite(sensitivity_error) or sensitivity_error > thresholds.max_sensitivity_error_nm_per_riu:
            reasons.append("sensitivity tolerance exceeded")
        if not np.isfinite(fom_error) or fom_error > thresholds.max_fom_error_per_riu:
            reasons.append("FOM tolerance exceeded")
        if not np.isfinite(lambda_error) or lambda_error > thresholds.max_lambda_error_nm:
            reasons.append("resonance-wavelength tolerance exceeded")
        if not np.isfinite(linearity) or linearity < thresholds.min_linearity_r2:
            reasons.append("RI-to-resonance linearity below threshold")

        records.append(
            {
                "target_id": int(target_id),
                "source_target_id": int(first["source_target_id"]) if "source_target_id" in first else int(target_id),
                "accepted": not reasons,
                "reason": "; ".join(reasons),
                "target_sensitivity_nm_per_riu": target_sensitivity,
                "actual_sensitivity_nm_per_riu": actual_sensitivity,
                "sensitivity_abs_error_nm_per_riu": sensitivity_error,
                "target_fom_per_riu": target_fom,
                "actual_fom_per_riu": actual_fom,
                "fom_abs_error_per_riu": fom_error,
                "target_lambda_res_nm": target_lambda,
                "actual_lambda_res_nm": actual_lambda,
                "lambda_abs_error_nm": lambda_error,
                "linearity_r2": linearity,
                "uncertainty": float(first["uncertainty"])
                if "uncertainty" in first and pd.notna(first["uncertainty"])
                else float("nan"),
                **{column: float(first[column]) for column in GEOMETRY_COLUMNS},
            }
        )
    return pd.DataFrame(records)


def append_accepted_simulation_rows(
    base_dataset: pd.DataFrame,
    simulation: pd.DataFrame,
    verification: pd.DataFrame,
) -> tuple[pd.DataFrame, int]:
    """Append accepted physics rows while deduplicating geometry+RI observations."""
    accepted_ids = set(verification.loc[verification["accepted"].astype(bool), "target_id"].astype(int).tolist())
    additions = simulation.loc[
        simulation["target_id"].astype(int).isin(accepted_ids) & simulation["status"].eq("ok")
    ].copy()

    training_columns = list(dict.fromkeys([*base_dataset.columns, *additions.columns]))
    combined = pd.concat(
        [base_dataset.reindex(columns=training_columns), additions.reindex(columns=training_columns)],
        ignore_index=True,
    )
    if combined.empty:
        return combined, 0

    dedup_columns = [*GEOMETRY_COLUMNS, "analyte_ri"]
    missing = [column for column in dedup_columns if column not in combined.columns]
    if missing:
        raise ValueError(f"Cannot deduplicate augmented dataset; missing columns: {missing}")

    marker = combined[dedup_columns].astype(float).round(10).astype(str).agg("|".join, axis=1)
    before = len(combined)
    combined = combined.loc[~marker.duplicated(keep="last")].reset_index(drop=True)
    appended_rows = max(0, len(combined) - len(base_dataset))
    if before < len(combined):
        raise RuntimeError("Unexpected dataset growth during deduplication.")
    return combined, appended_rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run_closed_loop_iteration(
    checkpoint_path: Path,
    target_path: Path,
    base_dataset_path: Path,
    output_dir: Path,
    *,
    backend: str = "comsol",
    model_path: Path | None = None,
    config_path: Path | None = None,
    passes: int = 32,
    uncertainty_threshold: float | None = None,
    ri_span: float = 0.04,
    ri_points: int = 5,
    thresholds: AcceptanceThresholds | None = None,
    device: str = "cpu",
    seed: int = 7,
    designer: TargetDesigner | None = None,
    runner: GeometryRunner | None = None,
    retrain: bool = False,
    retrain_epochs: int = 50,
    retrain_batch_size: int = 64,
    retrain_device: str = "cpu",
    retrainer: Retrainer | None = None,
) -> ClosedLoopArtifacts:
    """Execute one auditable inverse-design -> physics -> append closed-loop iteration."""
    if backend not in {"comsol", "synthetic"}:
        raise ValueError("backend must be 'comsol' or 'synthetic'.")
    if passes < 2:
        raise ValueError("passes must be >= 2.")
    acceptance = thresholds or AcceptanceThresholds()
    acceptance.validate()
    if retrain_epochs < 1:
        raise ValueError("retrain_epochs must be >= 1.")
    if retrain_batch_size < 1:
        raise ValueError("retrain_batch_size must be >= 1.")

    targets = read_table(target_path).copy()
    required_targets = METRIC_COLUMNS + CONDITION_COLUMNS
    missing = [column for column in required_targets if column not in targets.columns]
    if missing:
        raise ValueError(f"Target table is missing required columns: {missing}")
    targets = targets.dropna(subset=required_targets).reset_index(drop=True)
    if "source_target_id" not in targets.columns:
        targets["source_target_id"] = np.arange(len(targets), dtype=int)
    if targets.empty:
        raise ValueError("No valid targets remain after dropping missing sensing values.")

    if designer is None:
        designed = design_targets_from_checkpoint(
            checkpoint_path,
            targets,
            passes=passes,
            uncertainty_threshold=uncertainty_threshold,
            device=device,
        )
    else:
        designed = designer(targets.copy())
    designed = designed.reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    targets_path = output_dir / "targets_with_geometry.csv"
    simulation_path = output_dir / "simulation_results.csv"
    verification_path = output_dir / "verification.csv"
    augmented_path = output_dir / (
        "augmented_dataset.parquet" if base_dataset_path.suffix.lower() == ".parquet" else "augmented_dataset.csv"
    )
    manifest_path = output_dir / "iteration_manifest.json"

    designed.to_csv(targets_path, index=False)

    base_dataset = read_table(base_dataset_path)
    if designed.empty:
        verification = pd.DataFrame(columns=["target_id", "accepted", "reason"])
        simulation = pd.DataFrame()
        augmented = base_dataset.copy()
        appended_rows = 0
    else:
        request_plan, geometries = _target_requests(designed, ri_span=ri_span, ri_points=ri_points)
        if runner is not None:
            raw_simulation = runner(geometries)
        elif backend == "synthetic":
            raw_simulation = run_synthetic_geometry_validation(geometries, seed=seed)
        else:
            if model_path is None or config_path is None:
                raise ValueError("COMSOL backend requires model_path and config_path.")
            raw_simulation = run_comsol_geometries(model_path, config_path, geometries)
        simulation = _attach_request_metadata(raw_simulation, request_plan)
        verification = evaluate_closed_loop_results(simulation, acceptance, expected_ri_points=ri_points)
        augmented, appended_rows = append_accepted_simulation_rows(base_dataset, simulation, verification)

    write_dataset(
        simulation,
        simulation_path,
        metadata={
            "source": backend,
            "evidence_class": "software_only" if backend == "synthetic" else "comsol_physics",
            "iteration": "closed_loop",
        },
    )
    verification.to_csv(verification_path, index=False)
    write_dataset(
        augmented,
        augmented_path,
        metadata={
            "source": "closed_loop_augmented",
            "backend": backend,
            "accepted_targets": int(verification["accepted"].sum()) if "accepted" in verification else 0,
            "appended_rows": int(appended_rows),
            "parent_dataset_sha256": _file_sha256(base_dataset_path),
        },
    )

    retrained_checkpoint: Path | None = None
    retrained_onnx: Path | None = None
    retrain_metrics: dict[str, Any] | None = None
    retrain_status = "not_requested"
    if retrain:
        if appended_rows <= 0:
            retrain_status = "skipped_no_new_rows"
        else:
            retrained_checkpoint = output_dir / "retrained_tandem.pt"
            retrained_onnx = output_dir / "retrained_inverse_pcf_spr.onnx"
            if retrainer is not None:
                retrain_metrics = retrainer(augmented_path, retrained_checkpoint, retrained_onnx)
            else:
                from sprpcf.ml.train_tandem import train_tandem_pipeline

                retrain_metrics = train_tandem_pipeline(
                    data_path=augmented_path,
                    checkpoint_out=retrained_checkpoint,
                    onnx_out=retrained_onnx,
                    epochs=retrain_epochs,
                    batch_size=retrain_batch_size,
                    device_name=retrain_device,
                    seed=seed,
                )
            retrain_status = "completed"

    accepted_targets = int(verification["accepted"].sum()) if "accepted" in verification else 0
    payload: dict[str, Any] = {
        "schema_version": 1,
        "backend": backend,
        "evidence_class": "software_only" if backend == "synthetic" else "comsol_physics",
        "selected_targets": int(len(designed)),
        "accepted_targets": accepted_targets,
        "appended_rows": int(appended_rows),
        "ri_span": float(ri_span),
        "ri_points": int(ri_points),
        "passes": int(passes),
        "uncertainty_threshold": uncertainty_threshold,
        "seed": int(seed),
        "acceptance_thresholds": asdict(acceptance),
        "retraining": {
            "requested": bool(retrain),
            "status": retrain_status,
            "epochs": int(retrain_epochs),
            "batch_size": int(retrain_batch_size),
            "device": retrain_device,
            "metrics": retrain_metrics,
        },
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path) if checkpoint_path.exists() else None,
            "targets": str(target_path),
            "targets_sha256": _file_sha256(target_path),
            "base_dataset": str(base_dataset_path),
            "base_dataset_sha256": _file_sha256(base_dataset_path),
            "comsol_model": str(model_path) if model_path is not None else None,
            "comsol_model_sha256": _file_sha256(model_path)
            if model_path is not None and model_path.exists()
            else None,
            "comsol_config": str(config_path) if config_path is not None else None,
            "comsol_config_sha256": _file_sha256(config_path)
            if config_path is not None and config_path.exists()
            else None,
        },
        "outputs": {
            "targets_with_geometry": str(targets_path),
            "simulation_results": str(simulation_path),
            "verification": str(verification_path),
            "augmented_dataset": str(augmented_path),
            "augmented_dataset_sha256": _file_sha256(augmented_path),
            "retrained_checkpoint": str(retrained_checkpoint) if retrained_checkpoint is not None else None,
            "retrained_checkpoint_sha256": (
                _file_sha256(retrained_checkpoint)
                if retrained_checkpoint is not None and retrained_checkpoint.exists()
                else None
            ),
            "retrained_onnx": str(retrained_onnx) if retrained_onnx is not None else None,
            "retrained_onnx_sha256": (
                _file_sha256(retrained_onnx) if retrained_onnx is not None and retrained_onnx.exists() else None
            ),
        },
    }
    _write_json(manifest_path, payload)

    return ClosedLoopArtifacts(
        output_dir=output_dir,
        targets_with_geometry=targets_path,
        simulation_results=simulation_path,
        verification_results=verification_path,
        augmented_dataset=augmented_path,
        manifest=manifest_path,
        selected_targets=int(len(designed)),
        accepted_targets=accepted_targets,
        appended_rows=int(appended_rows),
        backend=backend,
        retrained_checkpoint=retrained_checkpoint,
        retrained_onnx=retrained_onnx,
    )
