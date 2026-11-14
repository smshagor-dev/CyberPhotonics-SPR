from __future__ import annotations

"""Consistency audit of claimed COMSOL closed-loop results.

This validates recorded values and provenance against stored spectra; it cannot
establish that the supplied model solves the intended electromagnetic problem.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS
from sprpcf.simulation.metrics import extract_metrics
from sprpcf.utils.reproducibility import sha256_file
from sprpcf.validation.closed_loop import (
    AcceptanceThresholds,
    build_validation_ri_values,
    evaluate_closed_loop_results,
)

_NUMERIC_FIELDS = (
    "actual_sensitivity_nm_per_riu",
    "actual_fom_per_riu",
    "actual_lambda_res_nm",
    "linearity_r2",
)


def _artifact(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing evidence artifact path.")
    stored = Path(value)
    candidates = [stored] if stored.is_absolute() else [Path.cwd() / stored, root / stored, root.parent / stored]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Required evidence artifact not found: {value}")


def _spectrum(value: Any) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing wavelength or loss spectrum.")
    try:
        values = np.array([float(part.strip()) for part in value.split(",")], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Spectrum must be a comma-delimited numeric array.") from exc
    if values.size < 16 or not np.all(np.isfinite(values)):
        raise ValueError("Spectrum must contain at least 16 finite points.")
    return values


def _close(a: Any, b: Any, *, atol: float = 0.05) -> bool:
    try:
        left, right = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(left) and np.isfinite(right) and np.isclose(left, right, atol=atol, rtol=0.0))


def audit_comsol_iteration(iteration_dir: str | Path, *, metric_tolerance_nm: float = 0.05) -> dict[str, Any]:
    """Recompute optical metrics from archived spectra and compare verified targets.

    Requires the actual model/config files and hash-bound COMSOL manifest. An
    audit pass checks consistency, NOT physical correctness, solver execution,
    mesh convergence, sensor performance, or experimental authenticity.
    """
    root = Path(iteration_dir).resolve()
    errors: list[str] = []
    hashes: dict[str, str] = {}
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_type": "archived_comsol_spectrum_consistency",
        "consistency_ok": False,
        "physical_correctness_verified": False,
        "errors": errors,
        "source_hashes": hashes,
        "spectrum_rows_recomputed": 0,
        "target_count": 0,
        "accepted_target_count": 0,
    }

    if not np.isfinite(metric_tolerance_nm) or metric_tolerance_nm <= 0:
        raise ValueError("metric_tolerance_nm must be finite and > 0.")

    manifest_path = root / "iteration_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be a JSON object.")
        hashes["manifest_sha256"] = sha256_file(manifest_path)
        if manifest.get("backend") != "comsol" or manifest.get("evidence_class") != "comsol_physics":
            errors.append("Manifest must identify COMSOL backend and comsol_physics evidence.")
        inputs = manifest.get("inputs", {})
        outputs = manifest.get("outputs", {})
        if not isinstance(inputs, dict) or not isinstance(outputs, dict):
            raise ValueError("Manifest inputs and outputs must be JSON objects.")
        for key, expected in (
            ("comsol_model", "comsol_model_sha256"),
            ("comsol_config", "comsol_config_sha256"),
        ):
            artifact = _artifact(root, inputs.get(key))
            actual_hash = sha256_file(artifact)
            hashes[expected] = actual_hash
            if not isinstance(inputs.get(expected), str) or actual_hash != inputs[expected]:
                errors.append(f"{key} hash mismatch.")

        paths = {
            name: _artifact(root, outputs.get(name))
            for name in ("targets_with_geometry", "simulation_results", "verification")
        }
        for name, path in paths.items():
            hashes[f"{name}_sha256"] = sha256_file(path)
        sidecar = paths["simulation_results"].with_suffix(paths["simulation_results"].suffix + ".meta.json")
        sidecar_data = json.loads(sidecar.read_text(encoding="utf-8"))
        if (
            sidecar_data.get("sha256") != hashes["simulation_results_sha256"]
            or sidecar_data.get("source") != "comsol"
            or sidecar_data.get("evidence_class") != "comsol_physics"
        ):
            errors.append("Simulation sidecar source, class, or checksum mismatch.")

        targets = pd.read_csv(paths["targets_with_geometry"]).reset_index(drop=True)
        simulation = pd.read_csv(paths["simulation_results"])
        verification = pd.read_csv(paths["verification"])
        if targets.empty or simulation.empty or verification.empty:
            raise ValueError("Target, simulation, and verification tables must be nonempty.")

        required_simulation = [
            "target_id", "sample_id", "status", "source", "evidence_class",
            "analyte_ri", "lambda_res_nm", "fwhm_nm",
            "wavelength_nm", "loss_db_per_cm", *GEOMETRY_COLUMNS,
        ]
        required_target = ["analyte_ri", *GEOMETRY_COLUMNS]
        required_verification = ["target_id", "accepted", *_NUMERIC_FIELDS]
        for name, frame, columns in (
            ("simulation", simulation, required_simulation),
            ("targets", targets, required_target),
            ("verification", verification, required_verification),
        ):
            missing = sorted(set(columns).difference(frame.columns))
            if missing:
                raise ValueError(f"{name} table is missing required columns: {missing}")

        if simulation["sample_id"].duplicated().any():
            errors.append("Duplicated simulation sample IDs.")
        if verification["target_id"].duplicated().any():
            errors.append("Duplicated verification target IDs.")
        if not simulation["source"].eq("comsol").all() or not simulation["evidence_class"].eq("comsol_physics").all():
            errors.append("Simulation includes non-COMSOL or mislabeled source rows.")

        ri_points = int(manifest["ri_points"])
        span = float(manifest["ri_span"])
        if ri_points < 3 or ri_points % 2 != 1:
            raise ValueError("Invalid RI-point count in manifest.")
        if len(targets) != int(manifest["selected_targets"]):
            errors.append("Selected-target count does not match target table.")
        if set(simulation["target_id"]) != set(range(len(targets))):
            errors.append("Simulation target IDs do not match selected target rows.")
        if set(verification["target_id"]) != set(range(len(targets))):
            errors.append("Verification target IDs do not match selected target rows.")

        recomputed = simulation.copy()
        for target_id, target in targets.iterrows():
            group = simulation.loc[simulation["target_id"].eq(target_id)]
            if len(group) != ri_points:
                errors.append(f"Target {target_id}: wrong RI sweep length.")
                continue
            for geometry in GEOMETRY_COLUMNS:
                values = pd.to_numeric(group[geometry], errors="coerce").to_numpy(dtype=float)
                if not np.all(np.isfinite(values)) or not np.all(
                    np.isclose(values, float(target[geometry]), atol=1e-9, rtol=0.0)
                ):
                    errors.append(f"Target {target_id}: {geometry} does not match requested geometry.")
            observed_ri = np.sort(pd.to_numeric(group["analyte_ri"], errors="coerce").to_numpy(dtype=float))
            expected_ri = np.sort(build_validation_ri_values(float(target["analyte_ri"]), span=span, points=ri_points))
            if not np.allclose(observed_ri, expected_ri, atol=1e-9, rtol=0.0):
                errors.append(f"Target {target_id}: RI grid differs from requested sweep.")

        for index, row in simulation.iterrows():
            if str(row["status"]) != "ok":
                continue
            try:
                metrics = extract_metrics(_spectrum(row["wavelength_nm"]), _spectrum(row["loss_db_per_cm"]))
                if not np.isfinite(metrics.fwhm_nm) or metrics.fwhm_nm <= 0:
                    raise ValueError("Resonance FWHM could not be determined.")
                for field in ("lambda_res_nm", "fwhm_nm"):
                    if not _close(row[field], getattr(metrics, field), atol=metric_tolerance_nm):
                        errors.append(f"Sample {row['sample_id']}: stored {field} differs from archived spectrum.")
                recomputed.at[index, "lambda_res_nm"] = metrics.lambda_res_nm
                recomputed.at[index, "fwhm_nm"] = metrics.fwhm_nm
                report["spectrum_rows_recomputed"] += 1
            except (TypeError, ValueError, IndexError, np.linalg.LinAlgError) as exc:
                errors.append(f"Sample {row['sample_id']}: invalid spectrum ({exc}).")
                recomputed.at[index, "status"] = "failed: spectrum audit"

        # Check the original acceptance decisions against metrics recomputed
        # from the archived spectra rather than the reported surrogate outputs.
        limits = AcceptanceThresholds(**manifest["acceptance_thresholds"])
        audited = evaluate_closed_loop_results(recomputed, limits, expected_ri_points=ri_points)
        for _, row in audited.iterrows():
            original = verification.loc[verification["target_id"].eq(row["target_id"])]
            if len(original) != 1:
                errors.append(f"Target {row['target_id']}: verification row absent or duplicated.")
                continue
            source = original.iloc[0]
            if str(source["accepted"]).strip().lower() not in {"true", "false"}:
                errors.append(f"Target {row['target_id']}: invalid accepted flag.")
                continue
            recorded_accepted = str(source["accepted"]).strip().lower() == "true"
            if recorded_accepted != bool(row["accepted"]):
                errors.append(f"Target {row['target_id']}: verification acceptance does not match recomputation.")
            for field in _NUMERIC_FIELDS:
                left, right = source[field], row[field]
                if pd.isna(left) and pd.isna(right):
                    continue
                if not _close(left, right, atol=metric_tolerance_nm if field != "linearity_r2" else 1e-5):
                    errors.append(f"Target {row['target_id']}: verification {field} mismatch.")
        report["target_count"] = int(len(audited))
        report["accepted_target_count"] = int(audited["accepted"].sum())
        if int(manifest.get("accepted_targets", -1)) != report["accepted_target_count"]:
            errors.append("Manifest accepted-target count disagrees with recomputed verification.")
    except (OSError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
        errors.append(f"Audit could not complete: {exc}")

    report["consistency_ok"] = len(errors) == 0
    return report
