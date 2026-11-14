"""Crash-resilient, hash-bound batches for real local COMSOL verification campaigns.

The production runner calls the genuine mph/COMSOL backend. An injected runner
is for controlled software tests only. Neither execution nor file hashes
authenticate the external solver or the underlying electromagnetic physics.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.simulation.comsol_sweep import run_comsol_geometries
from sprpcf.simulation.metrics import extract_metrics
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.verification_campaign import (
    SWEEP_FIELDS, _check_frozen_plan, _clean_output, _finite, _now, _read_json, _require_file,
)
from sprpcf.utils.reproducibility import sha256_file

CHUNK_FIELDS = ("sample_id", *SWEEP_FIELDS)
CHECKPOINT_NAME = "checkpoint_manifest.json"
PROGRESS_NAME = "campaign_progress.json"
LOCK_NAME = ".active-comsol-campaign.lock"


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    part = path.with_name(path.name + ".partial")
    with part.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


def _atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    part = path.with_name(path.name + ".partial")
    with part.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CHUNK_FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(part, path)


def _checkpoint_contract(
    plan_dir: Path, plan: dict[str, Any], batch_size: int, real_backend: bool,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "comsol_checkpointed_campaign",
        "plan_sha256": sha256_file(_require_file(plan_dir / "campaign_plan.json")),
        "planned_sweeps_sha256": plan["planned_sweeps_sha256"],
        "model_sha256": plan["model_sha256"],
        "config_sha256": plan["config_sha256"],
        "scheduled_solver_evaluations": plan["planned_solver_evaluations"],
        "batch_size": batch_size,
        "backend_kind": "comsol_mph" if real_backend else "injected_software_fixture",
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
    }


def _batch_name(start: int, stop: int) -> str:
    return f"batch_{start:08d}_{stop:08d}"


def _geometry(row: pd.Series) -> Geometry:
    instance = Geometry(
        d_over_lambda=float(row["d_over_lambda"]),
        pitch_um=float(row["pitch_um"]),
        metal_thickness_nm=float(row["metal_thickness_nm"]),
        channel_radius_um=float(row["channel_radius_um"]),
        analyte_ri=float(row["analyte_ri"]),
    )
    instance.validate()
    return instance


def _parse_spectrum(value: Any, name: str) -> np.ndarray:
    if not isinstance(value, str):
        raise ValueError(f"{name}: missing CSV spectrum")
    try:
        result = np.array([float(part.strip()) for part in value.split(",")], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}: invalid numeric spectrum") from exc
    if len(result) < 16 or not np.isfinite(result).all():
        raise ValueError(f"{name}: require at least 16 finite samples")
    return result


def _normalize_chunk_rows(
    raw: pd.DataFrame, planned: pd.DataFrame, *,
    start: int, backend: str, local_ids: bool,
) -> list[dict[str, Any]]:
    if not isinstance(raw, pd.DataFrame) or len(raw) != len(planned):
        raise ValueError("COMSOL batch did not return the exact planned number of samples")
    required = (
        ("sample_id", "status", "analyte_ri", "d_over_lambda",
         "pitch_um", "metal_thickness_nm", "channel_radius_um")
        if local_ids else CHUNK_FIELDS
    )
    if not set(required).issubset(raw.columns) or raw.columns.has_duplicates:
        raise ValueError("COMSOL batch rows missing required fields or have duplicate headings")
    rows: list[dict[str, Any]] = []
    expected_source = "comsol" if backend == "comsol_mph" else "test_fixture"
    evidence_class = "comsol_physics" if backend == "comsol_mph" else "software_only"
    for offset, (_, measured) in enumerate(raw.iterrows()):
        original = planned.iloc[offset]
        sample_id = int(original["sample_id"])
        if sample_id != start + offset or measured["sample_id"] != (offset if local_ids else sample_id):
            raise ValueError("Solver batch sample IDs do not match frozen plan")
        status = str(measured["status"]).strip()
        if status != "ok" and not (status.startswith("failed:") and status[7:].strip()):
            raise ValueError("Unsupported or empty COMSOL solver status")
        for key in ("analyte_ri", "d_over_lambda", "pitch_um",
                    "metal_thickness_nm", "channel_radius_um"):
            measured_number = _finite(measured[key], key)
            expected_number = _finite(original[key], key)
            if not math.isclose(measured_number, expected_number, abs_tol=1e-9, rel_tol=0):
                raise ValueError(f"COMSOL batch returned mismatched frozen {key}")
        if not local_ids:
            for key in ("request_id", "source", "evidence_class"):
                expected = (str(original["request_id"]) if key == "request_id" else
                            expected_source if key == "source" else evidence_class)
                if str(measured[key]) != expected:
                    raise ValueError(f"Stored batch {key} mismatches fixed source contract")
        item: dict[str, Any] = {
            "sample_id": sample_id, "request_id": str(original["request_id"]),
            "analyte_ri": float(original["analyte_ri"]),
            **{key: float(original[key]) for key in (
                "d_over_lambda", "pitch_um", "metal_thickness_nm", "channel_radius_um",
            )},
            "status": status, "source": expected_source,
            "evidence_class": evidence_class,
            "wavelength_nm": "", "loss_db_per_cm": "",
            "lambda_res_nm": "", "fwhm_nm": "",
        }
        if status == "ok":
            wavelength = _parse_spectrum(measured.get("wavelength_nm"), "wavelength_nm")
            loss = _parse_spectrum(measured.get("loss_db_per_cm"), "loss_db_per_cm")
            if wavelength.shape != loss.shape or not np.all(np.diff(wavelength) > 0):
                raise ValueError("Nonmonotonic or mismatched archived wavelengths/losses")
            metrics = extract_metrics(wavelength, loss)
            if (not np.isfinite(metrics.fwhm_nm) or metrics.fwhm_nm <= 0
                    or not math.isclose(float(metrics.lambda_res_nm),
                                        _finite(measured.get("lambda_res_nm"), "lambda_res_nm"),
                                        rel_tol=0, abs_tol=0.05)
                    or not math.isclose(float(metrics.fwhm_nm),
                                        _finite(measured.get("fwhm_nm"), "fwhm_nm"),
                                        rel_tol=0, abs_tol=0.05)):
                raise ValueError("Stored COMSOL spectrum and resonance/FWHM values disagree")
            item.update(
                wavelength_nm=str(measured["wavelength_nm"]),
                loss_db_per_cm=str(measured["loss_db_per_cm"]),
                lambda_res_nm=float(metrics.lambda_res_nm),
                fwhm_nm=float(metrics.fwhm_nm),
            )
        elif not local_ids:
            for field in ("wavelength_nm", "loss_db_per_cm", "lambda_res_nm", "fwhm_nm"):
                value = measured[field]
                if pd.notna(value) and str(value).strip():
                    raise ValueError(f"Failed solver row carries a misleading physical result: {field}")
        rows.append(item)
    return rows


def _read_committed_chunks(
    output: Path, planned: pd.DataFrame, contract: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    chunk_root = output / "chunks"
    if not chunk_root.is_dir() or chunk_root.is_symlink():
        raise ValueError("Missing safe checkpoint chunk directory")
    orphan = [p.name for p in chunk_root.iterdir() if (
        p.is_symlink() or p.suffix not in {".csv", ".json"} or p.name.endswith(".partial")
    )]
    if orphan:
        raise ValueError("Unexpected or incomplete checkpoint entries: " + ", ".join(sorted(orphan)))
    all_names = {p.name for p in chunk_root.iterdir() if p.is_file()}
    metadata: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for start in range(0, len(planned), contract["batch_size"]):
        stop = min(start + contract["batch_size"], len(planned))
        base = _batch_name(start, stop)
        csv_name, json_name = base + ".csv", base + ".json"
        if json_name not in all_names and csv_name not in all_names:
            if any(any_file in all_names for any_file in (
                _batch_name(next_start, min(next_start + contract["batch_size"], len(planned))) + ".json"
                for next_start in range(stop, len(planned), contract["batch_size"])
            )):
                raise ValueError("Checkpoint has a gap before later committed batches")
            break
        if csv_name not in all_names or json_name not in all_names:
            raise ValueError("Incomplete checkpoint pair for " + base)
        all_names.remove(csv_name)
        all_names.remove(json_name)
        manifest_path = _require_file(chunk_root / json_name)
        manifest = _read_json(manifest_path)
        expected = {
            "schema_version": 1, "kind": "committed_comsol_batch",
            "start": start, "stop": stop,
            "plan_sha256": contract["plan_sha256"],
            "backend_kind": contract["backend_kind"],
        }
        for key, value in expected.items():
            if manifest.get(key) != value:
                raise ValueError("Checkpoint provenance mismatch: " + base + " " + key)
        source = _require_file(chunk_root / csv_name)
        if manifest.get("csv_sha256") != sha256_file(source):
            raise ValueError("Checkpoint SHA-256 mismatch: " + base)
        data = pd.read_csv(source, dtype={"request_id": str})
        if not set(CHUNK_FIELDS).issubset(data.columns) or data.columns.has_duplicates:
            raise ValueError("Checkpoint schema differs from canonical batch: " + base)
        normalized = _normalize_chunk_rows(
            data, planned.iloc[start:stop], start=start,
            backend=contract["backend_kind"], local_ids=False,
        )
        if (manifest.get("successful") != sum(r["status"] == "ok" for r in normalized)
                or manifest.get("failed") != sum(r["status"] != "ok" for r in normalized)):
            raise ValueError("Checkpoint success/failure ledger mismatch: " + base)
        metadata.append({
            "start": start, "stop": stop,
            "csv_sha256": manifest["csv_sha256"],
            "manifest_sha256": sha256_file(manifest_path),
        })
        rows.extend(normalized)
    if all_names:
        raise ValueError("Unknown, uncommitted or out-of-order checkpoint files: " +
                         ", ".join(sorted(all_names)))
    return rows, metadata


def _write_progress(output: Path, contract: dict[str, Any], completed: list[dict[str, Any]],
                    metadata: list[dict[str, Any]]) -> dict[str, Any]:
    status = {
        "schema_version": 1,
        "checkpoint_contract_sha256": sha256_file(output / CHECKPOINT_NAME),
        "source_authenticity_verified": False,
        "scientific_submission_ready": False,
        "backend_kind": contract["backend_kind"],
        "scheduled_solver_evaluations": contract["scheduled_solver_evaluations"],
        "committed_solver_evaluations": len(completed),
        "successful_solver_evaluations": sum(item["status"] == "ok" for item in completed),
        "failed_solver_evaluations": sum(item["status"] != "ok" for item in completed),
        "remaining_solver_evaluations": contract["scheduled_solver_evaluations"] - len(completed),
        "committed_batches": len(metadata),
        "updated_at_utc": _now(),
    }
    _atomic_json(output / PROGRESS_NAME, status)
    return status


def inspect_campaign_checkpoint(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    """Read-only integrity check: no model loading, no solver calls, no mutation."""
    plan, planned = _check_frozen_plan(plan_dir, benchmark_dir, model_path, config_path)
    output = Path(output_dir).resolve()
    contract = _read_json(_require_file(output / CHECKPOINT_NAME))
    if not isinstance(contract.get("batch_size"), int) or contract["batch_size"] < 1:
        raise ValueError("Checkpoint contains invalid batch size")
    expected = _checkpoint_contract(Path(plan_dir), plan, contract["batch_size"],
                                    contract.get("backend_kind") == "comsol_mph")
    if contract != expected:
        raise ValueError("Checkpoint contract differs from frozen campaign inputs")
    completed, metadata = _read_committed_chunks(output, planned, expected)
    return {
        "contract_valid": True,
        "committed_solver_evaluations": len(completed),
        "remaining_solver_evaluations": len(planned) - len(completed),
        "failed_solver_evaluations": sum(item["status"] != "ok" for item in completed),
        "committed_batches": len(metadata),
        "backend_kind": expected["backend_kind"],
        "source_authenticity_verified": False,
        "scientific_submission_ready": False,
    }


def run_checkpointed_campaign(
    plan_dir: str | Path, benchmark_dir: str | Path, model_path: str | Path,
    config_path: str | Path, output_dir: str | Path, *,
    batch_size: int = 8, resume: bool = False,
    recover_stale_lock: bool = False,
    runner: Callable[[Path, Path, Sequence[Geometry]], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Persist each validated COMSOL batch; refuse mismatched resume provenance."""
    if not isinstance(batch_size, int) or not 1 <= batch_size <= 10000:
        raise ValueError("batch_size must be an integer between 1 and 10000")
    plan_dir = Path(plan_dir).resolve()
    model = _require_file(model_path, suffix=".mph")
    config = _require_file(config_path)
    plan, planned = _check_frozen_plan(plan_dir, benchmark_dir, model, config)
    real_backend = runner is None
    contract = _checkpoint_contract(plan_dir, plan, batch_size, real_backend)
    output = Path(output_dir).resolve()
    if resume:
        if not output.is_dir():
            raise ValueError("Cannot resume: prior checkpoint directory is missing")
        persisted = _read_json(_require_file(output / CHECKPOINT_NAME))
        if persisted != contract:
            raise ValueError("Cannot resume changed batch size, source, plan/model/config or backend")
        if (output / "execution_manifest.json").exists():
            raise FileExistsError("Already finalized; refuse to rerun an immutable campaign")
    else:
        _clean_output(output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "chunks").mkdir(exist_ok=False)
        _atomic_json(output / CHECKPOINT_NAME, contract)
    if (output / LOCK_NAME).exists():
        if not recover_stale_lock:
            raise RuntimeError("Campaign lock exists; refuse simultaneous execution or unsafe resume")
        (output / LOCK_NAME).unlink()
    lock = output / LOCK_NAME
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise RuntimeError("COMSOL campaign already has an active writer") from exc
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(json.dumps({"created_at_utc": _now(), "pid": os.getpid()}))
        completed, metadata = _read_committed_chunks(output, planned, contract)
        _write_progress(output, contract, completed, metadata)
        executor = run_comsol_geometries if real_backend else runner
        while len(completed) < len(planned):
            start = len(completed)
            stop = min(start + batch_size, len(planned))
            batch = planned.iloc[start:stop]
            sampled = [_geometry(row) for _, row in batch.iterrows()]
            # A missing COMSOL runtime/license raises here; no fake continuation.
            raw = executor(model, config, sampled)
            normal = _normalize_chunk_rows(
                raw, batch, start=start, backend=contract["backend_kind"], local_ids=True,
            )
            name = _batch_name(start, stop)
            path = output / "chunks" / (name + ".csv")
            manifest_path = output / "chunks" / (name + ".json")
            _atomic_csv(path, normal)
            chunk_record = {
                "schema_version": 1, "kind": "committed_comsol_batch",
                "start": start, "stop": stop,
                "plan_sha256": contract["plan_sha256"],
                "backend_kind": contract["backend_kind"],
                "csv_sha256": sha256_file(path),
                "successful": sum(item["status"] == "ok" for item in normal),
                "failed": sum(item["status"] != "ok" for item in normal),
                "committed_at_utc": _now(),
                "source_authenticity_verified": False,
            }
            _atomic_json(manifest_path, chunk_record)
            metadata.append({
                "start": start, "stop": stop,
                "csv_sha256": chunk_record["csv_sha256"],
                "manifest_sha256": sha256_file(manifest_path),
            })
            completed.extend(normal)
            _write_progress(output, contract, completed, metadata)
        # Exactly the Stage 7 output contract; usable by the original seal CLI.
        sweeps = output / "sweeps.csv"
        part = sweeps.with_name(sweeps.name + ".partial")
        with part.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=SWEEP_FIELDS)
            writer.writeheader()
            writer.writerows({key: row[key] for key in SWEEP_FIELDS} for row in completed)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(part, sweeps)
        result = {
            "schema_version": 1, "kind": "comsol_campaign_execution",
            "execution_mode": "checkpointed_batches",
            "completed_at_utc": _now(),
            "real_comsol_backend_invoked": real_backend,
            "source_authenticity_verified": False,
            "physical_correctness_verified": False,
            "scientific_submission_ready": False,
            "campaign_plan_sha256": contract["plan_sha256"],
            "planned_sweeps_sha256": plan["planned_sweeps_sha256"],
            "model_sha256": plan["model_sha256"],
            "config_sha256": plan["config_sha256"],
            "sweeps_sha256": sha256_file(sweeps),
            "checkpoint_manifest_sha256": sha256_file(output / CHECKPOINT_NAME),
            "chunk_evidence": metadata,
            "scheduled_solver_evaluations": len(completed),
            "successful_solver_evaluations": sum(row["status"] == "ok" for row in completed),
            "failed_solver_evaluations": sum(row["status"] != "ok" for row in completed),
            "native_solver_log_captured": False,
            "qualification": "execution_trace_only_unqualified",
            "remaining": [
                "Retrieve genuine native solver logs for every execution session.",
                "Provide a qualified mesh/spectral refinement archive for the same model.",
                "Obtain independent COMSOL material/PML/mode/physics review.",
            ],
        }
        _atomic_json(output / "execution_manifest.json", result)
        return result
    finally:
        # Only an actual writer that acquired the lock removes it. Interrupted
        # hard-killed processes require explicit --recover-stale-lock.
        lock.unlink(missing_ok=True)
