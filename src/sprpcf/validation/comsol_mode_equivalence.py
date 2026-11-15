"""Independent, frozen-source comparison of COMSOL batch reload vs model reuse.

The audit recomputes data from committed checkpointed solver spectra. It can
identify numerical mode/order dependence but CANNOT authenticate external
COMSOL execution or certify Maxwell physics, mesh convergence, or manuscripts.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.simulation.resumable_campaign import (
    _checkpoint_contract, _parse_spectrum, inspect_campaign_checkpoint,
)
from sprpcf.simulation.session_provenance import audit_native_session_logs
from sprpcf.simulation.verification_campaign import (
    _check_frozen_plan, _clean_output, _read_json, _require_file,
)
from sprpcf.utils.reproducibility import sha256_file

PROTOCOL_NAME = "COMSOL_MODE_COMPARISON_PROTOCOL.json"
REPORT_NAME = "COMSOL_MODE_EQUIVALENCE_AUDIT.json"
ROWS_NAME = "COMSOL_MODE_EQUIVALENCE_ROWS.csv"
THRESHOLD_FIELDS = (
    "max_wavelength_grid_delta_nm",
    "max_resonance_delta_nm",
    "max_fwhm_delta_nm",
    "max_loss_delta_db_per_cm",
)


def _thresholds(values: dict[str, Any]) -> dict[str, float]:
    if set(values) != set(THRESHOLD_FIELDS):
        raise ValueError("Must explicitly declare all four COMSOL numeric equivalence tolerances")
    result: dict[str, float] = {}
    for key in THRESHOLD_FIELDS:
        try:
            number = float(values[key])
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid numerical comparison tolerance: {key}") from exc
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"Numerical comparison tolerance must be finite positive: {key}")
        result[key] = number
    return result


def _source_hashes(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
) -> tuple[dict[str, Any], dict[str, str]]:
    plan, _ = _check_frozen_plan(
        plan_dir, benchmark_dir, model_path, config_path
    )
    sources = {
        "campaign_plan_sha256": Path(plan_dir) / "campaign_plan.json",
        "planned_sweeps_sha256": Path(plan_dir) / "planned_sweeps.csv",
        "benchmark_manifest_sha256": Path(benchmark_dir) / "INVERSE_BENCHMARK_MANIFEST.json",
        "requests_sha256": Path(benchmark_dir) / "PHYSICS_VERIFICATION_REQUESTS.csv",
        "model_sha256": _require_file(model_path, suffix=".mph"),
        "config_sha256": _require_file(config_path),
    }
    return plan, {name: sha256_file(_require_file(source)) for name, source in sources.items()}


def freeze_mode_comparison_protocol(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
    output_dir: str | Path, *, tolerances: dict[str, Any],
) -> dict[str, Any]:
    """Freeze measured variables and tolerances before interpreting two runs.

    Creation time is descriptive only. File hashes cannot prove the protocol
    was preregistered before physical COMSOL execution.
    """
    limits = _thresholds(tolerances)
    plan, hashes = _source_hashes(plan_dir, benchmark_dir, model_path, config_path)
    out = _clean_output(output_dir)
    protocol = {
        "schema_version": 1,
        "kind": "comsol_reload_reuse_numeric_comparison_protocol",
        "frozen_input_sha256": hashes,
        "scheduled_solver_evaluations": int(plan["planned_solver_evaluations"]),
        "expected_run_modes": ["per_batch_load", "reused_loaded_model"],
        "tolerances": limits,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
        "warning": "Time and hashes do not authenticate COMSOL, nor prove preregistration.",
    }
    out.mkdir(parents=True, exist_ok=True)
    with (out / PROTOCOL_NAME).open("x", encoding="utf-8") as stream:
        json.dump(protocol, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return protocol


def _check_run(
    run_dir: str | Path, *, expected_mode: str,
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    root = Path(run_dir).resolve()
    inspected = inspect_campaign_checkpoint(
        plan_dir, benchmark_dir, model_path, config_path, root
    )
    execution = _read_json(_require_file(root / "execution_manifest.json"))
    if (execution.get("execution_mode") != "checkpointed_batches"
            or execution.get("real_comsol_backend_invoked") is not True
            or inspected.get("backend_kind") != "comsol_mph"):
        raise ValueError("Mode comparison requires completed COMSOL backend checkpoint runs")
    if (execution.get("model_session_mode", "per_batch_load") != expected_mode
            or inspected["model_session_mode"] != expected_mode):
        raise ValueError("Wrong COMSOL reload/reuse lifecycle assigned to comparison run")
    frozen, source_hashes = _source_hashes(
        plan_dir, benchmark_dir, model_path, config_path
    )
    for key in (
        "campaign_plan_sha256", "planned_sweeps_sha256",
        "model_sha256", "config_sha256",
    ):
        if execution.get(key) != source_hashes[key]:
            raise ValueError("Run execution metadata mismatches frozen input: " + key)
    planned = int(execution.get("scheduled_solver_evaluations", -1))
    if planned != frozen["planned_solver_evaluations"]:
        raise ValueError("Run execution sample count differs from frozen plan")
    completed = inspected["committed_solver_evaluations"]
    if (inspected["remaining_solver_evaluations"] != 0
            or planned != completed
            or execution.get("successful_solver_evaluations", -1) +
               execution.get("failed_solver_evaluations", -1) != completed
            or inspected["failed_solver_evaluations"] != execution.get("failed_solver_evaluations")):
        raise ValueError("Mode comparison run has incomplete/altered sample denominator")
    source = _require_file(root / "sweeps.csv")
    digest = sha256_file(source)
    if (digest != inspected["canonical_merged_sweeps_sha256"]
            or digest != execution.get("sweeps_sha256")
            or execution.get("checkpoint_manifest_sha256") != inspected["checkpoint_manifest_sha256"]
            or execution.get("chunk_evidence") != inspected["chunk_evidence"]
            or execution.get("session_evidence") != inspected["session_evidence"]):
        raise ValueError("Mode comparison run disagrees with immutable checkpoint evidence")
    if any(row["status"] != "completed" for row in inspected["session_evidence"]):
        raise ValueError("Mode comparison requires fully closed completed solver sessions")
    rows = pd.read_csv(source, dtype={"request_id": str})
    if len(rows) != completed:
        raise ValueError("Mode comparison has missing or duplicated final sweeps")
    # Missing native logs cannot be silently ignored. The numerical
    # comparison can still diagnose discrepancies, but evidence is incomplete.
    native_logs_complete = False
    native_log_error: str | None = None
    try:
        plan, _ = _check_frozen_plan(
            plan_dir, benchmark_dir, model_path, config_path
        )
        contract = _checkpoint_contract(
            Path(plan_dir), plan,
            _read_json(_require_file(root / "checkpoint_manifest.json"))["batch_size"],
            real_backend=True,
            reuse_loaded_model=expected_mode == "reused_loaded_model",
        )
        receipts = audit_native_session_logs(
            root, contract, inspected["chunk_evidence"]
        )
        native_logs_complete = bool(receipts)
    except (ValueError, OSError, KeyError) as exc:
        native_log_error = str(exc)
    return rows, inspected, {
        "execution_manifest_sha256": sha256_file(root / "execution_manifest.json"),
        "sweeps_sha256": digest,
        "checkpoint_manifest_sha256": inspected["checkpoint_manifest_sha256"],
        "sessions": len(inspected["session_evidence"]),
        "native_session_logs_complete": native_logs_complete,
        "native_session_log_error": native_log_error,
        "failed_samples": inspected["failed_solver_evaluations"],
    }


def compare_comsol_mode_results(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
    protocol_path: str | Path,
    reload_run_dir: str | Path, reuse_run_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Compare all paired data from checked commits; never synthesize spectra."""
    protocol_file = _require_file(protocol_path)
    protocol = _read_json(protocol_file)
    plan, sources = _source_hashes(
        plan_dir, benchmark_dir, model_path, config_path
    )
    if (protocol.get("schema_version") != 1
            or protocol.get("kind") != "comsol_reload_reuse_numeric_comparison_protocol"
            or protocol.get("frozen_input_sha256") != sources
            or protocol.get("expected_run_modes") !=
               ["per_batch_load", "reused_loaded_model"]
            or protocol.get("scheduled_solver_evaluations") != plan["planned_solver_evaluations"]
            or protocol.get("origin_authenticity_verified") is not False
            or protocol.get("scientific_submission_ready") is not False):
        raise ValueError("Mode equivalence protocol/source plan mismatch")
    limits = _thresholds(protocol.get("tolerances", {}))
    if Path(reload_run_dir).resolve() == Path(reuse_run_dir).resolve():
        raise ValueError("COMSOL mode comparison requires two distinct execution directories")
    reload, reload_inspect, reload_meta = _check_run(
        reload_run_dir, expected_mode="per_batch_load",
        plan_dir=plan_dir, benchmark_dir=benchmark_dir,
        model_path=model_path, config_path=config_path,
    )
    reused, reuse_inspect, reuse_meta = _check_run(
        reuse_run_dir, expected_mode="reused_loaded_model",
        plan_dir=plan_dir, benchmark_dir=benchmark_dir,
        model_path=model_path, config_path=config_path,
    )
    if (len(reload) != len(reused)
            or len(reload) != plan["planned_solver_evaluations"]
            or len(reload) == 0):
        raise ValueError("Reload/reuse runs have unequal frozen sample coverage")
    # Canonical Stage7/8 sweeps.csv deliberately omits sample_id. Its
    # absolute ID is the zero-based frozen row position, already rechecked
    # against checkpoint chunks by inspect_campaign_checkpoint above.
    fields = (
        "request_id", "analyte_ri", "pitch_um",
        "d_over_lambda", "metal_thickness_nm", "channel_radius_um",
        "source", "evidence_class",
    )
    if not set(fields).issubset(reload.columns) or not set(fields).issubset(reused.columns):
        raise ValueError("Missing frozen optical identities in compared output")
    for field in fields:
        if not reload[field].equals(reused[field]):
            raise ValueError("Reload/reuse sample alignment/source mismatch: " + field)
    out = _clean_output(output_dir)
    comparisons: list[dict[str, Any]] = []
    for idx in range(len(reload)):
        base = reload.iloc[idx]
        variant = reused.iloc[idx]
        item: dict[str, Any] = {
            "sample_id": idx,
            "request_id": str(base["request_id"]),
            "analyte_ri": float(base["analyte_ri"]),
            "reload_status": str(base["status"]),
            "reuse_status": str(variant["status"]),
            "grid_delta_nm": "",
            "resonance_delta_nm": "",
            "fwhm_delta_nm": "",
            "max_loss_delta_db_per_cm": "",
            "passed": False,
            "failure_reason": "",
        }
        if base["status"] != "ok" or variant["status"] != "ok":
            item["failure_reason"] = "one_or_both_solvers_failed"
        else:
            wave_a = _parse_spectrum(base["wavelength_nm"], "reload wavelength")
            wave_b = _parse_spectrum(variant["wavelength_nm"], "reuse wavelength")
            loss_a = _parse_spectrum(base["loss_db_per_cm"], "reload loss")
            loss_b = _parse_spectrum(variant["loss_db_per_cm"], "reuse loss")
            if (wave_a.shape != wave_b.shape or loss_a.shape != loss_b.shape
                    or wave_a.shape != loss_a.shape):
                item["failure_reason"] = "wavelength_grid_or_spectrum_length_changed"
            else:
                grid = float(np.max(np.abs(wave_a - wave_b)))
                resonance = abs(float(base["lambda_res_nm"]) -
                                float(variant["lambda_res_nm"]))
                fwhm = abs(float(base["fwhm_nm"]) - float(variant["fwhm_nm"]))
                loss = float(np.max(np.abs(loss_a - loss_b)))
                item.update({
                    "grid_delta_nm": grid,
                    "resonance_delta_nm": resonance,
                    "fwhm_delta_nm": fwhm,
                    "max_loss_delta_db_per_cm": loss,
                })
                exceeded = [
                    key for key, delta, threshold in (
                        ("grid", grid, limits["max_wavelength_grid_delta_nm"]),
                        ("resonance", resonance, limits["max_resonance_delta_nm"]),
                        ("fwhm", fwhm, limits["max_fwhm_delta_nm"]),
                        ("loss", loss, limits["max_loss_delta_db_per_cm"]),
                    )
                    if not math.isfinite(delta) or delta > threshold
                ]
                item["failure_reason"] = (
                    "tolerance_exceeded:" + ",".join(exceeded) if exceeded else ""
                )
                item["passed"] = not exceeded
        comparisons.append(item)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / ROWS_NAME
    with rows_path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)
    passed = sum(row["passed"] is True for row in comparisons)
    numeric_equal = passed == len(comparisons)
    logs_ok = (reload_meta["native_session_logs_complete"]
               and reuse_meta["native_session_logs_complete"])
    report = {
        "schema_version": 1,
        "kind": "comsol_reload_reuse_numeric_equivalence_audit",
        "protocol_sha256": sha256_file(protocol_file),
        "frozen_input_sha256": sources,
        "tolerances": limits,
        "paired_evaluations": len(comparisons),
        "paired_passed": passed,
        "paired_failed": len(comparisons) - passed,
        "all_paired_spectra_numerically_equivalent": numeric_equal,
        "native_session_logs_complete": bool(logs_ok),
        "operational_evidence_complete": bool(numeric_equal and logs_ok),
        "reloaded_run": reload_meta,
        "reused_run": reuse_meta,
        "row_comparison_sha256": sha256_file(rows_path),
        "row_comparison_file": ROWS_NAME,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
        "qualification": "numerical_mode_consistency_diagnostic_only",
        "warnings": [
            "Matching local byte hashes and logs do not authenticate actual COMSOL origin.",
            "Model-reuse equivalence cannot establish mode tracking, physics validity or numerical convergence.",
            "A protocol file timestamp cannot prove it was preregistered before observing runs.",
            "Raw spectra and numerical tolerance changes must be independently reviewed.",
        ],
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out / REPORT_NAME).write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return report
