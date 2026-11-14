"""Source-bound RI-response diagnostics for a real COMSOL pilot campaign.

A single successful center-RI spectrum cannot establish that the analyte RI
couples to the photonic model. Probe low/center/high RI plus a repeat center
solve before launching an expensive campaign. The resulting diagnostic is
NOT a scientific sensitivity/FOM estimate or a COMSOL authenticity proof.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from sprpcf.simulation.comsol_sweep import run_comsol_geometries
from sprpcf.simulation.model_gate import validate_live_pilot_report
from sprpcf.simulation.resumable_campaign import _normalize_chunk_rows, _parse_spectrum
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.verification_campaign import (
    _check_frozen_plan, _clean_output, _now, _require_file,
)
from sprpcf.utils.reproducibility import sha256_file

_GEOMETRY = (
    "d_over_lambda", "pitch_um", "metal_thickness_nm",
    "channel_radius_um", "analyte_ri",
)
_PROBE_NAMES = ("low_ri", "center_ri", "high_ri", "center_repeat")


def _positive(value: float, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite positive") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be finite positive")
    return number


def _thresholds(
    min_shift_nm: float, max_repeat_nm: float, min_diagnostic_r2: float
) -> dict[str, float]:
    shift = _positive(min_shift_nm, "min_shift_nm")
    repeat = _positive(max_repeat_nm, "max_repeat_nm")
    r2 = float(min_diagnostic_r2)
    if not math.isfinite(r2) or not 0 < r2 <= 1:
        raise ValueError("min_diagnostic_r2 must be finite and in (0, 1]")
    return {
        "min_shift_nm": shift,
        "max_repeat_nm": repeat,
        "min_diagnostic_r2": r2,
    }


def _select_probe(plan: dict[str, Any], planned: pd.DataFrame
                  ) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    ri_points = int(plan["ri_points"])
    if ri_points < 3 or ri_points % 2 != 1 or len(planned) < ri_points:
        raise ValueError("The frozen Stage 7 plan needs an odd RI sweep of >=3 points")
    first = planned.iloc[:ri_points].copy()
    if first["request_id"].nunique() != 1:
        raise ValueError("First request RI sweep has inconsistent identities")
    positions = (0, ri_points // 2, ri_points - 1, ri_points // 2)
    original = first.iloc[list(positions)].reset_index(drop=True)
    selected = original.copy()
    selected["sample_id"] = list(range(4))  # MPh returns local call IDs 0..3
    requests = []
    for idx, (_, row) in enumerate(original.iterrows()):
        requests.append({
            "probe": _PROBE_NAMES[idx],
            "frozen_sample_id": int(row["sample_id"]),
            "request_id": str(row["request_id"]),
            "analyte_ri": float(row["analyte_ri"]),
        })
    if not requests[0]["analyte_ri"] < requests[1]["analyte_ri"] < requests[2]["analyte_ri"]:
        raise ValueError("Frozen RI probe must be strictly increasing")
    if requests[3]["analyte_ri"] != requests[1]["analyte_ri"]:
        raise ValueError("Repeated center-RI point changed")
    return requests, selected


def _metrics(rows: list[dict[str, Any]], conditions: dict[str, float]
             ) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    if len(rows) != 4 or any(row["status"] != "ok" for row in rows):
        return {}, ["One or more RI probe solves failed or were incomplete"]
    wavelengths = [_parse_spectrum(row["wavelength_nm"], "wavelength_nm") for row in rows]
    losses = [_parse_spectrum(row["loss_db_per_cm"], "loss_db_per_cm") for row in rows]
    primary = wavelengths[0]
    spacing = float(np.median(np.diff(primary)))
    if spacing <= 0:
        errors.append("Spectral wavelength grid is not increasing")
    for idx, w in enumerate(wavelengths):
        if w.shape != primary.shape or not np.allclose(w, primary, atol=1e-6, rtol=0):
            errors.append(f"RI probe {idx} changed the wavelength sampling grid")
    lambdas = np.array([float(row["lambda_res_nm"]) for row in rows], dtype=float)
    widths = np.array([float(row["fwhm_nm"]) for row in rows], dtype=float)
    ri = np.array([float(row["analyte_ri"]) for row in rows], dtype=float)
    if not np.isfinite(lambdas).all() or not np.isfinite(widths).all():
        errors.append("Nonfinite resonance peak or FWHM")
    for index, wavelength in enumerate(wavelengths):
        if lambdas[index] <= wavelength[1] or lambdas[index] >= wavelength[-2]:
            errors.append(f"RI probe {index} has a truncated/boundary resonance")
        if widths[index] < 3 * spacing:
            errors.append(f"RI probe {index} has an under-resolved resonance FWHM")
        if np.min(losses[index]) < -1e-9:
            errors.append(f"RI probe {index} reported physically negative confinement loss")
    shift = float(lambdas[2] - lambdas[0])
    repeat_delta = float(abs(lambdas[3] - lambdas[1]))
    half_grid = spacing / 2.0
    if abs(shift) < max(conditions["min_shift_nm"], half_grid):
        errors.append("Low-to-high RI shift is below preregistered detectability/resolution")
    if repeat_delta > conditions["max_repeat_nm"]:
        errors.append("Repeated center-RI solve was not numerically stable")
    if abs(widths[3] - widths[1]) > conditions["max_repeat_nm"]:
        errors.append("Repeated center-RI FWHM changed beyond numeric tolerance")
    if not ((lambdas[0] < lambdas[1] < lambdas[2])
            or (lambdas[0] > lambdas[1] > lambdas[2])):
        errors.append("Resonance does not track monotonically with analyte RI")
    centers = ri[:3]
    peak_values = lambdas[:3]
    fitted = np.polyval(np.polyfit(centers, peak_values, 1), centers)
    denominator = float(np.sum((peak_values - np.mean(peak_values)) ** 2))
    r2 = (1 - float(np.sum((peak_values - fitted) ** 2)) / denominator
          if denominator > 0 else 0.0)
    if r2 < conditions["min_diagnostic_r2"]:
        errors.append("Three-point RI response fails the preregistered diagnostic linearity threshold")
    return {
        "low_ri_nm": float(lambdas[0]),
        "center_ri_nm": float(lambdas[1]),
        "high_ri_nm": float(lambdas[2]),
        "repeat_center_ri_nm": float(lambdas[3]),
        "low_to_high_resonance_shift_nm": shift,
        "center_retest_peak_deviation_nm": repeat_delta,
        "spectral_grid_spacing_nm": spacing,
        "three_point_diagnostic_r2": r2,
        "n_numeric_evaluations": len(rows),
        "diagnostic_only_not_for_publication": True,
    }, errors


def probe_comsol_ri_response(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
    pilot_readiness_report: str | Path, output_dir: str | Path, *,
    min_shift_nm: float, max_repeat_nm: float, min_diagnostic_r2: float = 0.90,
    runner: Callable[[Path, Path, Sequence[Geometry]], pd.DataFrame] | None = None,
    check_pilot: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run 4 actual solver calls only with a prior source-matched one-RI pilot.

    Test dependency injection deliberately prevents promotion of test arrays.
    """
    conditions = _thresholds(min_shift_nm, max_repeat_nm, min_diagnostic_r2)
    model = _require_file(model_path, suffix=".mph")
    config = _require_file(config_path)
    out = _clean_output(output_dir)
    plan, planned = _check_frozen_plan(plan_dir, benchmark_dir, model, config)
    checker = check_pilot or validate_live_pilot_report
    # The previous single-center pilot is a required prerequisite.
    preflight = checker(pilot_readiness_report, plan_dir, benchmark_dir, model, config)
    if not isinstance(preflight, dict) or not preflight.get("operational_pilot_verified"):
        raise ValueError("Source-bound one-RI pilot readiness has not passed")
    probes, selected = _select_probe(plan, planned)
    # Freeze RI choices, numeric thresholds and input sources BEFORE solver output.
    out.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": 1,
        "type": "frozen_comsol_ri_response_protocol",
        "campaign_plan_sha256": sha256_file(Path(plan_dir) / "campaign_plan.json"),
        "planned_sweeps_sha256": sha256_file(Path(plan_dir) / "planned_sweeps.csv"),
        "pilot_readiness_report_sha256": sha256_file(_require_file(pilot_readiness_report)),
        "model_sha256": sha256_file(model),
        "config_sha256": sha256_file(config),
        "frozen_samples": probes,
        "preregistered_detection_thresholds": conditions,
        "scientific_submission_ready": False,
        "registered_before_solver_at_utc": _now(),
    }
    with (out / "RI_PROBE_PROTOCOL.json").open("x", encoding="utf-8") as writer:
        json.dump(protocol, writer, indent=2, sort_keys=True, allow_nan=False)
        writer.write("\n")
    real_backend = runner is None and check_pilot is None
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_type": "frozen_comsol_ri_response_sanity",
        "observed_backend": "comsol_mph_api" if real_backend else "software_fixture",
        "operational_pilot_verified": bool(preflight["operational_pilot_verified"]),
        "pilot_readiness_report_sha256": sha256_file(_require_file(pilot_readiness_report)),
        "campaign_plan_sha256": sha256_file(Path(plan_dir) / "campaign_plan.json"),
        "model_sha256": sha256_file(model),
        "config_sha256": sha256_file(config),
        "planned_sweeps_sha256": sha256_file(Path(plan_dir) / "planned_sweeps.csv"),
        "request_id": probes[0]["request_id"],
        "frozen_samples": probes,
        "preregistered_detection_thresholds": conditions,
        "preregistered_protocol_sha256": sha256_file(out / "RI_PROBE_PROTOCOL.json"),
        "solver_called": False,
        "numerical_ri_response_consistent": False,
        "ready_for_full_solver_run": False,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
        "warnings": [
            "Four-call RI diagnostic with repeated center is NOT full sensitivity/FOM or physical proof.",
            "A local JSON/hash ledger is not external COMSOL-origin authentication.",
            "Retain authentic native solver logs, mesh refinement, materials/PML review separately.",
        ],
        "errors": [],
        "started_at_utc": _now(),
    }
    saved_rows: list[dict[str, Any]] = []
    try:
        geoms = [
            Geometry(**{name: float(row[name]) for name in _GEOMETRY})
            for _, row in selected.iterrows()
        ]
        solver = runner or run_comsol_geometries
        report["solver_called"] = True
        raw = solver(model, config, geoms)
        # Source identity is derived from the runner branch, not editable CSV fields.
        normalized = _normalize_chunk_rows(
            raw, selected, start=0,
            backend="comsol_mph" if real_backend else "software_fixture",
            local_ids=True,
        )
        saved_rows = [
            {"probe": p["probe"], "frozen_sample_id": p["frozen_sample_id"], **data}
            for p, data in zip(probes, normalized)
        ]
        diagnostic, failures = _metrics(saved_rows, conditions)
        report["diagnostic"] = diagnostic
        report["errors"].extend(failures)
        report["numerical_ri_response_consistent"] = not failures
    except (Exception,) as exc:
        report["errors"].append("COMSOL RI response probe failed: " + str(exc))
    report["ready_for_full_solver_run"] = bool(
        real_backend and report["numerical_ri_response_consistent"] and not report["errors"]
    )
    out.mkdir(parents=True, exist_ok=True)
    if saved_rows:
        filename = out / "RI_PROBE_SPECTRA.csv"
        columns = list(saved_rows[0])
        with filename.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(saved_rows)
        report["raw_spectra_sha256"] = sha256_file(filename)
        report["raw_spectra_file"] = filename.name
    report_file = out / "COMSOL_RI_RESPONSE_READINESS.json"
    report_file.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report


def validate_ri_response_report(
    report_path: str | Path, plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path, pilot_report: str | Path,
) -> dict[str, Any]:
    """Recalculate RI response using the original source-bound frozen grid."""
    record = _require_file(report_path)
    report = json.loads(record.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("RI response readiness report must be an object")
    expected_flags = {
        "schema_version": 1,
        "audit_type": "frozen_comsol_ri_response_sanity",
        "observed_backend": "comsol_mph_api",
        "solver_called": True,
        "numerical_ri_response_consistent": True,
        "ready_for_full_solver_run": True,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
    }
    for name, expected in expected_flags.items():
        if report.get(name) != expected:
            raise ValueError(f"RI response source-bound gate failed: {name}")
    if report.get("errors"):
        raise ValueError("RI response has scientific/solver blockers")
    plan, planned = _check_frozen_plan(plan_dir, benchmark_dir, model_path, config_path)
    sources = {
        "model_sha256": _require_file(model_path, suffix=".mph"),
        "config_sha256": _require_file(config_path),
        "campaign_plan_sha256": _require_file(Path(plan_dir) / "campaign_plan.json"),
        "planned_sweeps_sha256": _require_file(Path(plan_dir) / "planned_sweeps.csv"),
        "pilot_readiness_report_sha256": _require_file(pilot_report),
    }
    for key, source in sources.items():
        if report.get(key) != sha256_file(source):
            raise ValueError("RI response report input hash mismatch: " + key)
    requested, expected = _select_probe(plan, planned)
    if report.get("frozen_samples") != requested:
        raise ValueError("RI diagnostic probe identities differ from frozen RI plan")
    frozen_protocol_path = _require_file(record.parent / "RI_PROBE_PROTOCOL.json")
    if report.get("preregistered_protocol_sha256") != sha256_file(frozen_protocol_path):
        raise ValueError("RI diagnostic pre-run protocol hash mismatch")
    protocol = json.loads(frozen_protocol_path.read_text(encoding="utf-8"))
    if (not isinstance(protocol, dict)
            or protocol.get("type") != "frozen_comsol_ri_response_protocol"
            or protocol.get("frozen_samples") != requested
            or protocol.get("preregistered_detection_thresholds")
            != report.get("preregistered_detection_thresholds")
            or protocol.get("scientific_submission_ready") is not False):
        raise ValueError("RI response thresholds/samples changed from pre-solver protocol")
    for name, file in sources.items():
        if protocol.get(name) != sha256_file(file):
            raise ValueError("RI response protocol source hash mismatch: " + name)
    if report.get("raw_spectra_file") != "RI_PROBE_SPECTRA.csv":
        raise ValueError("RI response raw spectrum file has an unexpected path")
    raw = _require_file(record.parent / "RI_PROBE_SPECTRA.csv")
    if report.get("raw_spectra_sha256") != sha256_file(raw):
        raise ValueError("RI response spectra SHA-256 mismatch")
    candidate = pd.read_csv(raw, dtype={"request_id": str})
    if len(candidate) != 4 or candidate["probe"].tolist() != list(_PROBE_NAMES):
        raise ValueError("RI response spectrum sample coverage is incomplete")
    if candidate["frozen_sample_id"].astype(int).tolist() != [
        p["frozen_sample_id"] for p in requested
    ]:
        raise ValueError("RI response original sample IDs have changed")
    canonical = _normalize_chunk_rows(
        candidate, expected, start=0, backend="comsol_mph", local_ids=True,
    )
    values = [{"probe": req["probe"], "frozen_sample_id": req["frozen_sample_id"],
               **row} for req, row in zip(requested, canonical)]
    conditions = report.get("preregistered_detection_thresholds")
    if not isinstance(conditions, dict) or set(conditions) != {
        "min_shift_nm", "max_repeat_nm", "min_diagnostic_r2"
    }:
        raise ValueError("RI response diagnostic thresholds missing")
    fixed = _thresholds(**conditions)
    recalculated, errors = _metrics(values, fixed)
    if errors or recalculated != report.get("diagnostic"):
        raise ValueError("RI response diagnostic changed or fails original thresholds")
    return {
        "ri_response_operationally_consistent": True,
        "response_report_sha256": sha256_file(record),
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
    }
