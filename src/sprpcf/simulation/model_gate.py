"""Static and live COMSOL model-readiness checks before expensive optical sweeps.

A passed gate proves only that a given model is loadable, has the requested
parameters/study and can return one internally consistent solver spectrum.
It does NOT establish authenticated COMSOL execution, scientifically correct
Maxwell physics, adequate mesh/PML, dispersion or paper-ready results.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from sprpcf.simulation.comsol_sweep import _load_config, _load_mph, run_comsol_geometries
from sprpcf.simulation.resumable_campaign import _normalize_chunk_rows
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.verification_campaign import _check_frozen_plan, _clean_output, _now, _require_file
from sprpcf.utils.reproducibility import sha256_file

_PARAMETERS = (
    "d_over_lambda", "pitch_um", "metal_thickness_nm",
    "analyte_ri", "channel_radius_um",
)
_CONFIG_TEXT = ("study", "wavelength_expression", "loss_expression")
_CONFIG_SCALE = ("wavelength_scale_to_nm", "loss_scale_to_db_per_cm")


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def _config_checks(config_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config = _load_config(config_path)
    checks: list[dict[str, Any]] = []
    for key in _CONFIG_TEXT:
        text = config.get(key)
        ok = (isinstance(text, str) and bool(text.strip())
              and "REPLACE" not in text.upper() and "TODO" not in text.upper())
        checks.append(_check("config:" + key, ok, "explicit non-placeholder expression/study required"))
    for key in _CONFIG_SCALE:
        value = config.get(key)
        try:
            scale = float(value)
        except (TypeError, ValueError):
            scale = float("nan")
        checks.append(_check("config:" + key, math.isfinite(scale) and scale > 0,
                             "explicit finite positive conversion to nm or dB/cm required"))
    limits = config.get("expected_wavelength_nm")
    valid_limits = isinstance(limits, (tuple, list)) and len(limits) == 2
    if valid_limits:
        try:
            low, high = float(limits[0]), float(limits[1])
            valid_limits = all(map(math.isfinite, [low, high])) and 0 < low < high <= 1e7
        except (TypeError, ValueError):
            valid_limits = False
    checks.append(_check("config:expected_wavelength_nm", valid_limits,
                         "must be two increasing finite positive wavelength bounds in nm"))
    is_template = ".example." in config_path.name.lower()
    checks.append(_check("config:is_not_template", not is_template,
                         "template example files cannot qualify an actual research campaign"))
    return config, checks


def _live_checks(model: Any, cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    parameters = model.parameters()
    if not isinstance(parameters, dict):
        raise ValueError("COMSOL model.parameters() must return a parameter dictionary")
    missing = sorted(set(_PARAMETERS) - set(parameters))
    checks.append(_check("model:required_parameters", not missing, f"missing={missing}"))
    studies = list(model.studies())
    checks.append(_check("model:configured_study", cfg.get("study") in studies,
                         f"configured={cfg.get('study')!r}; available={studies}"))
    for item, name in (
        ("model:geometry_sequences", "geometries"),
        ("model:mesh_sequences", "meshes"),
        ("model:physics_interfaces", "physics"),
        ("model:material_definitions", "materials"),
    ):
        members = list(getattr(model, name)())
        checks.append(_check(item, bool(members), f"declared_count={len(members)}"))
        metadata[name] = members
    problems = list(model.problems())
    checks.append(_check("model:no_reported_problems", not problems,
                         f"COMSOL model reports {len(problems)} problem(s)"))
    metadata["studies"] = studies
    metadata["model_version_when_saved"] = str(model.version())
    metadata["problems_count"] = len(problems)
    # Metadata presence is NOT a review of physical settings or numerical convergence.
    return checks, metadata


def audit_comsol_model_readiness(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
    output_dir: str | Path, *, live: bool = False, pilot: bool = False,
    client_factory: Callable[[], Any] | None = None,
    geometry_runner: Callable[[Path, Path, list[Geometry]], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Write a blocker-first JSON report. Pilot explicitly solves one central RI.

    A test-injected client or runner is always classified as software-fixture.
    Report files intentionally contain no fictional physical performance.
    """
    if pilot and not live:
        raise ValueError("A solver pilot requires --live; cannot infer results from static files")
    plan_root, bench_root = Path(plan_dir).resolve(), Path(benchmark_dir).resolve()
    model = _require_file(model_path, suffix=".mph")
    config_path = _require_file(config_path)
    output = _clean_output(output_dir)
    # This rechecks the frozen Stage 4 benchmark, plan, model and config SHA.
    frozen, planned = _check_frozen_plan(plan_root, bench_root, model, config_path)
    cfg, checks = _config_checks(config_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_type": "comsol_model_and_pilot_readiness",
        "mode": "pilot" if pilot else "live_inspection" if live else "static",
        "observed_backend": "software_fixture" if (client_factory or geometry_runner) else (
            "comsol_mph_api" if live else "not_invoked"
        ),
        "model_sha256": sha256_file(model),
        "config_sha256": sha256_file(config_path),
        "campaign_plan_sha256": sha256_file(plan_root / "campaign_plan.json"),
        "planned_sweeps_sha256": sha256_file(plan_root / "planned_sweeps.csv"),
        "planned_solver_evaluations": len(planned),
        "preflight_checks": checks,
        "static_valid": all(row["passed"] for row in checks),
        "live_model_checked": False,
        "pilot_solver_called": False,
        "pilot_numerically_consistent": False,
        "ready_for_full_solver_run": False,
        "native_solver_log_captured": False,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
        "errors": [],
        "warnings": [
            "Loading an .mph model and validating expression names does not prove COMSOL physics correctness.",
            "Even a genuine one-RI pilot cannot establish sensitivity, FOM, convergence or method superiority.",
            "Keep independently obtained native COMSOL logs and genuine mesh/spectral refinement archives.",
        ],
        "input_source": str(frozen.get("source_evidence_class", "unknown")),
        "created_at_utc": _now(),
    }
    pilot_row: dict[str, Any] | None = None
    if live:
        if not report["static_valid"]:
            report["errors"].append("Static configuration checks failed; refusing COMSOL model startup")
        else:
            try:
                if client_factory is None:
                    mph = _load_mph()
                    client = mph.start()
                else:
                    client = client_factory()
                live_model = client.load(str(model))
                live_checks, metadata = _live_checks(live_model, cfg)
                report["preflight_checks"].extend(live_checks)
                report["model_metadata"] = metadata
                report["live_model_checked"] = True
                if not all(row["passed"] for row in live_checks):
                    report["errors"].append("Loaded COMSOL model lacks required study/physics/mesh/material contracts")
                elif pilot:
                    center_idx = int(frozen["ri_points"]) // 2
                    sample = planned.iloc[center_idx]
                    request_id = str(sample["request_id"])
                    geometry = Geometry(**{
                        key: float(sample[key]) for key in _PARAMETERS
                    })
                    geometry.validate()
                    # Production uses precisely the repository's real mph runner.
                    # A test-injected callback must never qualify as physical evidence.
                    evaluator = geometry_runner or run_comsol_geometries
                    report["pilot_solver_called"] = True
                    result = evaluator(model, config_path, [geometry])
                    normalized = _normalize_chunk_rows(
                        result, planned.iloc[center_idx:center_idx + 1],
                        start=center_idx, backend="comsol_mph" if (
                            client_factory is None and geometry_runner is None
                        ) else "injected_software_fixture", local_ids=True,
                    )
                    pilot_row = normalized[0]
                    report["pilot_request_id"] = request_id
                    report["pilot_sample_id"] = int(sample["sample_id"])
                    report["pilot_status"] = pilot_row["status"]
                    if pilot_row["status"] == "ok":
                        report["pilot_numerically_consistent"] = True
                    else:
                        report["errors"].append("Real/patched solver reported a failed pilot: " + pilot_row["status"])
            except (Exception,) as exc:
                report["errors"].append("Model load, runtime/license or pilot solver failed: " +
                                        str(exc))
    else:
        report["warnings"].append("Static inspection only: real COMSOL runtime/license/model structure not tested")
    # For actual costly runs an explicit successful pilot is required; test doubles
    # deliberately cannot promote readiness, even with internally consistent data.
    report["ready_for_full_solver_run"] = bool(
        report["static_valid"]
        and report["live_model_checked"]
        and report["pilot_numerically_consistent"]
        and report["observed_backend"] == "comsol_mph_api"
        and not report["errors"]
    )
    output.mkdir(parents=True, exist_ok=True)
    if pilot_row is not None:
        dest = output / "pilot_single_ri_spectrum.csv"
        with dest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=pilot_row.keys())
            writer.writeheader()
            writer.writerow(pilot_row)
        report["pilot_spectrum_sha256"] = sha256_file(dest)
        report["pilot_spectrum_file"] = dest.name
        report["warnings"].append("Single-RI pilot spectrum is unqualified evidence; no sensitivity or FOM inference permitted")
    report_path = output / "COMSOL_MODEL_PILOT_READINESS.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
                           encoding="utf-8")
    return report



def validate_live_pilot_report(
    report_path: str | Path, plan_dir: str | Path,
    benchmark_dir: str | Path, model_path: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Require source-bound real-backend pilot success before a long CLI run.

    A local JSON/hash match still cannot authenticate COMSOL's physical
    execution or validate Maxwell equations. It is an operational safeguard.
    """
    report_file = _require_file(report_path)
    report = json.loads(report_file.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Malformed COMSOL readiness report")
    plan, planned = _check_frozen_plan(
        plan_dir, benchmark_dir, model_path, config_path,
    )
    required_flags = {
        "schema_version": 1,
        "audit_type": "comsol_model_and_pilot_readiness",
        "mode": "pilot",
        "observed_backend": "comsol_mph_api",
        "static_valid": True,
        "live_model_checked": True,
        "pilot_solver_called": True,
        "pilot_numerically_consistent": True,
        "pilot_status": "ok",
        "ready_for_full_solver_run": True,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
    }
    for key, expected in required_flags.items():
        if report.get(key) != expected:
            raise ValueError("Missing genuine pilot readiness condition: " + key)
    checks = report.get("preflight_checks")
    if (report.get("errors") or not isinstance(checks, list) or not checks
            or any(not isinstance(check, dict) or check.get("passed") is not True
                   for check in checks)):
        raise ValueError("Real pilot report includes errors or failed model checks")
    paths = {
        "model_sha256": _require_file(model_path, suffix=".mph"),
        "config_sha256": _require_file(config_path),
        "campaign_plan_sha256": _require_file(Path(plan_dir) / "campaign_plan.json"),
        "planned_sweeps_sha256": _require_file(Path(plan_dir) / "planned_sweeps.csv"),
    }
    for name, file in paths.items():
        if report.get(name) != sha256_file(file):
            raise ValueError("Pilot readiness source hash mismatch: " + name)
    central = int(plan["ri_points"]) // 2
    expected = planned.iloc[central]
    if (report.get("pilot_sample_id") != int(expected["sample_id"])
            or report.get("pilot_request_id") != str(expected["request_id"])):
        raise ValueError("Pilot sample is not the frozen first request central RI")
    if report.get("pilot_spectrum_file") != "pilot_single_ri_spectrum.csv":
        raise ValueError("Pilot archive path is not canonical")
    spectrum = _require_file(report_file.parent / "pilot_single_ri_spectrum.csv")
    if sha256_file(spectrum) != report.get("pilot_spectrum_sha256"):
        raise ValueError("Pilot raw spectrum hash mismatch")
    rows = pd.read_csv(spectrum, dtype={"request_id": str})
    if (len(rows) != 1 or rows.iloc[0]["request_id"] != str(expected["request_id"])
            or rows.iloc[0]["status"] != "ok"
            or rows.iloc[0]["source"] != "comsol"):
        raise ValueError("Pilot numeric source row does not match the frozen request")
    if not math.isclose(float(rows.iloc[0]["analyte_ri"]),
                        float(expected["analyte_ri"]), abs_tol=1e-9, rel_tol=0):
        raise ValueError("Pilot target refractive index was altered")
    return {
        "operational_pilot_verified": True,
        "pilot_readiness_report_sha256": sha256_file(report_file),
        "source_authenticity_verified": False,
        "scientific_submission_ready": False,
    }
