"""Execute frozen inverse-design COMSOL RI sweeps and archive verifiable exports.

This module NEVER manufactures solver outputs. The only production backend
is the repository's COMSOL mph API runner. Injected runners are test-only,
cannot be sealed as COMSOL evidence, and are explicitly labeled as fixtures.

Scientific authenticity/Maxwell correctness remain subject to independent
review even when an archive passes numerical consistency checks.
"""
from __future__ import annotations

import csv
import json
import math
import shutil
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS, METRIC_COLUMNS
from sprpcf.simulation.comsol_sweep import run_comsol_geometries
from sprpcf.simulation.metrics import extract_metrics
from sprpcf.simulation.schema import Geometry
from sprpcf.utils.reproducibility import sha256_file
from sprpcf.validation.closed_loop import build_validation_ri_values
from sprpcf.validation.convergence import audit_refinement_archive

PLAN_FIELDS = (
    "sample_id", "request_id", "method", "design_seed", "target_id",
    "target_group_id", "analyte_ri", *GEOMETRY_COLUMNS,
)
SWEEP_FIELDS = (
    "request_id", "analyte_ri", *GEOMETRY_COLUMNS,
    "status", "source", "evidence_class", "wavelength_nm",
    "loss_db_per_cm", "lambda_res_nm", "fwhm_nm",
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return data


def _require_file(path: str | Path, *, suffix: str | None = None) -> Path:
    file = Path(path).resolve()
    if not file.is_file() or file.is_symlink() or file.stat().st_size < 1:
        raise ValueError("Input must be an existing nonempty regular file: " + str(file))
    if suffix is not None and file.suffix.lower() != suffix:
        raise ValueError("Expected " + suffix + " source file: " + str(file))
    return file


def _clean_output(out: str | Path) -> Path:
    path = Path(out).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError("Output must be absent or an empty directory: " + str(path))
    return path


def _finite(value: Any, name: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid " + name) from exc
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError("Expected finite " + ("positive " if positive else "") + name)
    return result


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _validated_requests(benchmark_dir: Path) -> tuple[dict[str, Any], pd.DataFrame]:
    manifest_path = _require_file(benchmark_dir / "INVERSE_BENCHMARK_MANIFEST.json")
    manifest = _read_json(manifest_path)
    if manifest.get("experiment") != "matched_surrogate_query_inverse_design":
        raise ValueError("Expected frozen Stage 4 inverse-design benchmark")
    digests = manifest.get("outputs_sha256")
    if not isinstance(digests, dict):
        raise ValueError("Stage 4 output digests are absent")
    required = ("PHYSICS_VERIFICATION_REQUESTS.csv", "selected_designs.csv")
    for filename in required:
        original = _require_file(benchmark_dir / filename)
        if digests.get(filename) != sha256_file(original):
            raise ValueError("Stage 4 frozen output hash mismatch: " + filename)
    requests = pd.read_csv(benchmark_dir / required[0], dtype={"request_id": str, "target_group_id": str})
    selected = pd.read_csv(benchmark_dir / required[1], dtype={"target_group_id": str})
    keys = ("design_seed", "method", "target_id", "target_group_id")
    fields = ("request_id", *keys, "analyte_ri", *GEOMETRY_COLUMNS,
              *(f"target_{key}" for key in METRIC_COLUMNS),
              "verification_status", "fixed_geometry_ri_sweep_required")
    if not set(fields).issubset(requests.columns):
        raise ValueError("Stage 4 requests missing frozen schema fields")
    if (requests.empty or requests["request_id"].isna().any()
            or requests["request_id"].duplicated().any()
            or requests.duplicated(subset=list(keys)).any()
            or selected.duplicated(subset=list(keys)).any()
            or len(requests) != len(selected)):
        raise ValueError("Missing, duplicated, or unmatched Stage 4 requests")
    merged = requests.merge(selected, on=list(keys), suffixes=("", "_selected"),
                            validate="one_to_one", indicator=True)
    if len(merged) != len(requests) or not merged["_merge"].eq("both").all():
        raise ValueError("Stage 4 selected designs differ from request identities")
    for key in ("analyte_ri", *GEOMETRY_COLUMNS):
        left = pd.to_numeric(merged[key], errors="coerce").to_numpy(dtype=float)
        right = pd.to_numeric(merged[key + "_selected"], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(left).all() and np.isfinite(right).all()
                and np.allclose(left, right, rtol=0, atol=1e-9)):
            raise ValueError("Stage 4 requested geometry/RI differs from selected designs: " + key)
    if not requests["verification_status"].eq("not_run").all():
        raise ValueError("Frozen Stage 4 request queue has been edited")
    if not requests["fixed_geometry_ri_sweep_required"].astype(str).str.lower().eq("true").all():
        raise ValueError("Stage 4 requests do not require fixed-geometry RI sweeps")
    for _, row in requests.iterrows():
        Geometry(
            d_over_lambda=_finite(row["d_over_lambda"], "d_over_lambda"),
            pitch_um=_finite(row["pitch_um"], "pitch_um"),
            metal_thickness_nm=_finite(row["metal_thickness_nm"], "metal_thickness_nm"),
            channel_radius_um=_finite(row["channel_radius_um"], "channel_radius_um"),
            analyte_ri=_finite(row["analyte_ri"], "analyte_ri"),
        ).validate()
        for name in METRIC_COLUMNS:
            _finite(row["target_" + name], name)
    return manifest, requests


def prepare_comsol_campaign(
    benchmark_dir: str | Path, model_path: str | Path, config_path: str | Path,
    output_dir: str | Path, *,
    ri_span: float, ri_points: int, normalization_scales: dict[str, float],
    solver_version: str,
) -> dict[str, Any]:
    """Pre-register immutable targets, RI protocol, model and scoring scales."""
    benchmark = Path(benchmark_dir).resolve()
    model = _require_file(model_path, suffix=".mph")
    config = _require_file(config_path)
    out = _clean_output(output_dir)
    if not isinstance(solver_version, str) or not solver_version.strip():
        raise ValueError("Provide actual installed COMSOL solver version")
    if set(normalization_scales) != set(METRIC_COLUMNS):
        raise ValueError("All three per-metric error normalization scales are required")
    scales = {name: _finite(normalization_scales[name], name, positive=True)
              for name in METRIC_COLUMNS}
    _finite(ri_span, "ri_span", positive=True)
    if not isinstance(ri_points, int) or ri_points < 3 or ri_points % 2 != 1:
        raise ValueError("ri_points must be odd integer >=3")
    stage4, requests = _validated_requests(benchmark)
    rows: list[dict[str, Any]] = []
    for _, source in requests.iterrows():
        for ri in build_validation_ri_values(
            _finite(source["analyte_ri"], "central RI"), span=ri_span, points=ri_points
        ):
            rows.append({
                "sample_id": len(rows), "request_id": str(source["request_id"]),
                "method": str(source["method"]), "design_seed": int(source["design_seed"]),
                "target_id": int(source["target_id"]),
                "target_group_id": str(source["target_group_id"]),
                "analyte_ri": ri,
                **{key: float(source[key]) for key in GEOMETRY_COLUMNS},
            })
    out.mkdir(parents=True, exist_ok=True)
    plan_path = out / "planned_sweeps.csv"
    _write_csv(plan_path, PLAN_FIELDS, rows)
    plan = {
        "schema_version": 1,
        "kind": "frozen_comsol_execution_plan",
        "physical_source_authenticity_verified": False,
        "planned_request_count": int(len(requests)),
        "planned_solver_evaluations": len(rows),
        "benchmark_manifest_sha256": sha256_file(
            benchmark / "INVERSE_BENCHMARK_MANIFEST.json"
        ),
        "requests_sha256": sha256_file(
            benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv"
        ),
        "model_sha256": sha256_file(model), "config_sha256": sha256_file(config),
        "planned_sweeps_sha256": sha256_file(plan_path),
        "solver_version": solver_version.strip(), "ri_span": float(ri_span),
        "ri_points": ri_points, "normalization_scales": scales,
        "source_evidence_class": stage4.get("evidence_class"),
        "limitations": [
            "A model filename and SHA-256 do not establish COMSOL file authenticity.",
            "This is a pre-registered plan, not an executed simulation.",
            "Mesh refinement, material dispersion, PML and mode tracking need independent review.",
        ],
    }
    (out / "campaign_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return plan


def _check_frozen_plan(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
) -> tuple[dict[str, Any], pd.DataFrame]:
    plan_root = Path(plan_dir).resolve()
    plan = _read_json(_require_file(plan_root / "campaign_plan.json"))
    if plan.get("kind") != "frozen_comsol_execution_plan" or plan.get("schema_version") != 1:
        raise ValueError("Not a frozen Stage 7 COMSOL campaign plan")
    paths = {
        "benchmark_manifest_sha256": Path(benchmark_dir) / "INVERSE_BENCHMARK_MANIFEST.json",
        "requests_sha256": Path(benchmark_dir) / "PHYSICS_VERIFICATION_REQUESTS.csv",
        "model_sha256": Path(model_path), "config_sha256": Path(config_path),
        "planned_sweeps_sha256": plan_root / "planned_sweeps.csv",
    }
    _validated_requests(Path(benchmark_dir))
    for key, value in paths.items():
        if sha256_file(_require_file(value)) != plan.get(key):
            raise ValueError("Frozen COMSOL campaign input changed: " + key)
    planned = pd.read_csv(paths["planned_sweeps_sha256"], dtype={"request_id": str, "target_group_id": str})
    if len(planned) != plan["planned_solver_evaluations"] or not set(PLAN_FIELDS).issubset(planned.columns):
        raise ValueError("Campaign plan is incomplete")
    if (not planned["sample_id"].equals(pd.Series(range(len(planned)), name="sample_id"))
            or planned.isna().any().any()):
        raise ValueError("Campaign plan contains missing or altered sample identities")
    if not isinstance(plan.get("normalization_scales"), dict):
        raise ValueError("Frozen normalization scales missing")
    return plan, planned


def execute_comsol_campaign(
    plan_dir: str | Path, benchmark_dir: str | Path,
    model_path: str | Path, config_path: str | Path,
    output_dir: str | Path, *,
    runner: Callable[[Path, Path, Sequence[Geometry]], pd.DataFrame] | None = None,
) -> dict[str, Any]:
    """Execute COMSOL mph solver. Injected test runner is never sealed as physics."""
    model = _require_file(model_path, suffix=".mph")
    config = _require_file(config_path)
    out = _clean_output(output_dir)
    plan, planned = _check_frozen_plan(plan_dir, benchmark_dir, model, config)
    real_backend = runner is None
    executor = run_comsol_geometries if real_backend else runner
    geometries = [
        Geometry(
            d_over_lambda=float(row["d_over_lambda"]),
            pitch_um=float(row["pitch_um"]),
            metal_thickness_nm=float(row["metal_thickness_nm"]),
            channel_radius_um=float(row["channel_radius_um"]),
            analyte_ri=float(row["analyte_ri"]),
        )
        for _, row in planned.iterrows()
    ]
    started = _now()
    # Actual mph loading/solving happens here. Failure to start is propagated
    # without creating a misleading "solver completed" report.
    frame = executor(model, config, geometries)
    if not isinstance(frame, pd.DataFrame) or len(frame) != len(planned):
        raise ValueError("COMSOL runner did not return exactly one row per scheduled RI sweep")
    if ("sample_id" not in frame or "status" not in frame
            or frame["sample_id"].tolist() != list(range(len(planned)))):
        raise ValueError("COMSOL sweep sample identities differ from frozen plan")
    rows: list[dict[str, Any]] = []
    fail = 0
    for index, (_, measured) in enumerate(frame.iterrows()):
        requested = planned.iloc[index]
        state = str(measured["status"])
        if state != "ok" and not state.startswith("failed:"):
            raise ValueError("Unexpected solver run status: " + state)
        for name in ("analyte_ri", *GEOMETRY_COLUMNS):
            if not math.isclose(
                _finite(measured[name], name), float(requested[name]),
                abs_tol=1e-9, rel_tol=0.0,
            ):
                raise ValueError("COMSOL response differs from frozen geometry/RI: " + name)
        result: dict[str, Any] = {
            "request_id": str(requested["request_id"]),
            "analyte_ri": float(requested["analyte_ri"]),
            **{key: float(requested[key]) for key in GEOMETRY_COLUMNS},
            "status": state, "source": "comsol" if real_backend else "test_fixture",
            "evidence_class": "comsol_physics" if real_backend else "software_only",
            "wavelength_nm": "", "loss_db_per_cm": "",
            "lambda_res_nm": "", "fwhm_nm": "",
        }
        if state == "ok":
            for field in ("wavelength_nm", "loss_db_per_cm"):
                if not isinstance(measured.get(field), str):
                    raise ValueError("Successful COMSOL record lacks raw spectral arrays")
            wavelength = np.array(
                [float(v) for v in str(measured["wavelength_nm"]).split(",")], dtype=float
            )
            loss = np.array(
                [float(v) for v in str(measured["loss_db_per_cm"]).split(",")], dtype=float
            )
            if (len(wavelength) < 16 or wavelength.shape != loss.shape
                    or not np.isfinite(wavelength).all() or not np.isfinite(loss).all()
                    or not np.all(np.diff(wavelength) > 0)):
                raise ValueError("Successful COMSOL record has malformed wavelength/loss arrays")
            metric = extract_metrics(wavelength, loss)
            if (not math.isfinite(metric.fwhm_nm) or metric.fwhm_nm <= 0
                    or not math.isclose(metric.lambda_res_nm, _finite(measured["lambda_res_nm"], "lambda_res_nm"),
                                        abs_tol=0.05, rel_tol=0)
                    or not math.isclose(metric.fwhm_nm, _finite(measured["fwhm_nm"], "fwhm_nm"),
                                        abs_tol=0.05, rel_tol=0)):
                raise ValueError("COMSOL record disagrees with spectrum-derived metrics")
            result.update(
                wavelength_nm=str(measured["wavelength_nm"]),
                loss_db_per_cm=str(measured["loss_db_per_cm"]),
                lambda_res_nm=float(metric.lambda_res_nm),
                fwhm_nm=float(metric.fwhm_nm),
            )
        else:
            fail += 1
        rows.append(result)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "sweeps.csv", SWEEP_FIELDS, rows)
    run = {
        "schema_version": 1,
        "kind": "comsol_campaign_execution",
        "started_at_utc": started, "completed_at_utc": _now(),
        "real_comsol_backend_invoked": real_backend,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
        "campaign_plan_sha256": sha256_file(Path(plan_dir) / "campaign_plan.json"),
        "planned_sweeps_sha256": plan["planned_sweeps_sha256"],
        "model_sha256": plan["model_sha256"],
        "config_sha256": plan["config_sha256"],
        "sweeps_sha256": sha256_file(out / "sweeps.csv"),
        "scheduled_solver_evaluations": len(rows),
        "failed_solver_evaluations": fail,
        "successful_solver_evaluations": len(rows) - fail,
        "native_solver_log_captured": False,
        "qualification": "execution_trace_only_unqualified",
        "remaining": [
            "Export and bind genuine native solver log from the same campaign.",
            "Supply complete independently checked mesh/spectral convergence archive.",
            "Review electromagnetic formulation, materials, PML, mode choice and spectra.",
        ],
    }
    (out / "execution_manifest.json").write_text(
        json.dumps(run, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run


def seal_comsol_campaign(
    plan_dir: str | Path, run_dir: str | Path,
    benchmark_dir: str | Path, model_path: str | Path, config_path: str | Path,
    native_solver_log: str | Path, refinement_archive: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Produce the exact Stage 5 archive contract from a genuine prior mph run.

    Sealing still does not certify that a supplied native solver log is authentic.
    """
    model = _require_file(model_path, suffix=".mph")
    config = _require_file(config_path)
    log = _require_file(native_solver_log)
    out = _clean_output(output_dir)
    plan, _ = _check_frozen_plan(plan_dir, benchmark_dir, model, config)
    run_root = Path(run_dir).resolve()
    run = _read_json(_require_file(run_root / "execution_manifest.json"))
    if run.get("real_comsol_backend_invoked") is not True:
        raise ValueError("Cannot seal injected test-runner results as real COMSOL")
    expected = {
        "campaign_plan_sha256": sha256_file(Path(plan_dir) / "campaign_plan.json"),
        "planned_sweeps_sha256": plan["planned_sweeps_sha256"],
        "model_sha256": plan["model_sha256"],
        "config_sha256": plan["config_sha256"],
        "sweeps_sha256": sha256_file(_require_file(run_root / "sweeps.csv")),
    }
    for key, digest in expected.items():
        if run.get(key) != digest:
            raise ValueError("COMSOL run was changed after execution: " + key)
    if (run.get("scheduled_solver_evaluations") != plan["planned_solver_evaluations"]
            or run.get("successful_solver_evaluations", -1) + run.get("failed_solver_evaluations", -1)
            != plan["planned_solver_evaluations"]):
        raise ValueError("Run ledger has incomplete sample accounting")
    if run.get("execution_mode") == "checkpointed_batches":
        # Imported here to avoid an import cycle: the batch runner reuses the
        # original Stage 7 plan validator and sealing contract.
        from sprpcf.simulation.resumable_campaign import inspect_campaign_checkpoint

        checked = inspect_campaign_checkpoint(
            plan_dir, benchmark_dir, model, config, run_root,
        )
        if (checked["backend_kind"] != "comsol_mph"
                or checked["remaining_solver_evaluations"] != 0
                or checked["failed_solver_evaluations"] != run.get("failed_solver_evaluations")
                or checked["checkpoint_manifest_sha256"] != run.get("checkpoint_manifest_sha256")
                or checked["chunk_evidence"] != run.get("chunk_evidence")
                or checked["canonical_merged_sweeps_sha256"] != expected["sweeps_sha256"]):
            raise ValueError("Checkpointed COMSOL run chunks disagree with sealed execution manifest/sweeps")
    session_logs: list[dict[str, Any]] | None = None
    if run.get("execution_mode") == "checkpointed_batches":
        # This is not optional for checkpointed campaigns: every uninterrupted
        # or interrupted solver invocation needs a separately supplied native log.
        from sprpcf.simulation.session_provenance import audit_native_session_logs
        from sprpcf.simulation.resumable_campaign import _checkpoint_contract

        contract = _checkpoint_contract(
            Path(plan_dir), plan,
            _read_json(_require_file(run_root / "checkpoint_manifest.json"))["batch_size"],
            real_backend=True,
        )
        if run.get("session_log_provenance_required") is not True:
            raise ValueError("Checkpointed COMSOL campaign lacks required session-log provenance")
        session_logs = audit_native_session_logs(run_root, contract, checked["chunk_evidence"])
        if session_logs is None or not session_logs:
            raise ValueError("No native solver log evidence for checkpointed COMSOL campaign")
        if run.get("session_evidence") != checked.get("session_evidence"):
            raise ValueError("Solver invocation session inventory differs from execution record")
        if sha256_file(log) not in {item["native_log_sha256"] for item in session_logs}:
            raise ValueError("Provided --native-solver-log was not bound to any recorded session")
    ref_root = Path(refinement_archive).resolve()
    verification = audit_refinement_archive(ref_root)
    if not verification["convergence_consistent"]:
        raise ValueError("Real refinement archive did not pass numerical consistency")
    if verification["file_sha256"].get("model") != plan["model_sha256"]:
        raise ValueError("Refinement study and campaign require identical model hash")
    ref_manifest = _require_file(ref_root / "refinement_manifest.json")
    out.mkdir(parents=True, exist_ok=True)
    # Large physical model artifacts are copied unchanged; hashes must survive.
    files = {
        "model": (model, "model.mph"), "config": (config, "config.yaml"),
        "solver_log": (log, "solver.log"),
        "sweeps": (run_root / "sweeps.csv", "sweeps.csv"),
    }
    manifest: dict[str, Any] = {
        "schema_version": 1, "backend": "comsol",
        "evidence_class": "comsol_physics",
        "solver_version": plan["solver_version"],
        "benchmark_manifest_sha256": plan["benchmark_manifest_sha256"],
        "requests_sha256": plan["requests_sha256"],
        "refinement_manifest_sha256": sha256_file(ref_manifest),
        "ri_span": plan["ri_span"], "ri_points": plan["ri_points"],
        "normalization_scales": plan["normalization_scales"],
        "execution_manifest_sha256": sha256_file(run_root / "execution_manifest.json"),
        "campaign_plan_sha256": sha256_file(Path(plan_dir) / "campaign_plan.json"),
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
    }
    if session_logs is not None:
        from sprpcf.simulation.session_provenance import LOG_DIR, SESSION_DIR
        all_evidence: dict[str, str] = {}
        for record in session_logs:
            sid = record["session_id"]
            for source_subdir, extension in (
                (SESSION_DIR, ".start.json"),
                (SESSION_DIR, ".end.json"),
                (LOG_DIR, ".log"),
                (LOG_DIR, ".json"),
            ):
                source = run_root / source_subdir / (sid + extension)
                relative = "session_logs/" + sid + extension
                destination = out / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                digest = sha256_file(destination)
                if digest != sha256_file(source):
                    raise ValueError("Session log evidence changed during archival: " + relative)
                all_evidence[relative] = digest
        index = out / "session_log_evidence.json"
        index.write_text(json.dumps({
            "schema_version": 1,
            "type": "per_execution_session_log_provenance",
            "source_authenticity_verified": False,
            "session_count": len(session_logs),
            "session_records": session_logs,
            "files_sha256": all_evidence,
            "warning": "Native log bytes supplied externally; hashes do not establish COMSOL authenticity.",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["session_log_evidence"] = "session_log_evidence.json"
        manifest["session_log_evidence_sha256"] = sha256_file(index)
        manifest["session_log_provenance_required"] = True
        manifest["execution_mode"] = "checkpointed_batches"
        manifest["solver_log_type"] = "one_native_log_from_complete_bound_session_ledger"
    for role, (source, filename) in files.items():
        destination = out / filename
        shutil.copyfile(source, destination)
        manifest[role] = filename
        manifest[role + "_sha256"] = sha256_file(destination)
        if manifest[role + "_sha256"] != sha256_file(source):
            raise ValueError("Sealed COMSOL input changed while copying: " + role)
    (out / "verification_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
