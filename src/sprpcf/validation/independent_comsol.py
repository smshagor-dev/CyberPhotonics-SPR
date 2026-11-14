"""Strictly reconcile Stage-4 inverse requests with independently archived FEM sweeps.

The checks here establish *recorded numerical consistency*, not solver authenticity
or electromagnetic correctness. A plausible forged solver archive can pass hashes;
scientific review and a separately qualified source remain mandatory.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS, METRIC_COLUMNS
from sprpcf.simulation.metrics import extract_metrics
from sprpcf.utils.reproducibility import sha256_file
from sprpcf.validation.closed_loop import build_validation_ri_values
from sprpcf.validation.convergence import audit_refinement_archive
from sprpcf.validation.publication_ml import _bootstrap_group_mean

_REQUIRED_SWEEP = (
    "request_id", "analyte_ri", "status", "source", "evidence_class",
    *GEOMETRY_COLUMNS, "wavelength_nm", "loss_db_per_cm", "lambda_res_nm", "fwhm_nm",
)
_REQUIRED_REQUEST = (
    "request_id", "design_seed", "method", "target_id", "target_group_id",
    "analyte_ri", *GEOMETRY_COLUMNS,
    "fixed_geometry_ri_sweep_required", "verification_status",
    *(f"target_{name}" for name in METRIC_COLUMNS),
)
_ARCHIVE_NAMES = ("model", "config", "solver_log", "sweeps")


def _archive_file(root: Path, rel: Any) -> Path:
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError("Archive manifest is missing a file path")
    candidate = root / rel
    full = candidate.resolve()
    try:
        full.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError("Archive paths must remain inside the evidence archive") from exc
    if not candidate.is_file() or candidate.is_symlink() or any(
        parent.is_symlink() for parent in candidate.parents if parent != root.parent
    ):
        raise ValueError(f"Missing, symlinked or invalid archived evidence file: {rel}")
    return full


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path.name}")
    return value


def _csv(path: Path, required: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"request_id": str, "target_group_id": str})
    if frame.empty or not set(required).issubset(frame.columns):
        raise ValueError(f"{path.name} is empty or missing required fields: {sorted(set(required)-set(frame.columns))}")
    if frame.columns.has_duplicates:
        raise ValueError(f"{path.name} contains repeated column headings")
    return frame


def _finite(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: value is not numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label}: non-finite value")
    return number


def _spectrum(value: Any, *, label: str) -> np.ndarray:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}: missing archived spectrum")
    try:
        parts = np.array([float(part.strip()) for part in value.split(",")], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: malformed spectrum") from exc
    if len(parts) < 16 or not np.all(np.isfinite(parts)):
        raise ValueError(f"{label}: spectrum requires at least sixteen finite points")
    return parts


def _match(a: Any, b: Any, *, label: str, tolerance: float = 1e-8) -> None:
    if not math.isclose(_finite(a, label), _finite(b, label), abs_tol=tolerance, rel_tol=0):
        raise ValueError(f"{label}: submitted value differs from frozen requested value")


def _inputs(
    benchmark_dir: Path, archive_dir: Path, refinement_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    benchmark_manifest_path = benchmark_dir / "INVERSE_BENCHMARK_MANIFEST.json"
    requests_path = benchmark_dir / "PHYSICS_VERIFICATION_REQUESTS.csv"
    selected_path = benchmark_dir / "selected_designs.csv"
    bm = _read_json(benchmark_manifest_path)
    if bm.get("experiment") != "matched_surrogate_query_inverse_design":
        raise ValueError("Not a Stage 4 frozen inverse-design benchmark")
    for name, path in (
        ("PHYSICS_VERIFICATION_REQUESTS.csv", requests_path),
        ("selected_designs.csv", selected_path),
    ):
        if bm.get("outputs_sha256", {}).get(name) != sha256_file(path):
            raise ValueError(f"Benchmark artifact hash mismatch: {name}")
    requests = _csv(requests_path, _REQUIRED_REQUEST)
    chosen = _csv(selected_path, (
        "design_seed", "method", "target_id", "target_group_id", "analyte_ri", *GEOMETRY_COLUMNS,
    ))
    if requests["request_id"].isna().any() or requests["request_id"].duplicated().any():
        raise ValueError("Duplicate/missing frozen physics request_id")
    if len(requests) != len(chosen):
        raise ValueError("Every selected design must have exactly one COMSOL request")
    keys = ["design_seed", "method", "target_id", "target_group_id"]
    if chosen.duplicated(subset=keys).any() or requests.duplicated(subset=keys).any():
        raise ValueError("Duplicate method/target/seed identity in Stage 4 results")
    frozen = requests.merge(chosen, on=keys, how="outer", indicator=True, suffixes=("", "_selected"))
    if not frozen["_merge"].eq("both").all():
        raise ValueError("COMSOL request identities differ from frozen selected designs")
    for key in ("analyte_ri", *GEOMETRY_COLUMNS):
        for _, row in frozen.iterrows():
            _match(row[key], row[f"{key}_selected"], label=f"Stage 4 request {key}")

    ref_report = audit_refinement_archive(refinement_dir)
    ref_manifest = refinement_dir / "refinement_manifest.json"
    archive_manifest_path = archive_dir / "verification_manifest.json"
    manifest = _read_json(archive_manifest_path)
    if (manifest.get("schema_version") != 1 or manifest.get("backend") != "comsol"
            or manifest.get("evidence_class") != "comsol_physics"):
        raise ValueError("Solver manifest must declare schema 1 and COMSOL physical evidence")
    if manifest.get("benchmark_manifest_sha256") != sha256_file(benchmark_manifest_path):
        raise ValueError("COMSOL manifest does not match frozen benchmark manifest")
    if manifest.get("requests_sha256") != sha256_file(requests_path):
        raise ValueError("COMSOL manifest does not match frozen request table")
    if manifest.get("refinement_manifest_sha256") != sha256_file(ref_manifest):
        raise ValueError("COMSOL refinement archive binding mismatch")
    if manifest.get("solver_version") is None or not str(manifest.get("solver_version")).strip():
        raise ValueError("Solver version declaration is missing")
    if not ref_report["convergence_consistent"]:
        raise ValueError("Independent mesh/spectral refinement archive does not pass consistency checks")
    files: dict[str, Path] = {}
    for name in _ARCHIVE_NAMES:
        file = _archive_file(archive_dir, manifest.get(name))
        if sha256_file(file) != manifest.get(name + "_sha256"):
            raise ValueError(f"COMSOL {name} archive SHA-256 mismatch")
        files[name] = file
    if files["model"].suffix.lower() != ".mph":
        raise ValueError("COMSOL archive must contain an .mph model file")
    if sha256_file(files["model"]) != ref_report["file_sha256"].get("model"):
        raise ValueError("Refinement study and requested sweeps must reference identical model hashes")
    if not isinstance(manifest.get("normalization_scales"), dict):
        raise ValueError("Pre-registered per-metric error normalization scales are required")
    scales = {key: _finite(manifest["normalization_scales"].get(key), key) for key in METRIC_COLUMNS}
    if any(value <= 0 for value in scales.values()):
        raise ValueError("Error normalization scales must be positive")
    points = manifest.get("ri_points")
    if not isinstance(points, int) or points < 3 or points % 2 != 1:
        raise ValueError("COMSOL RI sweep must have odd ri_points >= 3")
    span = _finite(manifest.get("ri_span"), "ri_span")
    if not 0 < span < 0.5:
        raise ValueError("RI span must be in (0, 0.5)")
    sweeps = _csv(files["sweeps"], _REQUIRED_SWEEP)
    if not sweeps["source"].eq("comsol").all() or not sweeps["evidence_class"].eq("comsol_physics").all():
        raise ValueError("Every archived spectrum must claim source=comsol, evidence_class=comsol_physics")
    return requests, sweeps, bm, manifest, ref_report


def _evaluate_request(
    request: pd.Series, group: pd.DataFrame, *, ri_points: int, ri_span: float,
    normalization: dict[str, float], spectrum_tolerance_nm: float,
) -> dict[str, Any]:
    request_id = str(request["request_id"])
    result: dict[str, Any] = {
        "request_id": request_id,
        "method": str(request["method"]),
        "design_seed": int(request["design_seed"]),
        "target_id": int(request["target_id"]),
        "target_group_id": str(request["target_group_id"]),
        "status": "not_evaluated", "solver_failed_points": 0,
    }
    if len(group) != ri_points:
        raise ValueError(f"{request_id}: expected {ri_points} RI sweep records, received {len(group)}")
    expected_ri = np.array(build_validation_ri_values(
        _finite(request["analyte_ri"], "request RI"), span=ri_span, points=ri_points,
    ))
    observed_ri = group["analyte_ri"].to_numpy(dtype=float)
    if not np.isfinite(observed_ri).all() or not np.allclose(
        np.sort(observed_ri), np.sort(expected_ri), atol=1e-9, rtol=0,
    ):
        raise ValueError(f"{request_id}: RI sweep differs from frozen protocol")
    if len(set(np.round(observed_ri, 12))) != ri_points:
        raise ValueError(f"{request_id}: duplicated RI values")
    for key in GEOMETRY_COLUMNS:
        vals = group[key].to_numpy(dtype=float)
        if not np.isfinite(vals).all() or not np.allclose(
            vals, _finite(request[key], key), atol=1e-9, rtol=0,
        ):
            raise ValueError(f"{request_id}: returned geometry mismatch for {key}")
    statuses = [str(v) for v in group["status"]]
    if any(status != "ok" and not status.startswith("failed:") for status in statuses):
        raise ValueError(f"{request_id}: invalid solver status label")
    failures = sum(status != "ok" for status in statuses)
    result["solver_failed_points"] = failures
    if failures:
        result["status"] = "solver_failed"
        result["failure_reasons"] = "; ".join(status for status in statuses if status != "ok")
        return result

    measured: list[tuple[float, float, float]] = []
    for _, row in group.iterrows():
        waves = _spectrum(row["wavelength_nm"], label=f"{request_id} wavelengths")
        losses = _spectrum(row["loss_db_per_cm"], label=f"{request_id} losses")
        if waves.shape != losses.shape or not np.all(np.diff(waves) > 0):
            raise ValueError(f"{request_id}: nonmonotonic or unmatched spectra")
        metric = extract_metrics(waves, losses)
        if not np.isfinite(metric.lambda_res_nm) or not np.isfinite(metric.fwhm_nm) or metric.fwhm_nm <= 0:
            raise ValueError(f"{request_id}: non-finite resonance or FWHM")
        _match(row["lambda_res_nm"], metric.lambda_res_nm,
               label=f"{request_id} archived lambda", tolerance=spectrum_tolerance_nm)
        _match(row["fwhm_nm"], metric.fwhm_nm,
               label=f"{request_id} archived FWHM", tolerance=spectrum_tolerance_nm)
        measured.append((float(row["analyte_ri"]), float(metric.lambda_res_nm), float(metric.fwhm_nm)))
    observed = np.array(sorted(measured, key=lambda x: x[0]))
    ri, wavelength, widths = observed.T
    slope, intercept = np.polyfit(ri, wavelength, 1)
    if not np.all(np.isfinite([slope, intercept])) or np.any(widths <= 0):
        raise ValueError(f"{request_id}: invalid sensitivity/FOM fit")
    fit = slope * ri + intercept
    residual = float(np.square(wavelength - fit).sum())
    total = float(np.square(wavelength - wavelength.mean()).sum())
    linearity = 1.0 - residual / total if total > 1e-12 else (1.0 if residual < 1e-12 else 0.0)
    central_idx = int(np.argmin(np.abs(ri - float(request["analyte_ri"]))))
    if abs(ri[central_idx] - float(request["analyte_ri"])) > 1e-9:
        raise ValueError(f"{request_id}: exact target RI was not solved")
    actuals = {
        "sensitivity_nm_per_riu": float(slope),
        "fom_per_riu": float(abs(slope) / widths.mean()),
        "lambda_res_nm": float(wavelength[central_idx]),
    }
    normalized: list[float] = []
    for metric in METRIC_COLUMNS:
        actual = actuals[metric]
        expected = _finite(request[f"target_{metric}"], f"target_{metric}")
        if not np.isfinite(actual):
            raise ValueError(f"{request_id}: invalid recomputed {metric}")
        result[f"actual_{metric}"] = actual
        result[f"target_{metric}"] = expected
        absolute = abs(actual - expected)
        result[f"physics_abs_error_{metric}"] = absolute
        normalized.append(absolute / normalization[metric])
    result["physics_normalized_mae"] = float(np.mean(normalized))
    result["ri_linearity_r2"] = float(linearity)
    result["status"] = "spectrum_consistent_unqualified"
    return result


def _compare(
    evaluated: pd.DataFrame, methods: list[str], seeds: list[int], *,
    resamples: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    method_summary: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for method in methods:
        selected = evaluated.loc[evaluated["method"].eq(method)]
        ok = selected.loc[selected["status"].eq("spectrum_consistent_unqualified")]
        method_summary.append({
            "method": method,
            "requests_total": int(len(selected)),
            "complete_sweep_requests": int(len(ok)),
            "solver_failed_requests": int(selected["status"].eq("solver_failed").sum()),
            "candidate_success_rate": float(len(ok) / len(selected)) if len(selected) else 0.0,
            "mean_normalized_abs_error_on_completed_only": (
                float(ok["physics_normalized_mae"].mean()) if not ok.empty else None
            ),
            "completion_bias_warning": (
                "Completed-only average cannot support superiority if any solver request failed."
            ),
        })
    for seed in seeds:
        subset = evaluated.loc[evaluated["design_seed"].eq(seed)]
        for rival in methods:
            if rival == "tandem_latent":
                continue
            differences: list[float] = []
            skipped = 0
            for group_id, group in subset.groupby("target_group_id", sort=True):
                reference = group.loc[group["method"].eq("tandem_latent")]
                other = group.loc[group["method"].eq(rival)]
                if (len(reference) != 1 or len(other) != 1
                        or reference.iloc[0]["status"] != "spectrum_consistent_unqualified"
                        or other.iloc[0]["status"] != "spectrum_consistent_unqualified"):
                    skipped += 1
                    continue
                differences.append(float(
                    other.iloc[0]["physics_normalized_mae"]
                    - reference.iloc[0]["physics_normalized_mae"]
                ))
            if len(differences) >= 3:
                stats = _bootstrap_group_mean(np.array(differences), resamples=resamples, seed=int(seed))
            else:
                stats = {"mean": float(np.mean(differences)) if differences else None,
                         "ci_low": None, "ci_high": None,
                         "interval_status": "insufficient_complete_independent_geometry_pairs"}
            pairs.append({
                "design_seed": int(seed), "method": rival,
                "reference": "tandem_latent",
                "complete_pairs": len(differences), "excluded_failed_or_unpaired": skipped,
                "mean_rival_minus_tandem_physics_error": stats["mean"],
                "ci_low": stats["ci_low"], "ci_high": stats["ci_high"],
                "interval_status": stats["interval_status"],
            })
    return method_summary, pairs


def audit_independent_comsol_comparison(
    benchmark_dir: str | Path, archive_dir: str | Path,
    refinement_dir: str | Path, output_dir: str | Path, *,
    spectrum_tolerance_nm: float = 0.05,
    bootstrap_resamples: int = 500,
) -> dict[str, Any]:
    """Audit externally produced solver sweeps against frozen inverse requests.

    Consistency may pass for a forged or test-fixture archive; the report ALWAYS
    marks source authenticity and independent physical correctness as unverified.
    """
    for value, key in ((spectrum_tolerance_nm, "spectrum_tolerance_nm"),):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if bootstrap_resamples < 200:
        raise ValueError("At least 200 bootstrap resamples required")
    benchmark = Path(benchmark_dir).resolve()
    archive = Path(archive_dir).resolve()
    refinement = Path(refinement_dir).resolve()
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Output directory must be empty or absent")
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_type": "independent_comsol_request_to_spectrum_comparison",
        "numerical_archive_consistent": False,
        "all_requested_sweeps_successful": False,
        "source_authenticity_verified": False,
        "physical_correctness_verified": False,
        "scientific_submission_ready": False,
        "errors": [],
        "method_summary": [],
        "pairs": [],
        "counts": {"total_requests": 0, "solver_failed_requests": 0,
                   "completed_requests": 0},
        "limitations": [
            "Archive hashes and recomputed spectra cannot prove actual COMSOL execution or physical correctness.",
            "Stage 4 generation seeds reuse one checkpoint; paired bootstrap within one seed is descriptive.",
            "The tandem-trained forward surrogate selected all designs. A fresh untouched test and fair total compute budgets remain unverified.",
            "Incomplete solver sweeps remain in the denominator. Complete-case means must not be used to claim superiority.",
        ],
    }
    tables: dict[str, pd.DataFrame] = {}
    try:
        requests, sweeps, benchmark_manifest, solver_manifest, refine_report = _inputs(
            benchmark, archive, refinement,
        )
        report["source_checksums"] = {
            "benchmark_manifest_sha256": sha256_file(benchmark / "INVERSE_BENCHMARK_MANIFEST.json"),
            "requests_sha256": sha256_file(benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv"),
            "solver_manifest_sha256": sha256_file(archive / "verification_manifest.json"),
            "sweeps_sha256": sha256_file(_archive_file(archive, solver_manifest["sweeps"])),
            "refinement_manifest_sha256": sha256_file(refinement / "refinement_manifest.json"),
        }
        report["refinement_numerical_consistency"] = bool(refine_report["convergence_consistent"])
        report["benchmark_evidence_class"] = benchmark_manifest["evidence_class"]
        report["declared_solver_backend"] = solver_manifest["backend"]
        if requests["fixed_geometry_ri_sweep_required"].astype(str).str.lower().ne("true").any():
            raise ValueError("Frozen request table includes designs without required fixed RI sweeps")
        if not requests["verification_status"].eq("not_run").all():
            raise ValueError("Stage 4 request queue was modified in place; expected immutable not_run state")
        expected_ids = set(requests["request_id"].astype(str))
        observed_ids = set(sweeps["request_id"].astype(str))
        if expected_ids != observed_ids:
            raise ValueError("Solver archive has missing or unrequested request IDs: "
                             f"missing={sorted(expected_ids - observed_ids)}, "
                             f"extra={sorted(observed_ids - expected_ids)}")
        normalization = {key: float(solver_manifest["normalization_scales"][key])
                         for key in METRIC_COLUMNS}
        indexed = {key: part for key, part in sweeps.groupby("request_id", sort=False)}
        outcomes = []
        for _, request in requests.iterrows():
            outcomes.append(_evaluate_request(
                request, indexed[str(request["request_id"])],
                ri_points=int(solver_manifest["ri_points"]),
                ri_span=float(solver_manifest["ri_span"]),
                normalization=normalization,
                spectrum_tolerance_nm=spectrum_tolerance_nm,
            ))
        evaluated = pd.DataFrame(outcomes)
        methods = list(benchmark_manifest["methods"])
        seeds = [int(seed) for seed in benchmark_manifest["candidate_generation_seeds"]]
        if set(methods) != set(evaluated["method"]) or set(seeds) != set(evaluated["design_seed"]):
            raise ValueError("Method/generation seed coverage differs from benchmark manifest")
        counts = {
            "total_requests": len(evaluated),
            "solver_failed_requests": int(evaluated["status"].eq("solver_failed").sum()),
            "completed_requests": int(evaluated["status"].eq("spectrum_consistent_unqualified").sum()),
        }
        report["counts"] = counts
        summary, paired = _compare(
            evaluated, methods, seeds, resamples=bootstrap_resamples,
        )
        report["method_summary"] = summary
        report["pairs"] = paired
        report["numerical_archive_consistent"] = True
        report["all_requested_sweeps_successful"] = counts["solver_failed_requests"] == 0
        if not report["all_requested_sweeps_successful"]:
            report["errors"].append("Real solver marked failed RI points; complete paired verification unavailable")
        tables = {
            "request_outcomes.csv": evaluated,
            "method_completion_summary.csv": pd.DataFrame(summary),
            "paired_comsol_error_differences.csv": pd.DataFrame(paired),
        }
    except (OSError, UnicodeError, KeyError, TypeError, ValueError, IndexError, json.JSONDecodeError) as exc:
        report["errors"].append(f"Archived verification cannot complete: {exc}")
        report["numerical_archive_consistent"] = False
    output.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(output / name, index=False)
    report["output_hashes"] = {name: sha256_file(output / name) for name in tables}
    (output / "PHYSICS_VERIFICATION_AUDIT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
    )
    return report
