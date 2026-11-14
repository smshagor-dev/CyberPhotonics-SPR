"""Evidence-bound numerical refinement checks for archived PCF-SPR COMSOL studies.

This checks provenance and convergence *consistency*. It never certifies solver
execution, electromagnetic correctness, materials, fabrication or instrumentation.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from sprpcf.utils.reproducibility import sha256_file

FIELDS = (
    "study_kind", "case_id", "refinement_index", "mesh_elements",
    "wavelength_step_nm", "analyte_ri", "d_over_lambda", "pitch_um",
    "metal_thickness_nm", "channel_radius_um", "lambda_res_nm",
    "fwhm_nm", "peak_loss_db_per_cm", "source",
)
GEOMETRY = ("analyte_ri", "d_over_lambda", "pitch_um", "metal_thickness_nm", "channel_radius_um")
METRICS = ("lambda_res_nm", "fwhm_nm", "peak_loss_db_per_cm")
STUDIES = ("mesh", "spectral")


def _file(root: Path, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Manifest artifact path is missing")
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("All evidence inputs must be inside the archived evidence directory") from exc
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"Missing archived artifact: {value}")
    return path


def _finite(value: Any, key: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric {key}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric {key}")
    return number


def _load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV header missing or duplicated")
        missing = sorted(set(FIELDS).difference(reader.fieldnames))
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")
        rows: list[dict[str, Any]] = []
        for line_number, raw in enumerate(reader, 2):
            if None in raw:
                raise ValueError(f"Line {line_number}: unexpected additional columns")
            kind = str(raw["study_kind"]).strip().lower()
            case_id = str(raw["case_id"]).strip()
            if kind not in STUDIES or not case_id or raw["source"] != "comsol":
                raise ValueError(f"Line {line_number}: invalid study, case identifier or non-COMSOL source")
            item: dict[str, Any] = {"study_kind": kind, "case_id": case_id, "source": "comsol"}
            for key in FIELDS:
                if key not in item:
                    item[key] = _finite(raw[key], key)
            if item["refinement_index"] < 1 or int(item["refinement_index"]) != item["refinement_index"]:
                raise ValueError(f"Line {line_number}: refinement_index must be a positive integer")
            if item["mesh_elements"] < 1 or int(item["mesh_elements"]) != item["mesh_elements"]:
                raise ValueError(f"Line {line_number}: mesh_elements must be a positive integer")
            if not (item["wavelength_step_nm"] > 0 and item["fwhm_nm"] > 0 and item["peak_loss_db_per_cm"] > 0):
                raise ValueError(f"Line {line_number}: wavelength resolution, FWHM and loss must be positive")
            rows.append(item)
    if not rows:
        raise ValueError("Convergence table contains no data")
    return rows


def _group_evaluation(rows: list[dict[str, Any]], *, resonance_tol_nm: float,
                      fwhm_rel_tol: float, loss_rel_tol: float) -> dict[str, Any]:
    reasons: list[str] = []
    rows = sorted(rows, key=lambda item: item["refinement_index"])
    if len(rows) < 3:
        reasons.append("At least three refinement levels are required")
    if len({int(row["refinement_index"]) for row in rows}) != len(rows):
        reasons.append("Duplicate refinement index")
    for key in GEOMETRY:
        baseline = rows[0][key]
        if any(not math.isclose(row[key], baseline, abs_tol=1e-9, rel_tol=0) for row in rows[1:]):
            reasons.append(f"Physical case varied across refinements: {key}")
    kind = rows[0]["study_kind"]
    meshes = [row["mesh_elements"] for row in rows]
    steps = [row["wavelength_step_nm"] for row in rows]
    if kind == "mesh":
        if any(b <= a for a, b in zip(meshes, meshes[1:])):
            reasons.append("Mesh element counts must strictly increase")
        if any(not math.isclose(v, steps[0], abs_tol=1e-10, rel_tol=1e-8) for v in steps[1:]):
            reasons.append("Spectral grid must remain constant in mesh study")
    else:
        if any(b >= a for a, b in zip(steps, steps[1:])):
            reasons.append("Spectral step must strictly decrease")
        if any(int(v) != int(meshes[0]) for v in meshes[1:]):
            reasons.append("Mesh size must remain constant in spectral study")

    deltas: dict[str, dict[str, float]] = {}
    if len(rows) >= 3:
        thresholds = {
            "lambda_res_nm": resonance_tol_nm,
            "fwhm_nm": fwhm_rel_tol,
            "peak_loss_db_per_cm": loss_rel_tol,
        }
        for metric in METRICS:
            old, middle, finest = (row[metric] for row in rows[-3:])
            prior = abs(middle - old)
            latest = abs(finest - middle)
            relative = latest / max(abs(finest), abs(middle), 1e-12)
            accepted = latest <= thresholds[metric] if metric == "lambda_res_nm" else relative <= thresholds[metric]
            # A final small change is insufficient when the refinement progression is diverging.
            if latest > prior * 1.05 + 1e-12:
                accepted = False
                reasons.append(f"{metric}: last refinement change increased")
            if not accepted and f"{metric}: last refinement change increased" not in reasons:
                reasons.append(f"{metric}: convergence tolerance exceeded")
            deltas[metric] = {
                "last_absolute_difference": latest,
                "last_relative_difference": relative,
                "allowed_tolerance": thresholds[metric],
            }
    return {
        "case_id": rows[0]["case_id"],
        "study_kind": kind,
        "levels": len(rows),
        "convergence_consistent": not reasons,
        "reasons": reasons,
        "differences": deltas,
    }


def audit_refinement_archive(
    archive: str | Path,
    *,
    resonance_tol_nm: float = 0.1,
    fwhm_rel_tol: float = 0.01,
    loss_rel_tol: float = 0.02,
) -> dict[str, Any]:
    """Verify archived model/log/result hashes and numerical refinement trends.

    A report may pass `convergence_consistent` for synthetic fixture data; a pass
    is never proof that an authentic COMSOL solve was performed.
    """
    for name, value in (("resonance_tol_nm", resonance_tol_nm),
                        ("fwhm_rel_tol", fwhm_rel_tol), ("loss_rel_tol", loss_rel_tol)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    root = Path(archive).resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "audit_type": "comsol_numerical_refinement_consistency",
        "archive_hashes_consistent": False,
        "convergence_consistent": False,
        "scientific_correctness_verified": False,
        "source_authenticity_verified": False,
        "errors": [],
        "checks": [],
        "file_sha256": {},
    }
    errors = report["errors"]
    try:
        manifest_path = _file(root, "refinement_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be JSON object")
        if manifest.get("schema_version") != 1 or manifest.get("backend") != "comsol":
            raise ValueError("Manifest must claim schema version 1 and backend comsol")
        for key in ("solver_version", "materials_reference", "boundary_conditions",
                    "pml_description", "geometry_definition", "study_protocol"):
            if not isinstance(manifest.get(key), str) or not manifest[key].strip():
                errors.append(f"Manifest missing research metadata: {key}")
        required = ("model", "configuration", "solver_log", "results")
        verified = True
        paths: dict[str, Path] = {}
        for key in required:
            file = _file(root, manifest.get(key))
            paths[key] = file
            actual = sha256_file(file)
            report["file_sha256"][key] = actual
            expected = manifest.get(f"{key}_sha256")
            if not isinstance(expected, str) or actual != expected:
                verified = False
                errors.append(f"Archived {key} SHA-256 mismatch")
        report["file_sha256"]["manifest"] = sha256_file(manifest_path)
        report["archive_hashes_consistent"] = verified
        rows = _load_rows(paths["results"])
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        cases: dict[str, tuple[float, ...]] = {}
        for row in rows:
            case = row["case_id"]
            geometry = tuple(row[k] for k in GEOMETRY)
            if case in cases and not all(math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                                         for a, b in zip(cases[case], geometry)):
                errors.append(f"Case {case}: geometry or RI differs across studies")
            cases.setdefault(case, geometry)
            groups.setdefault((case, row["study_kind"]), []).append(row)
        for case in sorted(cases):
            for kind in STUDIES:
                group = groups.get((case, kind))
                if not group:
                    errors.append(f"Case {case}: missing {kind} convergence study")
                    continue
                result = _group_evaluation(group, resonance_tol_nm=resonance_tol_nm,
                                           fwhm_rel_tol=fwhm_rel_tol, loss_rel_tol=loss_rel_tol)
                report["checks"].append(result)
                if not result["convergence_consistent"]:
                    errors.extend(f"Case {case}/{kind}: {why}" for why in result["reasons"])
        report["convergence_consistent"] = not errors and verified
        report["cases"] = len(cases)
        report["records"] = len(rows)
    except (OSError, UnicodeError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
        errors.append(str(exc))
    report["limitations"] = [
        "Hash consistency is not proof that the solver was executed or that results were independently sourced.",
        "Mesh and spectral refinement do not independently certify electromagnetic equations, materials, PML or calibration.",
        "A reviewer must inspect the actual .mph model, solver version, geometry, mode selection and literature comparisons.",
    ]
    return report
