from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from sprpcf.utils.reproducibility import sha256_file
from sprpcf.validation.convergence import audit_refinement_archive


COLUMNS = (
    "study_kind", "case_id", "refinement_index", "mesh_elements",
    "wavelength_step_nm", "analyte_ri", "d_over_lambda", "pitch_um",
    "metal_thickness_nm", "channel_radius_um", "lambda_res_nm",
    "fwhm_nm", "peak_loss_db_per_cm", "source",
)


def _make_archive(tmp_path: Path) -> Path:
    """Numerical consistency fixture ONLY, never a real COMSOL experiment."""
    root = tmp_path / "archive"
    root.mkdir()
    (root / "model.mph").write_text("DUMMY MODEL NOT COMSOL", encoding="utf-8")
    (root / "config.yaml").write_text("study: dummy\n", encoding="utf-8")
    (root / "solver.log").write_text("DUMMY LOG NOT A SOLVER RUN\n", encoding="utf-8")
    rows = []
    for kind in ("mesh", "spectral"):
        for i, level in enumerate((1, 2, 3)):
            rows.append({
                "study_kind": kind, "case_id": "fixture-geometry", "refinement_index": level,
                "mesh_elements": (1000, 2000, 4000)[i] if kind == "mesh" else 4000,
                "wavelength_step_nm": 0.1 if kind == "mesh" else (0.4, 0.2, 0.1)[i],
                "analyte_ri": 1.37, "d_over_lambda": 0.5, "pitch_um": 2.0,
                "metal_thickness_nm": 45.0, "channel_radius_um": 0.6,
                "lambda_res_nm": (590.0, 590.08, 590.09)[i],
                "fwhm_nm": (24.8, 24.4, 24.3)[i],
                "peak_loss_db_per_cm": (20.0, 20.2, 20.21)[i],
                "source": "comsol",
            })
    _write_rows(root, rows)
    manifest = {
        "schema_version": 1, "backend": "comsol",
        "solver_version": "TEST-FIXTURE-NOT-COMSOL",
        "materials_reference": "TEST-FIXTURE",
        "boundary_conditions": "TEST-FIXTURE",
        "pml_description": "TEST-FIXTURE",
        "geometry_definition": "TEST-FIXTURE",
        "study_protocol": "TEST-FIXTURE",
    }
    for name, filename in (("model", "model.mph"), ("configuration", "config.yaml"),
                           ("solver_log", "solver.log"), ("results", "refinement.csv")):
        manifest[name] = filename
        manifest[f"{name}_sha256"] = sha256_file(root / filename)
    (root / "refinement_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def _write_rows(root: Path, rows: list[dict]) -> None:
    with (root / "refinement.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _rows(root: Path) -> list[dict]:
    with (root / "refinement.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _refresh_results_hash(root: Path) -> None:
    path = root / "refinement_manifest.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["results_sha256"] = sha256_file(root / "refinement.csv")
    path.write_text(json.dumps(data), encoding="utf-8")


def test_consistent_refinements_report_numerics_only(tmp_path: Path) -> None:
    result = audit_refinement_archive(_make_archive(tmp_path))
    assert result["archive_hashes_consistent"], result["errors"]
    assert result["convergence_consistent"], result["errors"]
    assert len(result["checks"]) == 2
    assert result["scientific_correctness_verified"] is False
    assert result["source_authenticity_verified"] is False


def test_refinement_archive_detects_tampered_hash(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    (root / "model.mph").write_text("CHANGED", encoding="utf-8")
    report = audit_refinement_archive(root)
    assert not report["archive_hashes_consistent"]
    assert not report["convergence_consistent"]
    assert any("model SHA-256" in error for error in report["errors"])


def test_rejects_synthetic_source_even_if_hash_updated(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    rows = _rows(root)
    rows[0]["source"] = "synthetic"
    _write_rows(root, rows)
    _refresh_results_hash(root)
    report = audit_refinement_archive(root)
    assert not report["convergence_consistent"]
    assert any("non-COMSOL" in error for error in report["errors"])


def test_rejects_variable_geometry_between_refinement_levels(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    rows = _rows(root)
    rows[2]["pitch_um"] = "2.7"
    _write_rows(root, rows)
    _refresh_results_hash(root)
    report = audit_refinement_archive(root)
    assert any("geometry or RI" in e or "Physical case varied" in e for e in report["errors"])


def test_rejects_missing_spectral_study(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    _write_rows(root, [row for row in _rows(root) if row["study_kind"] == "mesh"])
    _refresh_results_hash(root)
    report = audit_refinement_archive(root)
    assert not report["convergence_consistent"]
    assert any("missing spectral" in e for e in report["errors"])


def test_rejects_nonconverging_resonance_and_invalid_steps(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    rows = _rows(root)
    rows[2]["lambda_res_nm"] = "594.0"
    rows[4]["wavelength_step_nm"] = "0.6"
    _write_rows(root, rows)
    _refresh_results_hash(root)
    report = audit_refinement_archive(root)
    assert not report["convergence_consistent"]
    assert any("convergence" in e or "refinement change increased" in e for e in report["errors"])
    assert any("Spectral step" in e for e in report["errors"])


def test_rejects_invalid_manifest_metadata_and_missing_solver_log(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    manifest = root / "refinement_manifest.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data.pop("materials_reference")
    manifest.write_text(json.dumps(data), encoding="utf-8")
    report = audit_refinement_archive(root)
    assert not report["convergence_consistent"]
    assert any("materials_reference" in e for e in report["errors"])
    (root / "solver.log").unlink()
    missing = audit_refinement_archive(root)
    assert not missing["convergence_consistent"]
    assert any("solver.log" in e for e in missing["errors"])


def test_cli_strict_exit_and_report(tmp_path: Path) -> None:
    root = _make_archive(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_comsol_convergence.py"
    output = tmp_path / "report.json"
    args = [sys.executable, str(script), "--archive", str(root),
            "--json-out", str(output), "--strict"]
    passed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["scientific_correctness_verified"] is False
    (root / "model.mph").write_text("TAMPERED", encoding="utf-8")
    failed = subprocess.run(args, capture_output=True, text=True, check=False)
    assert failed.returncode == 2, failed.stderr
