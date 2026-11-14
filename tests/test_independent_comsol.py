"""Test fixtures intentionally simulate spectra and fake model bytes.

Every passing audit in these tests must still report physical correctness and
authenticity as UNVERIFIED. No fixture constitutes a real COMSOL experiment.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprpcf.ml.dataset import GEOMETRY_COLUMNS, METRIC_COLUMNS
from sprpcf.simulation.metrics import extract_metrics
from sprpcf.utils.reproducibility import sha256_file
from sprpcf.validation.independent_comsol import audit_independent_comsol_comparison
from sprpcf.validation.inverse_benchmark import METHODS


def _json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _fixture(root: Path) -> tuple[Path, Path, Path]:
    benchmark, archive, refinement = (root / "benchmark", root / "solver", root / "refinement")
    for directory in (benchmark, archive, refinement):
        directory.mkdir()

    # The same *dummy bytes* appear in each mock archive. These are not .mph models.
    for directory in (archive, refinement):
        (directory / "model.mph").write_bytes(b"TEST FIXTURE: not a COMSOL model")
        (directory / "config.yaml").write_text("test: simulation fixture only\n", encoding="utf-8")
        (directory / "solver.log").write_text("TEST FIXTURE: no solver execution\n", encoding="utf-8")

    refine_rows = []
    for kind in ("mesh", "spectral"):
        for i in range(3):
            refine_rows.append({
                "study_kind": kind, "case_id": "fake-example",
                "refinement_index": i + 1,
                "mesh_elements": (1000, 2000, 4000)[i] if kind == "mesh" else 4000,
                "wavelength_step_nm": 0.1 if kind == "mesh" else (0.4, 0.2, 0.1)[i],
                "analyte_ri": 1.33, "d_over_lambda": 0.5, "pitch_um": 2.0,
                "metal_thickness_nm": 40.0, "channel_radius_um": 0.6,
                "lambda_res_nm": (590.0, 590.08, 590.09)[i],
                "fwhm_nm": (24.8, 24.4, 24.3)[i],
                "peak_loss_db_per_cm": (20.0, 20.2, 20.21)[i],
                "source": "comsol",
            })
    pd.DataFrame(refine_rows).to_csv(refinement / "refinement.csv", index=False)
    refine_manifest = {
        "schema_version": 1, "backend": "comsol",
        "solver_version": "DUMMY-NOT-COMSOL",
        "materials_reference": "FAKE fixture only",
        "boundary_conditions": "FAKE fixture only",
        "pml_description": "FAKE fixture only",
        "geometry_definition": "FAKE fixture only",
        "study_protocol": "FAKE fixture only",
    }
    for name, filename in [
        ("model", "model.mph"), ("configuration", "config.yaml"),
        ("solver_log", "solver.log"), ("results", "refinement.csv"),
    ]:
        refine_manifest[name] = filename
        refine_manifest[name + "_sha256"] = sha256_file(refinement / filename)
    _json(refinement / "refinement_manifest.json", refine_manifest)

    requests = []
    chosen = []
    sweeps = []
    wavelengths = np.linspace(550.0, 650.0, 101)
    # Three independent groups, two candidate generation seeds, four methods.
    for seed in (7, 17):
        for group in range(3):
            for index, method in enumerate(METHODS):
                request_id = f"testcase-{seed}-{group}-{index}"
                geometry = {
                    "pitch_um": 1.5 + 0.15 * group + 0.01 * index,
                    "d_over_lambda": 0.5,
                    "metal_thickness_nm": 40.0,
                    "channel_radius_um": 0.6,
                }
                shared = {
                    "design_seed": seed, "method": method, "target_id": group,
                    "target_group_id": f"geometry-{group}",
                    "analyte_ri": 1.33, **geometry,
                }
                requests.append({
                    "request_id": request_id, **shared,
                    **{f"target_{name}": value for name, value in zip(
                        METRIC_COLUMNS, (500.0, 25.0, 600.0), strict=True
                    )},
                    "fixed_geometry_ri_sweep_required": True,
                    "verification_status": "not_run",
                    "physical_solver_result": "not_supplied",
                })
                chosen.append(shared)
                for ri in (1.31, 1.32, 1.33, 1.34, 1.35):
                    center = 600.0 + 500.0 * (ri - 1.33)
                    loss = 1.0 + 10.0 / (1 + ((wavelengths - center) / 9.0) ** 2)
                    metric = extract_metrics(wavelengths, loss)
                    sweeps.append({
                        "request_id": request_id, "analyte_ri": ri, **geometry,
                        "status": "ok", "source": "comsol", "evidence_class": "comsol_physics",
                        "wavelength_nm": ",".join(f"{v:.6f}" for v in wavelengths),
                        "loss_db_per_cm": ",".join(f"{v:.6f}" for v in loss),
                        "lambda_res_nm": metric.lambda_res_nm,
                        "fwhm_nm": metric.fwhm_nm,
                    })
    req_path = benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv"
    selected_path = benchmark / "selected_designs.csv"
    pd.DataFrame(requests).to_csv(req_path, index=False)
    pd.DataFrame(chosen).to_csv(selected_path, index=False)
    bm = {
        "experiment": "matched_surrogate_query_inverse_design",
        "evidence_class": "software_only",
        "methods": list(METHODS),
        "candidate_generation_seeds": [7, 17],
        "scientific_submission_ready": False,
        "outputs_sha256": {
            req_path.name: sha256_file(req_path),
            selected_path.name: sha256_file(selected_path),
        },
    }
    _json(benchmark / "INVERSE_BENCHMARK_MANIFEST.json", bm)
    pd.DataFrame(sweeps).to_csv(archive / "sweeps.csv", index=False)
    manifest = {
        "schema_version": 1, "backend": "comsol", "evidence_class": "comsol_physics",
        "solver_version": "FAKE-NO-REAL-COMSOL-EXECUTED",
        "benchmark_manifest_sha256": sha256_file(benchmark / "INVERSE_BENCHMARK_MANIFEST.json"),
        "requests_sha256": sha256_file(req_path),
        "refinement_manifest_sha256": sha256_file(refinement / "refinement_manifest.json"),
        "ri_points": 5, "ri_span": 0.04,
        "normalization_scales": {
            "sensitivity_nm_per_riu": 100.0,
            "fom_per_riu": 10.0,
            "lambda_res_nm": 10.0,
        },
    }
    for name, filename in [
        ("model", "model.mph"), ("config", "config.yaml"),
        ("solver_log", "solver.log"), ("sweeps", "sweeps.csv"),
    ]:
        manifest[name] = filename
        manifest[name + "_sha256"] = sha256_file(archive / filename)
    _json(archive / "verification_manifest.json", manifest)
    return benchmark, archive, refinement


def _refresh_sweeps_hash(archive: Path) -> None:
    path = archive / "verification_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["sweeps_sha256"] = sha256_file(archive / "sweeps.csv")
    _json(path, manifest)


def test_full_fixture_is_consistent_but_never_claims_physics(tmp_path: Path) -> None:
    benchmark, archive, refinement = _fixture(tmp_path)
    report = audit_independent_comsol_comparison(
        benchmark, archive, refinement, tmp_path / "out", bootstrap_resamples=200,
    )
    assert report["numerical_archive_consistent"], report["errors"]
    assert report["all_requested_sweeps_successful"]
    assert report["source_authenticity_verified"] is False
    assert report["physical_correctness_verified"] is False
    assert report["scientific_submission_ready"] is False
    assert report["counts"] == {
        "total_requests": 24, "solver_failed_requests": 0, "completed_requests": 24,
    }
    assert len(report["method_summary"]) == 4
    assert len(report["pairs"]) == 6
    out = tmp_path / "out"
    results = pd.read_csv(out / "request_outcomes.csv")
    assert len(results) == 24
    assert results["status"].eq("spectrum_consistent_unqualified").all()
    assert results["physics_normalized_mae"].notna().all()
    for path, digest in report["output_hashes"].items():
        assert sha256_file(out / path) == digest


def test_solver_failure_cannot_be_silently_dropped(tmp_path: Path) -> None:
    benchmark, archive, refinement = _fixture(tmp_path)
    path = archive / "sweeps.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "status"] = "failed: missing eigenmode"
    frame.to_csv(path, index=False)
    _refresh_sweeps_hash(archive)
    report = audit_independent_comsol_comparison(
        benchmark, archive, refinement, tmp_path / "out", bootstrap_resamples=200,
    )
    assert report["numerical_archive_consistent"]
    assert report["all_requested_sweeps_successful"] is False
    assert report["counts"]["solver_failed_requests"] == 1
    assert report["counts"]["completed_requests"] == 23
    assert any(row["requests_total"] == 6 and row["complete_sweep_requests"] == 5
               for row in report["method_summary"])
    assert any(row["excluded_failed_or_unpaired"] > 0 for row in report["pairs"])


@pytest.mark.parametrize("mutation,expected", [
    ("wrong_source", "source=comsol"),
    ("geometry", "geometry mismatch"),
    ("spectrum", "archived lambda"),
    ("missing_ri", "RI sweep records"),
    ("duplicate_ri", "RI sweep differs"),
])
def test_fake_or_inconsistent_sweeps_fail_closed(
    tmp_path: Path, mutation: str, expected: str,
) -> None:
    benchmark, archive, refinement = _fixture(tmp_path)
    path = archive / "sweeps.csv"
    frame = pd.read_csv(path)
    if mutation == "wrong_source":
        frame.loc[0, "source"] = "synthetic"
    elif mutation == "geometry":
        frame.loc[0, "pitch_um"] = 3.0
    elif mutation == "spectrum":
        frame.loc[0, "lambda_res_nm"] += 12.0
    elif mutation == "missing_ri":
        frame = frame.drop(index=0)
    elif mutation == "duplicate_ri":
        frame.loc[1, "analyte_ri"] = frame.loc[0, "analyte_ri"]
    frame.to_csv(path, index=False)
    _refresh_sweeps_hash(archive)
    report = audit_independent_comsol_comparison(
        benchmark, archive, refinement, tmp_path / "out", bootstrap_resamples=200,
    )
    assert not report["numerical_archive_consistent"]
    assert report["scientific_submission_ready"] is False
    assert any(expected in error for error in report["errors"])


def test_frozen_benchmark_cannot_be_modified(tmp_path: Path) -> None:
    benchmark, archive, refinement = _fixture(tmp_path)
    req = benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv"
    req.write_bytes(req.read_bytes() + b"\n")
    report = audit_independent_comsol_comparison(
        benchmark, archive, refinement, tmp_path / "out", bootstrap_resamples=200,
    )
    assert not report["numerical_archive_consistent"]
    assert any("Benchmark artifact hash mismatch" in e for e in report["errors"])


def test_model_must_match_refinement_archive(tmp_path: Path) -> None:
    benchmark, archive, refinement = _fixture(tmp_path)
    (archive / "model.mph").write_bytes(b"different dummy model")
    path = archive / "verification_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model_sha256"] = sha256_file(archive / "model.mph")
    _json(path, manifest)
    report = audit_independent_comsol_comparison(
        benchmark, archive, refinement, tmp_path / "out", bootstrap_resamples=200,
    )
    assert not report["numerical_archive_consistent"]
    assert any("identical model hashes" in e for e in report["errors"])


def test_missing_archive_fails_and_writes_blocker_report(tmp_path: Path) -> None:
    benchmark = tmp_path / "missing_benchmark"
    result = audit_independent_comsol_comparison(
        benchmark, tmp_path / "missing_solver", tmp_path / "missing_refinement",
        tmp_path / "out", bootstrap_resamples=200,
    )
    assert not result["numerical_archive_consistent"]
    assert not result["scientific_submission_ready"]
    assert result["errors"]
    assert (tmp_path / "out" / "PHYSICS_VERIFICATION_AUDIT.json").is_file()


def test_strict_cli_returns_two_for_incomplete_archive(tmp_path: Path) -> None:
    benchmark, archive, refinement = _fixture(tmp_path)
    frame = pd.read_csv(archive / "sweeps.csv")
    frame.loc[0, "status"] = "failed: solver unavailable"
    frame.to_csv(archive / "sweeps.csv", index=False)
    _refresh_sweeps_hash(archive)
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "scripts" / "verify_inverse_comsol.py"),
        "--benchmark", str(benchmark), "--solver-archive", str(archive),
        "--refinement-archive", str(refinement), "--out", str(tmp_path / "out"),
        "--bootstrap-resamples", "200", "--strict",
    ]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    assert process.returncode == 2, process.stderr
    report = json.loads((tmp_path / "out" / "PHYSICS_VERIFICATION_AUDIT.json").read_text(encoding="utf-8"))
    assert report["counts"]["solver_failed_requests"] == 1
    assert report["scientific_submission_ready"] is False
