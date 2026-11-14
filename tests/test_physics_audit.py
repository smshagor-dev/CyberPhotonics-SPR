from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from sprpcf.ml.dataset import GEOMETRY_COLUMNS
from sprpcf.simulation.comsol_sweep import write_dataset
from sprpcf.simulation.metrics import extract_metrics
from sprpcf.utils.reproducibility import sha256_file
from sprpcf.validation.closed_loop import (
    AcceptanceThresholds,
    build_validation_ri_values,
    evaluate_closed_loop_results,
)
from sprpcf.validation.physics_audit import audit_comsol_iteration


def _fixture(tmp_path: Path) -> Path:
    """Synthetic fixtures exercise the auditor, NOT the COMSOL physics claim."""
    root = tmp_path / "iteration"
    root.mkdir()
    model = tmp_path / "fixture.mph"
    model.write_text("not a real COMSOL model", encoding="utf-8")
    config = tmp_path / "sweep.yaml"
    config.write_text("study: fixture", encoding="utf-8")

    target = {
        "pitch_um": 2.0,
        "d_over_lambda": 0.5,
        "metal_thickness_nm": 45.0,
        "channel_radius_um": 0.6,
        "analyte_ri": 1.33,
        "sensitivity_nm_per_riu": 500.0,
        "fom_per_riu": 21.0,
        "lambda_res_nm": 590.0,
    }
    targets = root / "targets_with_geometry.csv"
    pd.DataFrame([target]).to_csv(targets, index=False)
    wavelengths = np.linspace(520.0, 680.0, 321)
    rows = []
    for sample_id, ri in enumerate(build_validation_ri_values(1.33, span=0.04, points=5)):
        center = 590.0 + 500.0 * (ri - 1.33)
        loss = 1.0 + 8.0 / (1.0 + ((wavelengths - center) / 12.0) ** 2)
        # Compute from the stored 6-decimal values, exactly as the auditor does.
        wavelength_stored = ",".join(f"{v:.6f}" for v in wavelengths)
        loss_stored = ",".join(f"{v:.6f}" for v in loss)
        metrics = extract_metrics(
            np.array([float(v) for v in wavelength_stored.split(",")]),
            np.array([float(v) for v in loss_stored.split(",")]),
        )
        rows.append(
            {
                "sample_id": sample_id,
                "target_id": 0,
                "source_target_id": 0,
                "status": "ok",
                "source": "comsol",
                "evidence_class": "comsol_physics",
                "target_sensitivity_nm_per_riu": target["sensitivity_nm_per_riu"],
                "target_fom_per_riu": target["fom_per_riu"],
                "target_lambda_res_nm": target["lambda_res_nm"],
                "target_analyte_ri": target["analyte_ri"],
                "analyte_ri": ri,
                "lambda_res_nm": metrics.lambda_res_nm,
                "fwhm_nm": metrics.fwhm_nm,
                "wavelength_nm": wavelength_stored,
                "loss_db_per_cm": loss_stored,
                **{key: target[key] for key in GEOMETRY_COLUMNS},
            }
        )

    simulation = root / "simulation_results.csv"
    write_dataset(
        pd.DataFrame(rows), simulation,
        metadata={"source": "comsol", "evidence_class": "comsol_physics", "iteration": "closed_loop"},
    )
    limits = AcceptanceThresholds(
        max_sensitivity_error_nm_per_riu=1000.0,
        max_fom_error_per_riu=100.0,
        max_lambda_error_nm=1000.0,
        min_linearity_r2=0.0,
    )
    verification = evaluate_closed_loop_results(pd.DataFrame(rows), limits, expected_ri_points=5)
    verification_path = root / "verification.csv"
    verification.to_csv(verification_path, index=False)
    payload = {
        "backend": "comsol",
        "evidence_class": "comsol_physics",
        "selected_targets": 1,
        "accepted_targets": int(verification["accepted"].sum()),
        "ri_points": 5,
        "ri_span": 0.04,
        "acceptance_thresholds": asdict(limits),
        "inputs": {
            "comsol_model": str(model),
            "comsol_model_sha256": sha256_file(model),
            "comsol_config": str(config),
            "comsol_config_sha256": sha256_file(config),
        },
        "outputs": {
            "targets_with_geometry": str(targets),
            "simulation_results": str(simulation),
            "verification": str(verification_path),
        },
    }
    (root / "iteration_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _rewrite_simulation(root: Path, frame: pd.DataFrame) -> None:
    write_dataset(
        frame, root / "simulation_results.csv",
        metadata={"source": "comsol", "evidence_class": "comsol_physics", "iteration": "closed_loop"},
    )


def test_audit_recomputes_archived_spectra_without_claiming_physical_correctness(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    report = audit_comsol_iteration(root)
    assert report["consistency_ok"], report["errors"]
    assert report["spectrum_rows_recomputed"] == 5
    assert report["target_count"] == 1
    assert report["accepted_target_count"] == 1
    assert report["physical_correctness_verified"] is False


def test_audit_rejects_tampered_metrics_even_with_updated_csv_hash(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    frame = pd.read_csv(root / "simulation_results.csv")
    frame.loc[0, "lambda_res_nm"] += 12.0
    _rewrite_simulation(root, frame)
    report = audit_comsol_iteration(root)
    assert not report["consistency_ok"]
    assert any("stored lambda_res_nm differs" in error for error in report["errors"])


def test_audit_rejects_synthetic_source_even_if_sidecar_claims_comsol(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    frame = pd.read_csv(root / "simulation_results.csv")
    frame.loc[0, "source"] = "synthetic"
    _rewrite_simulation(root, frame)
    report = audit_comsol_iteration(root)
    assert not report["consistency_ok"]
    assert any("non-COMSOL" in error for error in report["errors"])


def test_audit_rejects_changed_model_hash(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    (tmp_path / "fixture.mph").write_text("altered fixture", encoding="utf-8")
    report = audit_comsol_iteration(root)
    assert not report["consistency_ok"]
    assert any("comsol_model hash mismatch" in error for error in report["errors"])


def test_audit_rejects_wrong_geometry_and_ri_grid(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    frame = pd.read_csv(root / "simulation_results.csv")
    frame.loc[2, "pitch_um"] = 2.5
    frame.loc[3, "analyte_ri"] = 1.48
    _rewrite_simulation(root, frame)
    report = audit_comsol_iteration(root)
    assert not report["consistency_ok"]
    assert any("pitch_um does not match" in error for error in report["errors"])
    assert any("RI grid differs" in error for error in report["errors"])


def test_audit_rejects_synthetic_manifest(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    path = root / "iteration_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["backend"] = "synthetic"
    path.write_text(json.dumps(payload), encoding="utf-8")
    report = audit_comsol_iteration(root)
    assert not report["consistency_ok"]
    assert any("COMSOL backend" in error for error in report["errors"])


def test_research_completion_audit_command_exits_nonzero_on_tampered_input(tmp_path: Path) -> None:
    root = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "research_completion.py"
    command = [
        sys.executable,
        str(script),
        "audit-comsol",
        "--iteration", str(root),
        "--json-out", str(output),
    ]
    passed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert passed.returncode == 0, passed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["consistency_ok"] is True

    (tmp_path / "fixture.mph").write_text("tampered", encoding="utf-8")
    failed = subprocess.run(command, check=False, capture_output=True, text=True)
    assert failed.returncode == 2, failed.stderr
    assert json.loads(output.read_text(encoding="utf-8"))["consistency_ok"] is False
