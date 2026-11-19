"""Software-only source/provenance and native-array regression tests.

All arrays built here are synthetic fixtures; not electromagnetic measurements.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import verify_measured_au_fem_grid as checker
from sprpcf.simulation.materials_reference import au_johnson_christy_permittivity, au_reference_manifest
from sprpcf.simulation.open_vector_grid import run_vector_grid


def _fixture(tmp_path, monkeypatch):
    # Isolate material fingerprint checks from the existing archive-integrity gate.
    # This bypass is ONLY for software unit tests, never real experiment execution.
    monkeypatch.setattr(checker, "audit_grid", lambda p: {"passed": True, "errors": []})
    protocol = {
        "Au": "Johnson_Christy_1972_measured_bulk_reference",
        "Au_source_manifest": au_reference_manifest(),
    }
    (tmp_path / "experiment_protocol.json").write_text(json.dumps(protocol))
    report = {
        "Au_material_measured_data_provenance_verified": False,
        "Au_reference_dataset_selected": True,
        "Au_reference_transcription_independently_verified": False,
        "scientific_submission_ready": False,
        "pml_validated": False,
        "mode_tracking_physically_verified": False,
    }
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps(report))
    eps = complex(au_johnson_christy_permittivity(0.8))
    row = {
        "ri_index": "0", "wavelength_index": "0",
        "wavelength_um": "0.8", "gold_epsilon_real": str(eps.real),
        "gold_epsilon_imag": str(eps.imag), "mode_index_native_untracked": "0",
        "vector_field_e_dofs": "2", "vector_field_h_dofs": "2",
    }
    path = tmp_path / "raw_eigenmode_grid.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    np.savez_compressed(tmp_path / "raw_vector_eh_ri0_wl0.npz",
                        mode_0_E=np.array([1.0 + 1j, 2j]),
                        mode_0_H=np.array([2.0 + 1j, 3j]))
    return protocol, row


def test_fixture_fingerprint_passes_software_gate_not_publication(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    result = checker.verify_measured_au_grid(tmp_path)
    assert result["passed"] is True
    assert result["publication_qualified"] is False
    assert result["scientific_submission_ready"] is False


def test_changed_bulk_gold_constant_is_rejected(tmp_path, monkeypatch):
    _, row = _fixture(tmp_path, monkeypatch)
    row["gold_epsilon_real"] = str(float(row["gold_epsilon_real"]) + 0.1)
    with (tmp_path / "raw_eigenmode_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    result = checker.verify_measured_au_grid(tmp_path)
    assert not result["passed"]
    assert "does not reproduce source material" in str(result["errors"])


def test_modified_source_manifest_is_rejected(tmp_path, monkeypatch):
    protocol, _ = _fixture(tmp_path, monkeypatch)
    protocol["Au_source_manifest"]["dataset_sha256"] = "0" * 64
    (tmp_path / "experiment_protocol.json").write_text(json.dumps(protocol))
    result = checker.verify_measured_au_grid(tmp_path)
    assert not result["passed"]
    assert "source manifest differs" in str(result["errors"])


def test_corrupt_native_field_is_rejected(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    (tmp_path / "raw_vector_eh_ri0_wl0.npz").write_text("not an NPZ field archive")
    result = checker.verify_measured_au_grid(tmp_path)
    assert result["passed"] is False
    assert result["publication_qualified"] is False


def test_missing_complex_magnetic_field_is_rejected(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    np.savez_compressed(tmp_path / "raw_vector_eh_ri0_wl0.npz",
                        mode_0_E=np.array([1.0 + 1j, 2j]))
    assert not checker.verify_measured_au_grid(tmp_path)["passed"]


def test_unknown_gold_source_fails_before_meshing(tmp_path):
    with pytest.raises(ValueError, match="gold_source"):
        run_vector_grid(tmp_path, wavelengths_um=(0.75, 0.8),
                        analyte_ris=(1.35, 1.36), gold_source="invented")
    assert not (tmp_path / "GRID_READINESS.json").exists()


def test_unverified_au_transcription_cannot_self_certify(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    report_path = tmp_path / "GRID_READINESS.json"
    report = json.loads(report_path.read_text())
    report["Au_material_measured_data_provenance_verified"] = True
    report_path.write_text(json.dumps(report))
    result = checker.verify_measured_au_grid(tmp_path)
    assert not result["passed"]
    assert "must not claim verified provenance" in str(result["errors"])


def test_missing_au_selection_fails_closed(tmp_path, monkeypatch):
    _fixture(tmp_path, monkeypatch)
    report_path = tmp_path / "GRID_READINESS.json"
    report = json.loads(report_path.read_text())
    del report["Au_reference_dataset_selected"]
    report_path.write_text(json.dumps(report))
    assert not checker.verify_measured_au_grid(tmp_path)["passed"]
