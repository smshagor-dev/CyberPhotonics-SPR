"""Software-only tests of an ideal planar SPP equation and fail-closed gate.

Fixture eigenvalues are invented solely for testing and are NOT physics evidence.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import planar_spp_reference as planar


def test_passive_measured_reference_planar_formula():
    n = planar.planar_au_analyte_spp_index(0.8, 1.36)
    assert n.real == pytest.approx(1.41524750, abs=1e-7)
    assert n.imag == pytest.approx(0.00367443, abs=1e-7)
    assert n.real > 1.36
    assert n.imag > 0


def test_invalid_optical_inputs_rejected_without_extrapolation():
    for wavelength, ri in ((0.8, 0.5), (2.0, 1.36), (float("nan"), 1.36)):
        with pytest.raises(ValueError):
            planar.planar_au_analyte_spp_index(wavelength, ri)


def test_missing_actual_native_archive_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="native Au-FEM archive failed"):
        planar.audit_native_planar_reference(tmp_path)


def test_fixture_analytic_diagnostic_never_promoted_to_solver_comparison(tmp_path, monkeypatch):
    # Bypass real archive audit ONLY in this synthetic software unit test.
    monkeypatch.setattr(planar, "verify_measured_au_grid",
                        lambda p: {"passed": True, "errors": []})
    (tmp_path / "experiment_protocol.json").write_text(json.dumps({
        "wavelengths_um": [0.80], "analyte_ris": [1.36], "num_modes": 2,
    }))
    (tmp_path / "GRID_READINESS.json").write_text(json.dumps({
        "artifacts_sha256": {"raw_eigenmode_grid.csv": "fixture",
                             "pcf_frozen_exploratory_grid_mesh.msh": "fixture"},
        "full_vector_maxwell_solver": "fixture_not_a_solver",
    }))
    with (tmp_path / "raw_eigenmode_grid.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["analyte_ri", "wavelength_um",
                                         "n_eff_real", "n_eff_imag"])
        w.writeheader()
        w.writerow({"analyte_ri": 1.36, "wavelength_um": 0.8,
                    "n_eff_real": 1.45, "n_eff_imag": 1e-7})
        w.writerow({"analyte_ri": 1.36, "wavelength_um": 0.8,
                    "n_eff_real": 1.46, "n_eff_imag": 2e-7})
    result = planar.audit_native_planar_reference(tmp_path)
    assert result["sample_count"] == 1
    assert result["not_an_independent_full_vector_pcf_solver"]
    assert not result["core_spp_phase_matching_verified"]
    assert not result["independent_solver_verified"]
    assert not result["scientific_submission_ready"]
