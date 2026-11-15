"""Software-only tests for domain comparison logic; no FEM fixture is evidence."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import compare_domain_grids as study


def test_nearest_pairing_never_asserts_physical_mode_identity():
    match = study.eigenvalue_pairs([1.4 + 0.01j, 1.6 + 0.02j],
                                  [1.601 + 0.02j, 1.399 + 0.01j])
    assert [(a["small_mode_index_unverified"], a["large_mode_index_unverified"])
            for a in match["pairs"]] == [(0, 1), (1, 0)]
    assert not match["assignment_physics_verified"]
    assert match["eigenvalue_assignment_ambiguous"] is False
    assert all(not p["core_spp_identity_verified"] for p in match["pairs"])


def test_pairing_rejects_incomplete_and_nonfinite_modes():
    with pytest.raises(ValueError, match="equal sets"):
        study.eigenvalue_pairs([1 + 1j], [1 + 1j])
    with pytest.raises(ValueError, match="nonfinite"):
        study.eigenvalue_pairs([1 + 1j, complex(float("nan"), 0)], [1 + 1j, 2 + 2j])


def _make_fixtures(tmp_path, monkeypatch):
    """Bypasses archive auditor ONLY here; fake fields never enter an actual run."""
    monkeypatch.setattr(study, "verify_measured_au_grid",
                        lambda root: {"passed": True, "errors": []})
    paths = [tmp_path / "small", tmp_path / "large"]
    for idx, root in enumerate(paths):
        root.mkdir()
        protocol = {
            "wavelengths_um": [0.75, 0.80], "analyte_ris": [1.35, 1.36],
            "num_modes": 2, "gold_mesh_resolution_um": 0.025,
            "Au": "Johnson_Christy_1972_measured_bulk_reference",
            "Au_source_manifest": {"dataset_sha256": "software-fixture"},
            "silica": "three_term_Sellmeier",
            "temporal_convention": "exp(-i*omega*t)",
        }
        topology = {"geometry": {"pitch_um": 2.0,
                                "domain_halfwidth_um": [5.5, 7.0][idx]}}
        report = {"solver_version": "fixture-solver",
                  "mesh_elements": [400, 500][idx],
                  "artifacts_sha256": {
                      "pcf_frozen_exploratory_grid_mesh.msh": str(idx) * 64}}
        (root / "experiment_protocol.json").write_text(json.dumps(protocol))
        (root / "topology.json").write_text(json.dumps(topology))
        (root / "GRID_READINESS.json").write_text(json.dumps(report))
        with (root / "raw_eigenmode_grid.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "ri_index", "wavelength_index", "mode_index_native_untracked",
                "n_eff_real", "n_eff_imag",
            ])
            writer.writeheader()
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        writer.writerow({
                            "ri_index": i, "wavelength_index": j,
                            "mode_index_native_untracked": k,
                            "n_eff_real": 1.45 + 0.001 * k + 0.0001 * idx,
                            "n_eff_imag": 0.00001 + 0.000001 * k,
                        })
    return paths


def test_valid_fixture_stays_unqualified(tmp_path, monkeypatch):
    small, large = _make_fixtures(tmp_path, monkeypatch)
    report = study.compare_domain_grids(small, large)
    assert report["mode_pairs_diagnostic_only"] == 8
    assert "ambiguous_gridpoint_pairings" in report
    assert report["max_complex_n_eff_abs_delta"] > 0
    assert not report["PML_present"]
    assert not report["domain_convergence_verified"]
    assert not report["scientific_submission_ready"]


def test_changed_gold_or_geometry_rejected(tmp_path, monkeypatch):
    small, large = _make_fixtures(tmp_path, monkeypatch)
    protocol_path = large / "experiment_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    protocol["gold_mesh_resolution_um"] = 0.02
    protocol_path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="gold_mesh_resolution"):
        study.compare_domain_grids(small, large)
    protocol["gold_mesh_resolution_um"] = 0.025
    protocol_path.write_text(json.dumps(protocol))
    topology_path = large / "topology.json"
    topo = json.loads(topology_path.read_text())
    topo["geometry"]["pitch_um"] = 2.1
    topology_path.write_text(json.dumps(topo))
    with pytest.raises(ValueError, match="core/Au/air"):
        study.compare_domain_grids(small, large)


def test_same_mesh_or_reversed_domains_rejected(tmp_path, monkeypatch):
    small, large = _make_fixtures(tmp_path, monkeypatch)
    manifest_path = large / "GRID_READINESS.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts_sha256"]["pcf_frozen_exploratory_grid_mesh.msh"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identical mesh"):
        study.compare_domain_grids(small, large)
    with pytest.raises(ValueError, match="domain halfwidth must increase"):
        study.compare_domain_grids(large, small)


def test_nearly_degenerate_complex_eigenvalues_fail_ambiguity_diagnostic():
    modes = study.eigenvalue_pairs(
        [1.4000000 + 1e-7j, 1.4000001 + 2e-7j],
        [1.4005000 + 1e-7j, 1.4005001 + 2e-7j],
    )
    assert modes["eigenvalue_assignment_ambiguous"] is True
    assert modes["assignment_physics_verified"] is False
