"""Archive integrity fixtures test software logic, never Maxwell physics."""
import csv
import hashlib
import json

from sprpcf.simulation.vector_grid_audit import FALSE_GATES, audit_grid


def test_integrity_tamper_and_false_gates(tmp_path):
    files = {
        "topology.json": "{}",
        "experiment_protocol.json": json.dumps({"Au": "repository_Rakic_style_fitted_model_not_measured_constants", "analyte_ris": [1.35, 1.36], "wavelengths_um": [.75, .8], "num_modes": 1}),
        "vector_mode_overlap_diagnostics.json": "[]",
        "pcf_frozen_exploratory_grid_mesh.msh": "fixture",
    }
    for i in range(2):
        for j in range(2):
            files[f"raw_vector_eh_ri{i}_wl{j}.npz"] = "not-real-field"
    for name, value in files.items():
        (tmp_path / name).write_text(value)
    with (tmp_path / "raw_eigenmode_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ri_index", "wavelength_index", "mode_index_native_untracked"])
        writer.writeheader()
        writer.writerows({"ri_index": i, "wavelength_index": j, "mode_index_native_untracked": 0}
                         for i in range(2) for j in range(2))
    report = {gate: False for gate in FALSE_GATES}
    report.update(evidence_class="unqualified_solver_feasibility", completed=True, failures=[],
                  solver_calls_completed=4, raw_eigenmode_rows=4,
                  artifacts_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in tmp_path.iterdir()})
    path = tmp_path / "GRID_READINESS.json"
    path.write_text(json.dumps(report))
    assert audit_grid(tmp_path)["integrity_passed"]
    assert audit_grid(tmp_path)["physics_qualified"] is False
    (tmp_path / "topology.json").write_text("tampered")
    assert not audit_grid(tmp_path)["integrity_passed"]
    (tmp_path / "topology.json").write_text("{}")
    report["scientific_submission_ready"] = True
    path.write_text(json.dumps(report))
    assert not audit_grid(tmp_path)["integrity_passed"]


def test_missing_archive_fails_closed(tmp_path):
    result = audit_grid(tmp_path)
    assert not result["integrity_passed"]
    assert not result["physics_qualified"]


def test_reference_au_passes_integrity_without_promoting_sensor_physics(tmp_path):
    """Archive completeness fixtures are NOT actual vector Maxwell output."""
    from sprpcf.simulation.materials_reference import au_reference_manifest

    protocol = {
        "Au": "Johnson_Christy_1972_measured_bulk_reference",
        "Au_source_manifest": au_reference_manifest(),
        "analyte_ris": [1.35, 1.36],
        "wavelengths_um": [0.75, 0.8],
        "num_modes": 1,
    }
    raw = {
        "topology.json": "{}",
        "experiment_protocol.json": json.dumps(protocol),
        "vector_mode_overlap_diagnostics.json": "[]",
        "pcf_frozen_exploratory_grid_mesh.msh": "fixture: not numerical mesh",
        "raw_eigenmode_grid.csv": (
            "ri_index,wavelength_index,mode_index_native_untracked\n"
            "0,0,0\n0,1,0\n1,0,0\n1,1,0\n"
        ),
    }
    for i in range(2):
        for j in range(2):
            raw[f"raw_vector_eh_ri{i}_wl{j}.npz"] = "fixture: not a vector field"
    for name, value in raw.items():
        (tmp_path / name).write_text(value)
    report = {gate: False for gate in FALSE_GATES}
    report.update(
        evidence_class="unqualified_solver_feasibility",
        Au_material_measured_data_provenance_verified=True,
        Au_material_reference_only_not_sensor_film_measurement=True,
        completed=True, failures=[], solver_calls_completed=4,
        raw_eigenmode_rows=4,
        artifacts_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in tmp_path.iterdir()
        },
    )
    manifest = tmp_path / "GRID_READINESS.json"
    manifest.write_text(json.dumps(report))
    result = audit_grid(tmp_path)
    assert result["integrity_passed"] is True
    assert result["physics_qualified"] is False

    report["Au_material_measured_data_provenance_verified"] = False
    manifest.write_text(json.dumps(report))
    assert "Au source/provenance flag mismatch" in audit_grid(tmp_path)["errors"]
    report["Au_material_measured_data_provenance_verified"] = True
    report["pml_validated"] = True
    manifest.write_text(json.dumps(report))
    assert not audit_grid(tmp_path)["integrity_passed"]

    report["pml_validated"] = False
    manifest.write_text(json.dumps(report))
    protocol["Au_source_manifest"]["dataset_sha256"] = "0" * 64
    file = tmp_path / "experiment_protocol.json"
    file.write_text(json.dumps(protocol))
    report["artifacts_sha256"][file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(report))
    assert "measured Au manifest does not match pinned source" in audit_grid(tmp_path)["errors"]
