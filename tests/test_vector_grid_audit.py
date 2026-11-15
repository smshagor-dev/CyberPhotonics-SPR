"""Archive integrity fixtures test software logic, never Maxwell physics."""
import csv
import hashlib
import json

from sprpcf.simulation.vector_grid_audit import FALSE_GATES, audit_grid


def test_integrity_tamper_and_false_gates(tmp_path):
    files = {
        "topology.json": "{}",
        "experiment_protocol.json": json.dumps({"analyte_ris": [1.35, 1.36], "wavelengths_um": [.75, .8], "num_modes": 1}),
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
