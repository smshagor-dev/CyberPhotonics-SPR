"""Synthetic archives exercise integrity rejection only; never physics evidence."""
import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_femwell_grid_archive import audit


def make_archive(root):
    names = ["topology.json", "experiment_protocol.json",
             "pcf_frozen_exploratory_grid_mesh.msh", "raw_eigenmode_grid.csv",
             "vector_mode_overlap_diagnostics.json",
             "raw_vector_eh_ri0_wl0.npz", "raw_vector_eh_ri0_wl1.npz",
             "raw_vector_eh_ri1_wl0.npz", "raw_vector_eh_ri1_wl1.npz"]
    for name in names:
        (root / name).write_bytes(b"fixture-not-physics")
    with (root / "raw_eigenmode_grid.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["ri_index", "wavelength_index", "mode_index_native_untracked"])
        writer.writerows((i, j, k) for i in range(2) for j in range(2) for k in range(2))
    report = {"analyte_ris": [1.35, 1.36], "wavelengths_um": [0.8, 0.81],
              "num_modes_requested": 2, "solver_calls_completed": 4,
              "raw_eigenmode_rows": 8, "completed": True, "failures": [],
              "scientific_submission_ready": False,
              "artifacts_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                   for name in names}}
    (root / "GRID_READINESS.json").write_text(json.dumps(report))


def test_archive_integrity_and_tampering(tmp_path):
    make_archive(tmp_path)
    assert audit(tmp_path)["integrity_pass"]
    (tmp_path / "raw_vector_eh_ri0_wl0.npz").write_bytes(b"changed")
    assert not audit(tmp_path)["integrity_pass"]


def test_incomplete_solver_and_duplicate_modes(tmp_path):
    make_archive(tmp_path)
    report_path = tmp_path / "GRID_READINESS.json"
    report = json.loads(report_path.read_text())
    report["solver_calls_completed"] = 3
    report_path.write_text(json.dumps(report))
    assert not audit(tmp_path)["integrity_pass"]
    make_archive(tmp_path)
    with (tmp_path / "raw_eigenmode_grid.csv").open("a") as handle:
        handle.write("0,0,0\n")
    report = json.loads(report_path.read_text())
    report["artifacts_sha256"]["raw_eigenmode_grid.csv"] = hashlib.sha256(
        (tmp_path / "raw_eigenmode_grid.csv").read_bytes()).hexdigest()
    report_path.write_text(json.dumps(report))
    assert any("duplicate" in x for x in audit(tmp_path)["errors"])
