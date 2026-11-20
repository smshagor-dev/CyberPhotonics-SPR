"""Regression tests for the unqualified native-grid integrity audit."""
import csv
import hashlib
import json
from scripts.native_grid_integrity import audit


FLAGS = ("sensor_geometry_peer_reviewed_and_frozen", "pml_validated",
         "mesh_convergence_verified", "mode_tracking_physically_verified",
         "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
         "scientific_submission_ready")


def make_fixture(root):
    root.mkdir()
    protocol = {"wavelengths_um": [0.75, 0.8], "analyte_ris": [1.35], "num_modes": 1}
    (root / "experiment_protocol.json").write_text(json.dumps(protocol))
    rows = [{"ri_index": 0, "wavelength_index": j,
             "mode_index_native_untracked": 0, "analyte_ri": 1.35,
             "wavelength_um": wl, "n_eff_real": 1.4, "n_eff_imag": 0.01,
             "raw_attenuation_db_per_cm_unqualified": 2}
            for j, wl in enumerate(protocol["wavelengths_um"])]
    with (root / "raw_eigenmode_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    hashes = {}
    for j in range(2):
        name = f"raw_vector_eh_ri0_wl{j}.npz"
        (root / name).write_bytes(b"fixture only")
        hashes[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    for name in ("experiment_protocol.json", "raw_eigenmode_grid.csv",
                 "pcf_frozen_exploratory_grid_mesh.msh", "topology.json",
                 "vector_mode_overlap_diagnostics.json"):
        path = root / name
        if not path.exists():
            path.write_bytes(b"fixture only")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    (root / "GRID_READINESS.json").write_text(json.dumps({
        "artifacts_sha256": hashes, **{flag: False for flag in FLAGS}
    }))


def test_fixture_integrity_does_not_qualify_physics(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    result = audit(root)
    assert result["integrity_passed"]
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False


def test_tampered_field_archive_rejected(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    (root / "raw_vector_eh_ri0_wl0.npz").write_bytes(b"tampered")
    assert not audit(root)["integrity_passed"]


def test_missing_eigenmode_rejected(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    path = root / "raw_eigenmode_grid.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:2]) + "\n")
    assert not audit(root)["integrity_passed"]


def test_publication_flag_rejected(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    path = root / "GRID_READINESS.json"
    data = json.loads(path.read_text())
    data["scientific_submission_ready"] = True
    path.write_text(json.dumps(data))
    assert not audit(root)["integrity_passed"]


def test_unregistered_protocol_rejected(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    report = root / "GRID_READINESS.json"
    data = json.loads(report.read_text())
    del data["artifacts_sha256"]["experiment_protocol.json"]
    report.write_text(json.dumps(data))
    assert not audit(root)["integrity_passed"]


def test_symlink_artifact_rejected(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    path = root / "topology.json"
    target = root / "target.json"
    target.write_bytes(path.read_bytes())
    path.unlink()
    path.symlink_to(target)
    assert not audit(root)["integrity_passed"]


def test_unsorted_wavelength_grid_rejected(tmp_path):
    root = tmp_path / "grid"
    make_fixture(root)
    path = root / "experiment_protocol.json"
    data = json.loads(path.read_text())
    data["wavelengths_um"] = [0.8, 0.75]
    path.write_text(json.dumps(data))
    assert not audit(root)["integrity_passed"]
