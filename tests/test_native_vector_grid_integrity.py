"""Constructed fixtures test software integrity only, never Maxwell physics."""
import csv
import hashlib
import json

from scripts.verify_native_vector_grid import audit


def fixture(root):
    protocol = {"wavelengths_um": [0.75, 0.8], "analyte_ris": [1.35, 1.36], "num_modes": 2}
    (root / "experiment_protocol.json").write_text(json.dumps(protocol))
    fields = ["ri_index", "wavelength_index", "mode_index_native_untracked", "analyte_ri", "wavelength_um"]
    with (root / "raw_eigenmode_grid.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    writer.writerow(dict(ri_index=i, wavelength_index=j, mode_index_native_untracked=k,
                                         analyte_ri=protocol["analyte_ris"][i],
                                         wavelength_um=protocol["wavelengths_um"][j]))
    names = ["pcf_frozen_exploratory_grid_mesh.msh", "topology.json",
             "experiment_protocol.json", "raw_eigenmode_grid.csv",
             "vector_mode_overlap_diagnostics.json"]
    names += [f"raw_vector_eh_ri{i}_wl{j}.npz" for i in range(2) for j in range(2)]
    for name in names:
        if not (root / name).exists():
            (root / name).write_bytes(b"constructed fixture")
    digests = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}
    (root / "GRID_READINESS.json").write_text(json.dumps({
        "solver_calls_completed": 4, "completed": True,
        "scientific_submission_ready": False, "artifacts_sha256": digests}))


def test_complete_fixture_is_not_physics(tmp_path):
    fixture(tmp_path)
    result = audit(tmp_path)
    assert result["integrity_verified"]
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False


def test_corrupt_native_field_rejected(tmp_path):
    fixture(tmp_path)
    (tmp_path / "raw_vector_eh_ri0_wl0.npz").write_bytes(b"tampered")
    assert not audit(tmp_path)["integrity_verified"]


def test_incomplete_solver_campaign_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "GRID_READINESS.json"
    data = json.loads(path.read_text())
    data["solver_calls_completed"] = 3
    path.write_text(json.dumps(data))
    assert "incomplete solver campaign" in audit(tmp_path)["errors"]


def test_duplicate_eigenmode_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "raw_eigenmode_grid.csv"
    lines = path.read_text().splitlines()
    lines[-1] = lines[-2]
    path.write_text("\n".join(lines) + "\n")
    assert "duplicate mode identities" in audit(tmp_path)["errors"]


def test_missing_native_inputs_rejected(tmp_path):
    assert not audit(tmp_path)["integrity_verified"]


def test_malformed_manifest_types_fail_closed(tmp_path):
    fixture(tmp_path)
    p = tmp_path / "experiment_protocol.json"
    p.write_text(json.dumps({"wavelengths_um": "bad", "analyte_ris": None, "num_modes": None}))
    assert audit(tmp_path)["integrity_verified"] is False
    fixture(tmp_path)
    p = tmp_path / "GRID_READINESS.json"
    data = json.loads(p.read_text())
    data["artifacts_sha256"] = None
    p.write_text(json.dumps(data))
    assert audit(tmp_path)["integrity_verified"] is False


def test_unsafe_publication_claim_fails_closed(tmp_path):
    fixture(tmp_path)
    p = tmp_path / "GRID_READINESS.json"
    data = json.loads(p.read_text())
    data["comsol_verified"] = True
    p.write_text(json.dumps(data))
    result = audit(tmp_path)
    assert not result["integrity_verified"]
    assert "unsafe publication or COMSOL claim" in result["errors"]


def test_nonfinite_protocol_coordinate_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "experiment_protocol.json"
    data = json.loads(path.read_text())
    data["wavelengths_um"][0] = float("nan")
    path.write_text(json.dumps(data))
    assert not audit(tmp_path)["integrity_verified"]


def test_duplicate_protocol_coordinate_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "experiment_protocol.json"
    data = json.loads(path.read_text())
    data["analyte_ris"][1] = data["analyte_ris"][0]
    path.write_text(json.dumps(data))
    assert not audit(tmp_path)["integrity_verified"]


def test_symlinked_native_artifact_rejected(tmp_path):
    fixture(tmp_path)
    source = tmp_path / "raw_vector_eh_ri0_wl0.npz"
    target = tmp_path / "other.npz"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    assert not audit(tmp_path)["integrity_verified"]


def test_noncanonical_mode_index_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "raw_eigenmode_grid.csv"
    text = path.read_text()
    path.write_text(text.replace("0,0,0,", "0,0,0.0,", 1))
    assert not audit(tmp_path)["integrity_verified"]


def test_numeric_equivalent_grid_coordinates_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "experiment_protocol.json"
    data = json.loads(path.read_text())
    data["wavelengths_um"] = [1, 1.0]
    path.write_text(json.dumps(data))
    assert not audit(tmp_path)["integrity_verified"]


def test_noncanonical_leading_zero_index_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "raw_eigenmode_grid.csv"
    path.write_text(path.read_text().replace("0,0,0,", "00,0,0,", 1))
    assert not audit(tmp_path)["integrity_verified"]


def test_boolean_solver_count_rejected(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "GRID_READINESS.json"
    data = json.loads(path.read_text())
    data["solver_calls_completed"] = True
    path.write_text(json.dumps(data))
    assert not audit(tmp_path)["integrity_verified"]
