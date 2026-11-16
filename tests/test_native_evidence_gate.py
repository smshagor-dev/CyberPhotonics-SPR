"""Constructed software fixtures only: NOT Maxwell physics evidence."""
import hashlib
import json
from research.native_evidence_gate import REQUIRED, audit


def build(tmp_path, source="native_fem"):
    files = {}
    for key in REQUIRED:
        file = tmp_path / (key + ".dat")
        file.write_bytes(key.encode())
        files[key] = {"path": file.name, "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
    data = {"source_class": source, "solver_name": "femwell", "solver_version": "test",
            "git_commit": "test", "geometry_id": "test", "geometry_frozen": True,
            "boundary_conditions": "PEC exploratory", "mode_tracking_method": "overlap",
            "analyte_ri": 1.35, "wavelength_nm": 800,
            "physics_qualified": False, "publication_ready": False, "artifacts": files}
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(data))
    return manifest, data


def test_valid_integrity_does_not_qualify_physics(tmp_path):
    p, _ = build(tmp_path)
    r = audit(p)
    assert r["integrity_traceability_pass"]
    assert r["physics_qualified"] is False and r["publication_ready"] is False


def test_mutation_rejected(tmp_path):
    p, _ = build(tmp_path)
    (tmp_path / "fields_e.dat").write_bytes(b"modified")
    assert "fields_e_digest_mismatch" in audit(p)["failures"]


def test_fixture_rejected(tmp_path):
    p, _ = build(tmp_path, "fixture")
    assert "nonphysical_source" in audit(p)["failures"]


def test_comsol_provenance_rejected(tmp_path):
    p, data = build(tmp_path, "comsol")
    assert "comsol_model_hash_missing" in audit(p)["failures"]
    data["source_class"] = "native_fem"
    data["comsol_model_sha256"] = "0" * 64
    p.write_text(json.dumps(data))
    assert "comsol_provenance_mismatch" in audit(p)["failures"]


def test_escape_rejected(tmp_path):
    p, data = build(tmp_path)
    data["artifacts"]["mesh"]["path"] = "../outside"
    p.write_text(json.dumps(data))
    assert "mesh_unsafe_path" in audit(p)["failures"]


def test_unsafe_publication_flag_rejected(tmp_path):
    p, data = build(tmp_path)
    data["publication_ready"] = True
    p.write_text(json.dumps(data))
    assert "unsafe_publication_flags" in audit(p)["failures"]


def test_missing_native_field_rejected(tmp_path):
    p, data = build(tmp_path)
    del data["artifacts"]["fields_h"]
    p.write_text(json.dumps(data))
    assert "fields_h_record_missing" in audit(p)["failures"]
