import hashlib
from sprpcf.evidence.source_native_gate import REQUIRED, audit_source_bundle

def bundle(tmp_path):
    artifacts = {}
    for role in REQUIRED:
        p = tmp_path / (role + ".bin")
        p.write_bytes(role.encode())
        artifacts[role] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    return {"source": "open_femwell", "synthetic": False, "fixture": False, "artifacts": artifacts}

def test_integrity_does_not_qualify_physics(tmp_path):
    result = audit_source_bundle(tmp_path, bundle(tmp_path))
    assert result["integrity_passed"] and result["publication_qualified"] is False

def test_tamper_rejected(tmp_path):
    manifest = bundle(tmp_path)
    (tmp_path / "mesh.bin").write_bytes(b"changed")
    assert "hash_mismatch_mesh" in audit_source_bundle(tmp_path, manifest)["failures"]

def test_fixture_rejected(tmp_path):
    manifest = bundle(tmp_path)
    manifest["fixture"] = True
    assert not audit_source_bundle(tmp_path, manifest)["integrity_passed"]

def test_path_traversal_rejected(tmp_path):
    manifest = bundle(tmp_path)
    manifest["artifacts"]["mesh"]["path"] = "../outside"
    assert "missing_or_unsafe_mesh" in audit_source_bundle(tmp_path, manifest)["failures"]

def test_comsol_source_requires_mph_hash(tmp_path):
    manifest = bundle(tmp_path)
    manifest["source"] = "comsol"
    assert "missing_comsol_model_hash" in audit_source_bundle(tmp_path, manifest)["failures"]

def test_missing_native_fields_rejected(tmp_path):
    manifest = bundle(tmp_path)
    del manifest["artifacts"]["eigenfields"]
    assert "missing_eigenfields" in audit_source_bundle(tmp_path, manifest)["failures"]
