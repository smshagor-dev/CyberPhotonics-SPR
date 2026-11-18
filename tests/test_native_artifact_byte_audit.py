import hashlib
import json
from scripts.native_artifact_byte_audit import audit


def make(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    (root / "field.bin").write_bytes(b"native bytes")
    manifest = {"artifacts": [{"path": "field.bin",
                "sha256": hashlib.sha256(b"native bytes").hexdigest(),
                "bytes": 12}], "physics_qualified": False,
                "publication_ready": False, "comsol_verified": False}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, root, manifest


def test_valid(tmp_path):
    p, r, _ = make(tmp_path)
    assert audit(p, r)["integrity_verified"]


def test_tampered(tmp_path):
    p, r, _ = make(tmp_path)
    (r / "field.bin").write_bytes(b"wrong")
    assert not audit(p, r)["integrity_verified"]


def test_traversal(tmp_path):
    p, r, m = make(tmp_path)
    m["artifacts"][0]["path"] = "../manifest.json"
    p.write_text(json.dumps(m))
    assert not audit(p, r)["integrity_verified"]


def test_symlink(tmp_path):
    p, r, _ = make(tmp_path)
    (r / "field.bin").unlink()
    (r / "field.bin").symlink_to(p)
    assert not audit(p, r)["integrity_verified"]


def test_duplicate(tmp_path):
    p, r, m = make(tmp_path)
    m["artifacts"].append(dict(m["artifacts"][0]))
    p.write_text(json.dumps(m))
    assert not audit(p, r)["integrity_verified"]


def test_claim(tmp_path):
    p, r, m = make(tmp_path)
    m["physics_qualified"] = True
    p.write_text(json.dumps(m))
    assert not audit(p, r)["integrity_verified"]


def test_bool_size(tmp_path):
    p, r, m = make(tmp_path)
    m["artifacts"][0]["bytes"] = True
    p.write_text(json.dumps(m))
    assert not audit(p, r)["integrity_verified"]
