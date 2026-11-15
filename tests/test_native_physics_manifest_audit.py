import hashlib
import json
from tools.native_physics_manifest_audit import audit, REQUIRED


def valid(tmp_path):
    assets = {}
    for key in REQUIRED:
        p = tmp_path / (key + ".dat")
        p.write_bytes(key.encode())
        assets[key] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    m = {"source": "femwell-native", "evidence_class": "native_maxwell", "fixture": False,
         "geometry_frozen": True, "material_citation": "doi:10.1103/PhysRevB.6.4370",
         "mesh_converged": True, "pml_converged": True, "domain_converged": True,
         "mode_overlap_verified": True, "independent_solver_verified": True, "assets": assets}
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps(m))
    return f, m


def test_integrity_is_not_physics_qualification(tmp_path):
    f, _ = valid(tmp_path)
    assert audit(f) == {"integrity_verified": True, "qualified": False, "errors": [], "publication_ready": False}


def test_tamper(tmp_path):
    f, _ = valid(tmp_path)
    (tmp_path / "mesh.dat").write_text("changed")
    assert "asset_digest_mismatch:mesh" in audit(f)["errors"]


def test_escape(tmp_path):
    f, m = valid(tmp_path)
    m["assets"]["mesh"]["path"] = "../outside"
    f.write_text(json.dumps(m))
    assert "unsafe_or_invalid_asset:mesh" in audit(f)["errors"]


def test_fixture(tmp_path):
    f, m = valid(tmp_path)
    m["fixture"] = True
    f.write_text(json.dumps(m))
    assert not audit(f)["integrity_verified"]


def test_missing_convergence(tmp_path):
    f, m = valid(tmp_path)
    m["pml_converged"] = False
    f.write_text(json.dumps(m))
    assert "unverified_convergence" in audit(f)["errors"]


def test_placeholder_citation(tmp_path):
    f, m = valid(tmp_path)
    m["material_citation"] = "doi:placeholder"
    f.write_text(json.dumps(m))
    assert "missing_physics_provenance" in audit(f)["errors"]


def test_malformed_assets(tmp_path):
    f, m = valid(tmp_path)
    m["assets"] = None
    f.write_text(json.dumps(m))
    assert "invalid_assets" in audit(f)["errors"]


def test_invalid_digest(tmp_path):
    f, m = valid(tmp_path)
    m["assets"]["mesh"]["sha256"] = "not-sha"
    f.write_text(json.dumps(m))
    assert "unsafe_or_invalid_asset:mesh" in audit(f)["errors"]
