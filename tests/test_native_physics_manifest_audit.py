import hashlib
import json
from pathlib import Path
from tools.native_physics_manifest_audit import audit, REQUIRED

def valid(tmp_path):
    assets = {}
    for key in REQUIRED:
        p = tmp_path / (key + ".dat")
        p.write_bytes(key.encode())
        assets[key] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
    m = {"source": "femwell-native", "evidence_class": "native_maxwell", "fixture": False,
         "geometry_frozen": True, "material_citation": "doi:placeholder",
         "mesh_converged": True, "pml_converged": True, "domain_converged": True,
         "mode_overlap_verified": True, "independent_solver_verified": True, "assets": assets}
    f = tmp_path / "manifest.json"
    f.write_text(json.dumps(m))
    return f, m

def test_complete_contract(tmp_path):
    f, _ = valid(tmp_path)
    assert audit(f) == {"qualified": True, "errors": [], "publication_ready": False}

def test_tamper(tmp_path):
    f, _ = valid(tmp_path)
    (tmp_path / "mesh.dat").write_text("changed")
    assert "asset_digest_mismatch:mesh" in audit(f)["errors"]

def test_escape(tmp_path):
    f, m = valid(tmp_path)
    m["assets"]["mesh"]["path"] = "../outside"
    f.write_text(json.dumps(m))
    assert "unsafe_or_missing_asset:mesh" in audit(f)["errors"]

def test_fixture(tmp_path):
    f, m = valid(tmp_path)
    m["fixture"] = True
    f.write_text(json.dumps(m))
    assert not audit(f)["qualified"]

def test_missing_convergence(tmp_path):
    f, m = valid(tmp_path)
    m["pml_converged"] = False
    f.write_text(json.dumps(m))
    assert "unverified_convergence" in audit(f)["errors"]
