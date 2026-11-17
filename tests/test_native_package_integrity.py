import hashlib
import importlib.util
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1] / "research_validation" / "native_package_integrity.py"
spec = importlib.util.spec_from_file_location("native_package_integrity", MODULE)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def sample(tmp_path):
    p = tmp_path / "fields.npz"
    p.write_bytes(b"TEST FIXTURE - NOT SOLVER DATA")
    return {"source": "femwell", "comsol_verified": False, "fixture": False,
            "geometry_frozen": True, "geometry_reference": "reference:required",
            "checks": {key: True for key in module.REQUIRED},
            "artifacts": [{"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}]}

def test_structural_success_never_qualifies_physics(tmp_path):
    result = module.audit(sample(tmp_path), tmp_path)
    assert result["evidence_package_structurally_complete"]
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False

def test_every_missing_check_fails(tmp_path):
    for key in module.REQUIRED:
        data = sample(tmp_path)
        data["checks"][key] = False
        assert not module.audit(data, tmp_path)["evidence_package_structurally_complete"]

def test_digest_mismatch(tmp_path):
    data = sample(tmp_path)
    data["artifacts"][0]["sha256"] = "0" * 64
    assert not module.audit(data, tmp_path)["evidence_package_structurally_complete"]

def test_path_escape(tmp_path):
    data = sample(tmp_path)
    data["artifacts"][0]["path"] = "../outside"
    assert not module.audit(data, tmp_path)["evidence_package_structurally_complete"]

def test_fixture_rejected(tmp_path):
    data = sample(tmp_path)
    data["fixture"] = True
    assert not module.audit(data, tmp_path)["evidence_package_structurally_complete"]

def test_comsol_promotion_rejected(tmp_path):
    data = sample(tmp_path)
    data["comsol_verified"] = True
    assert not module.audit(data, tmp_path)["evidence_package_structurally_complete"]

def test_duplicate_artifacts_rejected(tmp_path):
    data = sample(tmp_path)
    data["artifacts"] *= 2
    assert not module.audit(data, tmp_path)["evidence_package_structurally_complete"]
