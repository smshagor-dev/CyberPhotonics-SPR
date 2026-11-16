import hashlib
import tempfile
import unittest
from pathlib import Path
from native_solver_source_evidence import audit, EvidenceError, REQUIRED, REQUIRED_CHECKS

class NativeSourceEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.artifacts = {}
        for role in REQUIRED:
            p = self.root / (role + ".dat")
            p.write_bytes(role.encode())
            self.artifacts[role] = {"path": p.name, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        self.registry = {"source": "femwell", "solver_version": "recorded-version",
                         "artifacts": self.artifacts,
                         "checks": {k: False for k in REQUIRED_CHECKS},
                         "physics_qualified": False, "publication_ready": False}
    def tearDown(self):
        self.tmp.cleanup()
    def test_valid_integrity_is_not_physics(self):
        result = audit(self.root, self.registry)
        self.assertTrue(result["integrity_verified"])
        self.assertFalse(result["physics_qualified"])
    def test_tamper_rejected(self):
        (self.root / "mesh.dat").write_text("changed")
        with self.assertRaises(EvidenceError): audit(self.root, self.registry)
    def test_comsol_mislabel_rejected(self):
        self.registry["comsol_verified"] = True
        with self.assertRaises(EvidenceError): audit(self.root, self.registry)
    def test_qualification_promotion_rejected(self):
        self.registry["publication_ready"] = True
        with self.assertRaises(EvidenceError): audit(self.root, self.registry)
    def test_missing_numerical_check_rejected(self):
        del self.registry["checks"]["pml_convergence"]
        with self.assertRaises(EvidenceError): audit(self.root, self.registry)
    def test_duplicate_native_file_rejected(self):
        self.artifacts["mesh"] = dict(self.artifacts["electric_field"])
        with self.assertRaises(EvidenceError): audit(self.root, self.registry)
    def test_path_escape_rejected(self):
        self.artifacts["mesh"]["path"] = "../outside.dat"
        with self.assertRaises(EvidenceError): audit(self.root, self.registry)

if __name__ == "__main__":
    unittest.main()
