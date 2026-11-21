"""Adversarial software-only checks for native artifact byte verification."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from research.native_artifact_integrity import audit

class NativeArtifactIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = b"real local bytes\\x00"
        (self.root / "native.bin").write_bytes(self.data)
        self.src = {"id": "native", "local_path": "native.bin", "sha256": hashlib.sha256(self.data).hexdigest()}

    def tearDown(self):
        self.tmp.cleanup()

    def run_audit(self, sources=None):
        return audit({"sources": [self.src] if sources is None else sources}, self.root)

    def test_match_not_physics(self):
        result = self.run_audit()
        self.assertTrue(result["all_bytes_verified"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_tampered_bytes(self):
        (self.root / "native.bin").write_bytes(b"tampered")
        self.assertFalse(self.run_audit()["all_bytes_verified"])

    def test_traversal(self):
        self.src["local_path"] = "../outside"
        with self.assertRaises(ValueError): self.run_audit()

    def test_absolute(self):
        self.src["local_path"] = "/etc/passwd"
        with self.assertRaises(ValueError): self.run_audit()

    def test_duplicate(self):
        with self.assertRaises(ValueError): self.run_audit([self.src, self.src])

    def test_missing(self):
        self.src["local_path"] = "missing"
        with self.assertRaises(ValueError): self.run_audit()

    def test_invalid_digest(self):
        self.src["sha256"] = "z" * 64
        with self.assertRaises(ValueError): self.run_audit()

    def test_empty(self):
        self.assertFalse(self.run_audit([])["all_bytes_verified"])

    def test_symlink_escape(self):
        (self.root / "escape").symlink_to(self.root.parent / "not-inside")
        self.src["local_path"] = "escape"
        with self.assertRaises(ValueError): self.run_audit()


    def test_duplicate_path_different_id(self):
        second = dict(self.src, id="alias")
        with self.assertRaises(ValueError): self.run_audit([self.src, second])

    def test_symlink_to_inside_rejected(self):
        (self.root / "alias").symlink_to(self.root / "native.bin")
        self.src["local_path"] = "alias"
        with self.assertRaises(ValueError): self.run_audit()

    def test_intermediate_symlink_rejected(self):
        (self.root / "sub").mkdir()
        (self.root / "sub" / "native.bin").write_bytes(self.data)
        (self.root / "aliasdir").symlink_to(self.root / "sub")
        self.src["local_path"] = "aliasdir/native.bin"
        with self.assertRaises(ValueError): self.run_audit()

    def test_directory_rejected(self):
        (self.root / "dir").mkdir()
        self.src["local_path"] = "dir"
        with self.assertRaises(ValueError): self.run_audit()

if __name__ == "__main__":
    unittest.main()
