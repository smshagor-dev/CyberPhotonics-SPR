"""Software-only tests: matching bytes never establish Maxwell physics."""
import hashlib
from pathlib import Path
import tempfile
import unittest
from scripts.native_artifact_integrity import verify_native_bytes

class NativeIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.p = Path(self.tmp.name) / "native.bin"
        self.p.write_bytes(b"actual solver bytes\\x00\\xff")
        self.digest = hashlib.sha256(self.p.read_bytes()).hexdigest()

    def test_valid_bytes_not_physics(self):
        result = verify_native_bytes(self.p, self.digest)
        self.assertTrue(result["integrity_verified"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_wrong_digest(self):
        self.assertEqual(verify_native_bytes(self.p, "a"*64)["reason"], "digest_mismatch")

    def test_invalid_digest(self):
        self.assertEqual(verify_native_bytes(self.p, "not-a-hash")["reason"], "invalid_expected_digest")

    def test_missing_file(self):
        self.assertEqual(verify_native_bytes(self.p.with_name("missing"), self.digest)["reason"], "not_regular_file")

    def test_symlink(self):
        link = self.p.with_name("link")
        link.symlink_to(self.p)
        self.assertEqual(verify_native_bytes(link, self.digest)["reason"], "not_regular_file")

    def test_size_limit(self):
        self.assertEqual(verify_native_bytes(self.p, self.digest, max_bytes=3)["reason"], "size_limit_exceeded")

    def test_boolean_limit(self):
        self.assertEqual(verify_native_bytes(self.p, self.digest, max_bytes=True)["reason"], "invalid_size_limit")

    def test_empty_file(self):
        self.p.write_bytes(b"")
        self.assertTrue(verify_native_bytes(self.p, hashlib.sha256(b"").hexdigest())["integrity_verified"])

if __name__ == "__main__":
    unittest.main()
