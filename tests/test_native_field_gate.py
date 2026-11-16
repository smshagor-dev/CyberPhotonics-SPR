import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from research_audits.native_field_gate import REQUIRED, audit_native_fields


class TestNativeFieldGate(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        files = {}
        for role in REQUIRED:
            path = self.root / (role + ".bin")
            path.write_bytes(role.encode())
            files[role] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.manifest = {"source_type": "open_source_fem_native", "comsol_model_verified": False, "files": files}
        self.save()

    def save(self):
        (self.root / "native_fields_manifest.json").write_text(json.dumps(self.manifest))

    def test_valid_archive_is_not_physics_evidence(self):
        result = audit_native_fields(self.root)
        self.assertTrue(result["native_archive_integrity_passed"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])

    def test_missing_electric_field(self):
        del self.manifest["files"]["electric_field"]
        self.save()
        self.assertFalse(audit_native_fields(self.root)["native_archive_integrity_passed"])

    def test_tampered_magnetic_field(self):
        (self.root / "magnetic_field.bin").write_bytes(b"tampered")
        self.assertFalse(audit_native_fields(self.root)["native_archive_integrity_passed"])

    def test_unsafe_path(self):
        self.manifest["files"]["mesh"]["path"] = "../mesh.bin"
        self.save()
        self.assertFalse(audit_native_fields(self.root)["native_archive_integrity_passed"])

    def test_comsol_mislabel(self):
        self.manifest["comsol_model_verified"] = True
        self.save()
        self.assertFalse(audit_native_fields(self.root)["native_archive_integrity_passed"])

    def test_missing_manifest(self):
        (self.root / "native_fields_manifest.json").unlink()
        self.assertFalse(audit_native_fields(self.root)["native_archive_integrity_passed"])

    def test_duplicate_native_path(self):
        self.manifest["files"]["magnetic_field"] = dict(self.manifest["files"]["electric_field"])
        self.save()
        self.assertFalse(audit_native_fields(self.root)["native_archive_integrity_passed"])
