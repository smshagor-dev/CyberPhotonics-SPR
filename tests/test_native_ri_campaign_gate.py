import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.native_ri_campaign_gate import EvidenceError, audit_ri_campaign


class NativeRICampaignGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = {
            "source_kind": "native_open_source_full_vector_fem",
            "solver": "femwell",
            "comsol_verified": False,
            "publication_ready": False,
            "geometry_sha256": "geometry",
            "mesh_sha256": "mesh",
            "material_source_sha256": "material",
            "solver_config_sha256": "solver-config",
            "commit_sha": "commit",
            "spectra": [],
        }
        for i, ri in enumerate((1.35, 1.36, 1.37)):
            payload = json.dumps({
                "analyte_ri": ri,
                "wavelength_nm": [700, 710, 720, 730, 740, 750, 760],
                "loss_db_per_cm": [1, 2, 3, 4, 3, 2, 1],
            }).encode()
            name = f"spectrum_{i}.json"
            (self.root / name).write_bytes(payload)
            self.manifest["spectra"].append({
                "analyte_ri": ri, "path": name, "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "source_kind": self.manifest["source_kind"],
                **{k: self.manifest[k] for k in (
                    "geometry_sha256", "mesh_sha256", "material_source_sha256",
                    "solver_config_sha256")},
            })

    def test_constructed_fixture_only_checks_software_gate(self):
        result = audit_ri_campaign(self.manifest, self.root)
        self.assertEqual(result["native_artifact_count"], 3)
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])

    def test_reject_changed_bytes(self):
        (self.root / "spectrum_0.json").write_text("{}")
        with self.assertRaises(EvidenceError):
            audit_ri_campaign(self.manifest, self.root)

    def test_reject_geometry_change(self):
        self.manifest["spectra"][1]["geometry_sha256"] = "different"
        with self.assertRaises(EvidenceError):
            audit_ri_campaign(self.manifest, self.root)

    def test_reject_mixed_source(self):
        self.manifest["spectra"][0]["source_kind"] = "synthetic"
        with self.assertRaises(EvidenceError):
            audit_ri_campaign(self.manifest, self.root)

    def test_reject_symlink(self):
        (self.root / "spectrum_0.json").unlink()
        (self.root / "spectrum_0.json").symlink_to("spectrum_1.json")
        with self.assertRaises(EvidenceError):
            audit_ri_campaign(self.manifest, self.root)

    def test_reject_publication_flag(self):
        self.manifest["publication_ready"] = True
        with self.assertRaises(EvidenceError):
            audit_ri_campaign(self.manifest, self.root)

    def test_reject_duplicate_ri(self):
        self.manifest["spectra"][1]["analyte_ri"] = 1.35
        with self.assertRaises(EvidenceError):
            audit_ri_campaign(self.manifest, self.root)


if __name__ == "__main__":
    unittest.main()
