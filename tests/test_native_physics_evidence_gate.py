"""Software-only adversarial checks; no physical eigenmode evidence."""
import unittest
from scripts.native_physics_evidence_gate import audit

H = "a" * 64
BASE = dict(source_kind="native_open_source_maxwell", solver="femwell",
            solver_version="recorded-version", material_source="measured-Au-source",
            boundary_condition="verified_pml", commit_sha="b" * 40,
            geometry_sha256=H, mesh_sha256=H, material_sha256=H,
            field_archive_sha256=H, pml_converged=True, mesh_converged=True,
            domain_converged=True, spectral_converged=True,
            mode_tracking_verified=True, fixed_geometry_ri_verified=True,
            independent_solver_verified=True, sample_count=9, failure_count=0)

class TestNativeEvidenceGate(unittest.TestCase):
    def test_complete_record_is_only_review_candidate(self):
        result = audit(BASE)
        self.assertTrue(result["qualified_for_review"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_fixture_rejected(self):
        self.assertIn("unsupported_source_kind", audit({**BASE, "source_kind": "fixture"})["reasons"])

    def test_missing_pml_rejected(self):
        self.assertIn("unverified:pml_converged", audit({**BASE, "pml_converged": False})["reasons"])

    def test_missing_mode_identity_rejected(self):
        self.assertIn("unverified:mode_tracking_verified", audit({**BASE, "mode_tracking_verified": False})["reasons"])

    def test_boolean_failure_count_rejected(self):
        self.assertIn("invalid_count:failure_count", audit({**BASE, "failure_count": False})["reasons"])

    def test_failed_solves_rejected(self):
        self.assertIn("unresolved_failures", audit({**BASE, "failure_count": 2})["reasons"])

    def test_comsol_claim_rejected(self):
        result = audit({**BASE, "comsol_verified": True})
        self.assertIn("unsupported_comsol_claim", result["reasons"])
        self.assertFalse(result["comsol_verified"])

    def test_bad_hash_rejected(self):
        self.assertIn("invalid_hash:mesh_sha256", audit({**BASE, "mesh_sha256": "x"})["reasons"])

    def test_empty_record_rejected(self):
        self.assertFalse(audit({})["qualified_for_review"])

if __name__ == "__main__":
    unittest.main()
