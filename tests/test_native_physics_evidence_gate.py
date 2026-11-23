"""Software-only adversarial checks; never physics evidence."""
import unittest
from scripts.native_physics_evidence_gate import audit


def digest(n):
    return format(n, "064x")


BASE = dict(source_kind="native_open_source_maxwell", solver="femwell",
            solver_version="0.1.9", material_source="Johnson-Christy-1972",
            boundary_condition="verified_pml",
            commit_sha="1234567890abcdef" * 2 + "12345678",
            geometry_sha256=digest(11), mesh_sha256=digest(12),
            material_sha256=digest(13), field_archive_sha256=digest(14),
            pml_converged=True, mesh_converged=True, domain_converged=True,
            spectral_converged=True, mode_tracking_verified=True,
            fixed_geometry_ri_verified=True, independent_solver_verified=True,
            sample_count=9, failure_count=0,
            artifact_url="https://github.com/example/repo/actions/runs/1",
            artifact_sha256=digest(15),
            verification_manifest_sha256=digest(16))


class TestNativeEvidenceGate(unittest.TestCase):
    def test_candidate_is_not_publication(self):
        result = audit(BASE)
        self.assertTrue(result["qualified_for_review"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_missing_artifact(self):
        self.assertIn("missing_native_artifact_url", audit({**BASE, "artifact_url": None})["reasons"])

    def test_placeholder_identity(self):
        self.assertIn("missing_or_placeholder_identity:solver_version",
                      audit({**BASE, "solver_version": "recorded-version"})["reasons"])

    def test_repeated_hash(self):
        self.assertIn("reused_artifact_hash",
                      audit({**BASE, "mesh_sha256": BASE["geometry_sha256"]})["reasons"])

    def test_placeholder_hash(self):
        self.assertIn("invalid_or_placeholder_hash:mesh_sha256",
                      audit({**BASE, "mesh_sha256": "a" * 64})["reasons"])

    def test_false_pml(self):
        self.assertIn("unverified:pml_converged", audit({**BASE, "pml_converged": False})["reasons"])

    def test_fixture(self):
        self.assertIn("unsupported_source_kind", audit({**BASE, "source_kind": "fixture"})["reasons"])

    def test_boolean_count(self):
        self.assertIn("invalid_count:failure_count", audit({**BASE, "failure_count": False})["reasons"])

    def test_unresolved_failure(self):
        self.assertIn("unresolved_failures", audit({**BASE, "failure_count": 1})["reasons"])

    def test_comsol(self):
        self.assertIn("unsupported_comsol_claim", audit({**BASE, "comsol_verified": True})["reasons"])

    def test_invalid_manifest(self):
        self.assertIn("invalid_verification_digest:verification_manifest_sha256",
                      audit({**BASE, "verification_manifest_sha256": "0" * 64})["reasons"])


if __name__ == "__main__":
    unittest.main()
