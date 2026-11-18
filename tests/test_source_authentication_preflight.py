import unittest
from scripts.source_authentication_preflight import audit

class PreflightTests(unittest.TestCase):
    def test_missing_provenance(self):
        result = audit({})
        self.assertFalse(result["physics_qualified"])
        self.assertIn("insufficient_ri_spectra", result["reasons"])

    def test_self_attestation_is_not_independent_verification(self):
        manifest = dict(geometry_source="paper", geometry_sha256="a"*64,
            au_source="measured", au_data_sha256="b"*64, solver="femwell",
            solver_version="1", mesh_sha256="c"*64, pml_spec="documented",
            domain_convergence=True, mesh_convergence=True, pml_convergence=True,
            mode_overlap_verified=True, independent_solver_comparison=True)
        manifest["spectra"] = [
            dict(analyte_ri=ri, geometry_sha256="a"*64, mesh_sha256="c"*64,
                 samples=[dict(wavelength_um=0.7+j*0.01, loss_db_cm=1.0) for j in range(7)])
            for ri in (1.33, 1.34, 1.35)]
        result = audit(manifest)
        self.assertEqual(result["reasons"], ["independent_external_authentication_required"])
        self.assertFalse(result["publication_ready"])

    def test_nonfinite_manifest_fails_closed_without_crash(self):
        result = audit(dict(spectra=[], unexpected=float("nan")))
        self.assertFalse(result["physics_qualified"])
        self.assertIsNone(result["manifest_sha256"])
        self.assertIn("manifest_not_canonical_json", result["reasons"])

    def test_non_mapping_manifest_fails_closed(self):
        result = audit(["not", "a", "manifest"])
        self.assertEqual(result["reasons"], ["invalid_manifest_type"])

    def test_boolean_wavelength_and_loss_are_rejected(self):
        spectra = [dict(analyte_ri=1.33 + i * 0.01, samples=[
            dict(wavelength_um=True, loss_db_cm=False) for _ in range(7)
        ]) for i in range(3)]
        result = audit(dict(spectra=spectra))
        self.assertIn("invalid_point:0:0", result["reasons"])
        self.assertFalse(result["publication_ready"])

    def test_nonpositive_first_wavelength_rejected(self):
        spectra = [dict(analyte_ri=1.33 + i * 0.01, samples=[
            dict(wavelength_um=-0.1 if j == 0 else 0.7 + j * 0.01, loss_db_cm=1.0)
            for j in range(7)
        ]) for i in range(3)]
        self.assertIn("invalid_point:0:0", audit(dict(spectra=spectra))["reasons"])

    def test_duplicate_ri_and_short_spectrum(self):
        result = audit(dict(spectra=[dict(analyte_ri=1.33, samples=[])]*3))
        self.assertIn("invalid_ri:1", result["reasons"])
        self.assertIn("underresolved_spectrum:0", result["reasons"])

if __name__ == "__main__":
    unittest.main()
