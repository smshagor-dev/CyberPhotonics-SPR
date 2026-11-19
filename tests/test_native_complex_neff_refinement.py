import unittest

from scripts.native_complex_neff_refinement import ConvergenceError, audit_refinement


class ComplexRefinementTests(unittest.TestCase):
    def setUp(self):
        self.runs = []
        for i, (h, im) in enumerate(((0.03, 0.0034), (0.025, 0.0033), (0.02, 0.00325))):
            self.runs.append({
                "resolution_um": h, "neff_real": 1.382 + 0.0001 / (i + 1),
                "neff_imag": im, "artifact_sha256": f"{i+1:064x}",
                "geometry_sha256": "geometry", "material_source_sha256": "material",
                "solver_config_sha256": "config", "mode_track_id": "candidate-A",
            })

    def audit(self):
        return audit_refinement(self.runs, wavelength_um=0.8)

    def test_constructed_diagnostic_not_physics_evidence(self):
        result = self.audit()
        self.assertEqual(len(result["loss_db_per_cm"]), 3)
        self.assertTrue(all(x > 0 for x in result["loss_db_per_cm"]))
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["mesh_convergence_verified"])
        self.assertFalse(result["mode_identity_verified"])

    def test_loss_conversion(self):
        import math
        result = self.audit()
        expected = 20 / math.log(10) * (2 * math.pi / 0.8) * 0.0034 * 1e4
        self.assertAlmostEqual(result["loss_db_per_cm"][0], expected)

    def test_reject_changed_material(self):
        self.runs[1]["material_source_sha256"] = "other"
        with self.assertRaises(ConvergenceError):
            self.audit()

    def test_reject_duplicate_artifact(self):
        self.runs[1]["artifact_sha256"] = self.runs[0]["artifact_sha256"]
        with self.assertRaises(ConvergenceError):
            self.audit()

    def test_reject_nonrefined_mesh(self):
        self.runs[2]["resolution_um"] = 0.03
        with self.assertRaises(ConvergenceError):
            self.audit()

    def test_reject_gain_or_nan(self):
        self.runs[0]["neff_imag"] = -0.01
        with self.assertRaises(ConvergenceError):
            self.audit()

    def test_reject_mode_swap_metadata(self):
        self.runs[2]["mode_track_id"] = "candidate-B"
        with self.assertRaises(ConvergenceError):
            self.audit()


if __name__ == "__main__":
    unittest.main()
