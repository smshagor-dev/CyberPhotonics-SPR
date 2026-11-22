import cmath
import unittest
from sprpcf.simulation.planar_boundary_audit import audit_tm_interface


class TestPlanarBoundaryAudit(unittest.TestCase):
    def test_exact_dispersion(self):
        m, d = -12 + 1.2j, 2.1
        n = cmath.sqrt(m * d / (m + d))
        result = audit_tm_interface(.8, m, d, n)
        self.assertTrue(result["boundary_consistent"])
        self.assertTrue(result["decaying_half_spaces"])
        self.assertFalse(result["publication_qualified"])

    def test_perturbed_mode_rejected(self):
        m, d = -12 + 1.2j, 2.1
        n = cmath.sqrt(m * d / (m + d))
        self.assertFalse(audit_tm_interface(.8, m, d, n + .02)["boundary_consistent"])

    def test_active_metal_rejected(self):
        with self.assertRaises(ValueError):
            audit_tm_interface(.8, -12 - 1j, 2, 1.5 + .01j)

    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError):
            audit_tm_interface(float("nan"), -12 + 1j, 2, 1.5 + .01j)

    def test_invalid_tolerance(self):
        with self.assertRaises(ValueError):
            audit_tm_interface(.8, -12 + 1j, 2, 1.5 + .01j, 0)
