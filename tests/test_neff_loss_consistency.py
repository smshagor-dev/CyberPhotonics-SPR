"""Constructed arithmetic regressions; NOT Maxwell solver evidence."""
import math
import unittest
from research.neff_loss_consistency import audit_neff_loss

class TestLossAudit(unittest.TestCase):
    def test_exact_units(self):
        expected = (20 / math.log(10)) * (2 * math.pi / (0.8e-4)) * 1e-6
        a = audit_neff_loss(.8, 1.44, 1e-6, expected)
        self.assertTrue(a.consistent)
        self.assertAlmostEqual(a.computed_db_cm, expected)
        self.assertFalse(a.physics_qualified)
        self.assertFalse(a.publication_ready)
        self.assertFalse(a.comsol_verified)
    def test_zero(self):
        self.assertTrue(audit_neff_loss(.8, 1.44, 0, 0).consistent)
    def test_wrong_factor(self):
        self.assertFalse(audit_neff_loss(.8, 1.44, 1e-6, 6.82).consistent)
    def test_wrong_sign(self):
        self.assertFalse(audit_neff_loss(.8, 1.44, -1e-6, 0).consistent)
    def test_nan(self):
        self.assertFalse(audit_neff_loss(float("nan"), 1.44, 0, 0).consistent)
    def test_inf(self):
        self.assertFalse(audit_neff_loss(.8, 1.44, float("inf"), 0).consistent)
    def test_negative_loss(self):
        self.assertFalse(audit_neff_loss(.8, 1.44, 1e-6, -1).consistent)
    def test_negative_tolerance(self):
        self.assertFalse(audit_neff_loss(.8, 1.44, 0, 0, relative_tolerance=-1).consistent)
    def test_boolean_rejected(self):
        self.assertFalse(audit_neff_loss(True, 1.44, 0, 0).consistent)
    def test_overflow(self):
        self.assertFalse(audit_neff_loss(1e-300, 1.44, 1e300, 1).consistent)
    def test_zero_wavelength(self):
        self.assertFalse(audit_neff_loss(0, 1.44, 0, 0).consistent)

if __name__ == "__main__":
    unittest.main()
