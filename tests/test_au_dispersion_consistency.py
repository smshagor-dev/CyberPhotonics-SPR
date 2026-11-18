"""Constructed fixture only: never interpret as measured gold."""
import unittest
from scripts.au_dispersion_consistency import interpolation_consistency,audit_interpolation

TABLE={"points":((700.,.15,4.),(800.,.16,4.5),(900.,.17,5.)),"sha256":"fixture-only"}

class TestAuInterpolationConsistency(unittest.TestCase):
    def test_nodes(self):
        for w in (700,800,900):
            self.assertAlmostEqual(interpolation_consistency(TABLE,w)["relative_discrepancy"],0,places=13)
    def test_midpoint(self):
        r=interpolation_consistency(TABLE,750)
        z0,z1=complex(.15,4)**2,complex(.16,4.5)**2
        self.assertAlmostEqual(abs(r["epsilon_linear"]-(z0+z1)/2),0)
        self.assertGreater(r["relative_discrepancy"],0)
    def test_extrapolation(self):
        for w in (699,901,float("nan"),float("inf")):
            with self.assertRaises(ValueError): interpolation_consistency(TABLE,w)
    def test_audit(self):
        a=audit_interpolation(TABLE,3)
        self.assertEqual(len(a["samples"]),6)
        self.assertGreater(a["max_relative_discrepancy"],0)
        self.assertFalse(a["physics_qualified"])
        self.assertFalse(a["source_verified"])
    def test_invalid_samples(self):
        for n in (0,101,1.5,True):
            with self.assertRaises(ValueError): audit_interpolation(TABLE,n)

if __name__=="__main__":
    unittest.main()
