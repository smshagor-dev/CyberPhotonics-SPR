"""Analytic Maxwell interface invariants; software-only cross-check."""
import math
import unittest
from scripts.planar_tm_fresnel_reference import passive_kz, tm_reflectance


class PlanarTMTests(unittest.TestCase):
    def test_equal_media(self):
        self.assertAlmostEqual(tm_reflectance(2.25, 2.25, 0.3), 0)

    def test_normal_incidence(self):
        self.assertAlmostEqual(tm_reflectance(1, 2.25, 0), ((1-1.5)/(1+1.5))**2)

    def test_brewster_angle(self):
        self.assertLess(tm_reflectance(1, 2.25, math.atan(1.5)), 1e-25)

    def test_total_internal_reflection(self):
        self.assertAlmostEqual(tm_reflectance(2.25, 1, math.radians(60)), 1)

    def test_absorbing_outgoing_branch(self):
        self.assertGreater(passive_kz(-10+1j, 0, 2*math.pi).imag, 0)
        self.assertLessEqual(tm_reflectance(2.25, -10+1j, 0.5), 1)

    def test_gain_rejected(self):
        with self.assertRaises(ValueError):
            tm_reflectance(1, 2-1j, 0)

    def test_inf_media_rejected(self):
        with self.assertRaises(ValueError):
            tm_reflectance(1, complex(float("inf"), 0), 0)

    def test_zero_epsilon_rejected_explicitly(self):
        with self.assertRaisesRegex(ValueError, "nonzero"):
            tm_reflectance(1, 0, 0.2)

    def test_tiny_gain_rejected(self):
        with self.assertRaisesRegex(ValueError, "gain"):
            passive_kz(2-1e-14j, 0, 2*math.pi)

    def test_evanescent_branch(self):
        kz = passive_kz(1, 2*math.pi*2, 2*math.pi)
        self.assertAlmostEqual(kz.real, 0)
        self.assertGreater(kz.imag, 0)


if __name__ == "__main__":
    unittest.main()
