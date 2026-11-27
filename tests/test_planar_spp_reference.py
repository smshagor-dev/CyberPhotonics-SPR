import math
import unittest
from scripts.planar_spp_reference import planar_spp

class TestPlanarSPP(unittest.TestCase):
    def test_bound_tm_and_boundary(self):
        r = planar_spp(0.8, complex(-24, 1.5), 1.77)
        self.assertLess(r["tm_boundary_relative_residual"], 1e-9)
        self.assertGreater(r["neff_imag"], 0)
        self.assertFalse(r["physics_qualified"])
    def test_loss_conversion(self):
        r = planar_spp(0.8, complex(-24, 1.5), 1.77)
        self.assertAlmostEqual(r["loss_db_per_cm"], 20/math.log(10)*r["beta_imag_per_um"]*1e4)
    def test_reject_gain(self):
        with self.assertRaises(ValueError): planar_spp(0.8, complex(-24,-1),1.77)
    def test_reject_nonfinite(self):
        with self.assertRaises(ValueError): planar_spp(float("nan"),complex(-24,1),1.77)
    def test_reject_nonbound(self):
        with self.assertRaises(ValueError): planar_spp(0.8,complex(2,1),1.77)

if __name__ == "__main__": unittest.main()
