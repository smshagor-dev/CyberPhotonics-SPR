import math
import unittest
from planar_spp_reference import planar_tm_spp

class PlanarSPPTests(unittest.TestCase):
    def test_lossless(self):
        r=planar_tm_spp(.8,-24+0j,1.77+0j)
        self.assertAlmostEqual(r["neff"].imag,0)
        self.assertLess(r["tm_boundary_residual"],1e-12)
        self.assertEqual(r["loss_db_cm"],0)
        self.assertFalse(r["physics_qualified"])
    def test_passive(self):
        r=planar_tm_spp(.8,-24+1.5j,1.77+0j)
        self.assertGreater(r["loss_db_cm"],0)
        self.assertLess(r["tm_boundary_residual"],1e-12)
    def test_invalid(self):
        for lam,em,ed in [(0,-24+1j,1.77),(.8,1+1j,1.77),(.8,-24-1j,1.77),(.8,-1+0j,1.77),(float("nan"),-24+1j,1.77)]:
            with self.subTest(lam=lam,em=em):
                with self.assertRaises(ValueError): planar_tm_spp(lam,em,ed)
    def test_index(self):
        r=planar_tm_spp(1,-25+0j,1+0j)
        self.assertAlmostEqual(r["neff"].real,math.sqrt(25/24))
