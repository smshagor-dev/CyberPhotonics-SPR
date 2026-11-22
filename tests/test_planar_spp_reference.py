"""Software regression tests use illustrative epsilon, NOT measured Au or PCF evidence."""
import math
import unittest
from sprpcf.simulation.planar_spp_reference import planar_spp_reference


class PlanarSPPReferenceTests(unittest.TestCase):
    def test_passive_branch_and_units(self):
        x = planar_spp_reference(.8, -24+1.5j, 2.1)
        self.assertGreater(x["n_eff_real"], math.sqrt(2.1))
        self.assertGreater(x["n_eff_imag"], 0)
        self.assertAlmostEqual(x["beta_real_per_um"], 2*math.pi/.8*x["n_eff_real"])
        self.assertAlmostEqual(x["power_attenuation_db_per_cm"],
                               20/math.log(10)*x["beta_imag_per_um"]*1e4)
        self.assertFalse(x["publication_qualified"])
        self.assertFalse(x["pcf_mode_verified"])

    def test_lossless_limit(self):
        self.assertLess(planar_spp_reference(.8, -24+1e-8j, 2.1)["n_eff_imag"], 1e-8)

    def test_reject_nonpassive(self):
        for eps in (-24-1j, -24+0j, -1+1j, complex(float("nan"), 1),
                    complex(-24, float("inf"))):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                planar_spp_reference(.8, eps, 2.1)

    def test_reject_invalid_inputs(self):
        for wl, ed in ((0,2.1),(-.8,2.1),(float("nan"),2.1),
                       (.8,0),(.8,-2),(.8,float("inf"))):
            with self.subTest(wl=wl, ed=ed), self.assertRaises(ValueError):
                planar_spp_reference(wl, -24+1j, ed)

    def test_dispersion_residual(self):
        em, ed = -24+1.5j, 2.1
        x = planar_spp_reference(.8, em, ed)
        neff = complex(x["n_eff_real"], x["n_eff_imag"])
        self.assertLess(abs(neff**2*(em+ed)-em*ed), 1e-12)
