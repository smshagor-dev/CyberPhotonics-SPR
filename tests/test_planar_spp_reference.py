"""Independent analytic planar-interface checks; no PCF qualification."""
import math
import unittest
from scripts.planar_spp_reference import planar_spp_beta, planar_spp_diagnostics


class PlanarSPPTests(unittest.TestCase):
    def test_lossless_closed_form(self):
        b = planar_spp_beta(2.25, -20, 0.8)
        self.assertAlmostEqual(b.real, 2*math.pi/0.8*math.sqrt(2.25*20/17.75))
        self.assertAlmostEqual(b.imag, 0)

    def test_passive_au_like_residual_and_attenuation(self):
        d = planar_spp_diagnostics(2.25, -20+1.5j, 0.8)
        self.assertTrue(d["decays_both_halfspaces"])
        self.assertLess(d["tm_boundary_relative_residual"], 1e-11)
        self.assertGreater(d["intensity_length_um"], 0)
        self.assertFalse(d["publication_ready"])

    def test_wavelength_scaling_at_fixed_epsilon(self):
        b1 = planar_spp_beta(1, -12+0.8j, 0.8)
        b2 = planar_spp_beta(1, -12+0.8j, 1.6)
        self.assertAlmostEqual(abs(b1/b2), 2)

    def test_pole_rejected(self):
        with self.assertRaisesRegex(ValueError, "bound TM SPP|pole"):
            planar_spp_beta(1, -1, 0.8)

    def test_gain_rejected(self):
        with self.assertRaises(ValueError):
            planar_spp_beta(1, -10-1e-12j, 0.8)

    def test_invalid_wavelength(self):
        for x in (0, -1, float("nan"), float("inf")):
            with self.subTest(x=x), self.assertRaises(ValueError):
                planar_spp_beta(1, -10+1j, x)

    def test_nonmetal_rejected(self):
        for em in (0, 2, 2+0.1j):
            with self.subTest(em=em), self.assertRaisesRegex(ValueError, "negative-real"):
                planar_spp_beta(1, em, 0.8)

    def test_lossless_nonbound_regime_rejected(self):
        for em in (-0.5, -1, -2.25):
            with self.subTest(em=em), self.assertRaisesRegex(ValueError, "bound TM SPP"):
                planar_spp_beta(2.25, em, 0.8)

    def test_lossless_bound_regime_allowed(self):
        d = planar_spp_diagnostics(2.25, -2.2501, 0.8)
        self.assertTrue(d["decays_both_halfspaces"])
        self.assertLess(d["tm_boundary_relative_residual"], 1e-10)


if __name__ == "__main__":
    unittest.main()
