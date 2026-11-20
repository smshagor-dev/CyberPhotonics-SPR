import math
import unittest
from scripts.planar_spp_control import planar_spp


class TestPlanarSPP(unittest.TestCase):
    def test_lossless_closed_form(self):
        result = planar_spp(0.8, -20, 0, 1.3)
        expected = math.sqrt(-20 * 1.3**2 / (-20 + 1.3**2))
        self.assertAlmostEqual(result["neff_real"], expected, places=12)
        self.assertEqual(result["neff_imag"], 0)
        self.assertFalse(result["pcf_physics_qualified"])

    def test_passive_attenuation_and_wavelength_scaling(self):
        a = planar_spp(0.8, -20, 1.2, 1.36)
        b = planar_spp(1.6, -20, 1.2, 1.36)
        self.assertGreater(a["neff_imag"], 0)
        self.assertAlmostEqual(a["attenuation_db_per_cm"],
                               2 * b["attenuation_db_per_cm"])

    def test_reject_invalid_inputs(self):
        for args in ((True, -20, 1, 1.36), (0, -20, 1, 1.36),
                     (.8, -20, -1, 1.36), (.8, -20, 1, float("nan")),
                     (.8, -1.36**2, 0, 1.36)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                planar_spp(*args)
