"""Software-only adversarial spectrum tests; not Maxwell physics evidence."""
import unittest
from scripts.spectral_identifiability import inspect_spectrum as inspect

class TestSpectralAmbiguity(unittest.TestCase):
    def test_clean_single_peak_is_diagnostic_only(self):
        r = inspect([1,2,3,4,5,6,7], [0,1,3,5,3,1,0])
        self.assertTrue(r["identified"])
        self.assertAlmostEqual(r["width_nm"], 2.5)
        self.assertFalse(r["fwhm_verified"])
        self.assertFalse(r["resonance_verified"])

    def test_secondary_local_maxima_rejected(self):
        r = inspect(list(range(1,12)), [0,1,3,1,2,6,2,1,2,1,0])
        self.assertEqual(r["reason"], "ambiguous_local_maxima_or_plateau")

    def test_plateau_rejected(self):
        self.assertFalse(inspect([1,2,3,4,5,6,7], [0,1,3,5,5,1,0])["identified"])

    def test_nonpositive_wavelength_rejected(self):
        self.assertFalse(inspect([-2,-1,0,1,2,3,4], [0,1,3,5,3,1,0])["identified"])

    def test_three_point_exploratory_grid_rejected(self):
        self.assertFalse(inspect([750,800,850], [1,3,1])["identified"])

    def test_rebounding_flank_rejected(self):
        self.assertFalse(inspect(list(range(1,10)), [0,1,2,5,4,3,3.5,1,0])["identified"])

if __name__ == "__main__":
    unittest.main()
