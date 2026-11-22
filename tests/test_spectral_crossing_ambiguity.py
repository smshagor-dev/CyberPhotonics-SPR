"""Software-only adversarial spectral fixtures; never Maxwell evidence."""
import unittest
from sprpcf.simulation.spectral_eligibility import screen_resonance_sampling


class CrossingAmbiguityTests(unittest.TestCase):
    def test_multiple_right_crossings_rejected(self):
        xs = [0.70 + 0.005*i for i in range(41)]
        ys = [1.0]*41
        for i in range(14, 23):
            ys[i] = 10.0
        ys[25:28] = [9.0, 9.0, 9.0]
        result = screen_resonance_sampling(xs, ys)
        self.assertFalse(result["eligible_for_fwhm_estimation"])
        self.assertIn("multiple half-maximum crossings; ambiguous resonance", result["failures"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])

    def test_single_crossing_screen_only(self):
        xs = [0.70 + 0.005*i for i in range(41)]
        ys = [1.0 + 10.0*max(0, 1-abs(i-20)/8) for i in range(41)]
        result = screen_resonance_sampling(xs, ys)
        self.assertTrue(result["eligible_for_fwhm_estimation"], result["failures"])
        self.assertFalse(result["fwhm_verified"])
        self.assertFalse(result["publication_ready"])


if __name__ == "__main__":
    unittest.main()
