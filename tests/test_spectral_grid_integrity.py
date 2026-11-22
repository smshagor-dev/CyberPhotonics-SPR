"""Adversarial software-only spectra; never physical PCF-SPR evidence."""
import unittest
from sprpcf.simulation.spectral_eligibility import screen_resonance_sampling


class SpectralGridIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.x = [0.70 + 0.005 * i for i in range(41)]
        self.y = [1.0 + 10.0 * max(0, 1 - abs(i - 20) / 8) for i in range(41)]

    def test_uniform_reference_only_screened(self):
        result = screen_resonance_sampling(self.x, self.y)
        self.assertTrue(result["eligible_for_fwhm_estimation"], result["failures"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["fwhm_verified"])

    def test_missing_wavelength_interval_rejected(self):
        # Large wavelength hole outside the resonance must still fail closed.
        x = [v + (0.10 if i >= 32 else 0) for i, v in enumerate(self.x)]
        result = screen_resonance_sampling(x, self.y)
        self.assertFalse(result["eligible_for_fwhm_estimation"])
        self.assertIn("unresolved wavelength sampling gap", result["failures"])

    def test_boolean_samples_rejected(self):
        for xs, ys in (([True] + self.x[1:], self.y),
                       (self.x, self.y[:20] + [False] + self.y[21:])):
            result = screen_resonance_sampling(xs, ys)
            self.assertFalse(result["eligible_for_fwhm_estimation"])
            self.assertFalse(result["publication_ready"])

    def test_invalid_gap_threshold_rejected(self):
        for threshold in (True, float("nan"), 0.5, float("inf")):
            with self.assertRaises(ValueError):
                screen_resonance_sampling(self.x, self.y, max_gap_ratio=threshold)


if __name__ == "__main__":
    unittest.main()
