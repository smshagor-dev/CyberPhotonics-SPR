"""Synthetic shapes are SOFTWARE TEST FIXTURES, never PCF evidence."""
import math
import unittest
from sprpcf.simulation.spectral_eligibility import screen_resonance_sampling as screen


class SpectralEligibilityTests(unittest.TestCase):
    def test_three_point_real_pilot_cannot_establish_fwhm(self):
        result = screen([.75, .80, .85], [1, 9, 1])
        self.assertFalse(result["eligible_for_fwhm_estimation"])
        self.assertFalse(result["publication_ready"])

    def test_synthetic_gaussian_only_passes_sampling_screen(self):
        xs = [.7 + .005*i for i in range(41)]
        ys = [2 + 9*math.exp(-((x-.8)/.025)**2) for x in xs]
        result = screen(xs, ys)
        self.assertTrue(result["eligible_for_fwhm_estimation"], result["failures"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["fwhm_verified"])

    def test_monotone_and_plateau_rejected(self):
        xs = [.7 + .005*i for i in range(41)]
        self.assertFalse(screen(xs, list(range(41)))["eligible_for_fwhm_estimation"])
        ys = [1.]*41
        ys[19:22] = [10., 10., 10.]
        self.assertFalse(screen(xs, ys)["eligible_for_fwhm_estimation"])

    def test_nonfinite_negative_and_unordered_rejected(self):
        xs = [.7 + .005*i for i in range(41)]
        for bad in (float("nan"), float("inf"), -1.):
            ys = [2.]*41
            ys[20] = bad
            self.assertFalse(screen(xs, ys)["eligible_for_fwhm_estimation"])
        self.assertFalse(screen(xs[::-1], [2.]*41)["eligible_for_fwhm_estimation"])

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            screen([.7, .8], [1, 2], min_samples=3)

    def test_unbracketed_half_max(self):
        xs = [.7 + .005*i for i in range(41)]
        ys = [8.]*41
        ys[20] = 10.
        self.assertFalse(screen(xs, ys)["eligible_for_fwhm_estimation"])


if __name__ == "__main__":
    unittest.main()
