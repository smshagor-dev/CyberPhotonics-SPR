"""Adversarial spectral input checks. Software-only, never physics evidence."""
import unittest
from scripts.spectral_identifiability import inspect_spectrum

class SpectralInputIntegrityTests(unittest.TestCase):
    def test_boolean_wavelength_rejected(self):
        r = inspect_spectrum([1, 2, True, 4, 5], [0, 1, 3, 1, 0])
        self.assertEqual(r["reason"], "boolean_sample_rejected")
        self.assertFalse(r["identified"])

    def test_boolean_attenuation_rejected(self):
        r = inspect_spectrum([1, 2, 3, 4, 5], [0, 1, True, 1, 0])
        self.assertEqual(r["reason"], "boolean_sample_rejected")
        self.assertFalse(r["resonance_verified"])

    def test_missing_iterable_rejected(self):
        self.assertEqual(inspect_spectrum(None, [1, 2])["reason"], "invalid_sample_container")

    def test_generator_supported(self):
        r = inspect_spectrum((v for v in [1,2,3,4,5,6,7]),
                             (v for v in [0,1,2,3,2,1,0]))
        self.assertTrue(r["identified"])
        self.assertFalse(r["fwhm_verified"])

if __name__ == "__main__":
    unittest.main()
