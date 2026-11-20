"""Regression: one-shot iterables must not be consumed twice (software-only)."""
import unittest
from scripts.spectral_refinement import compare_spectra

P = {
    "geometry_sha256": "a" * 64,
    "au_dataset_sha256": "b" * 64,
    "analyte_ri": 1.36,
    "mode_branch": "exploratory-mode-0",
    "boundary_condition": "unqualified-finite-domain",
    "solver": "software-fixture",
}

class SpectralOneShotTests(unittest.TestCase):
    def test_generator_matches_list_diagnostic(self):
        cx, cy = [1, 3, 5, 7, 9], [0, 1, 4, 1, 0]
        fx, fy = list(range(1, 10)), [0, .4, 1, 2.5, 4, 2.5, 1, .4, 0]
        kw = dict(coarse_provenance=P, fine_provenance=P)
        expected = compare_spectra(cx, cy, fx, fy, **kw)
        actual = compare_spectra(iter(cx), iter(cy), iter(fx), iter(fy), **kw)
        self.assertEqual(actual, expected)
        self.assertFalse(actual["physics_qualified"])
        self.assertFalse(actual["publication_ready"])

    def test_noniterable_rejected_without_exception(self):
        result = compare_spectra(None, [0, 1], [1, 2], [0, 1],
                                 coarse_provenance=P, fine_provenance=P)
        self.assertEqual(result["reason"], "invalid_samples")
        self.assertFalse(result["physics_qualified"])

if __name__ == "__main__":
    unittest.main()
