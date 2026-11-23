import unittest
from scripts.qualified_spectral_metrics import resonance_metrics, finite_difference_sensitivity


class TestSpectra(unittest.TestCase):
    def test_triangular_peak(self):
        d = resonance_metrics([.70,.72,.74,.76,.78,.80,.82], [1,2,4,6,4,2,1])
        self.assertAlmostEqual(d["fwhm_um"], .05)
        self.assertFalse(d["physics_qualified"])
        self.assertFalse(d["publication_ready"])
    def test_short_grid(self):
        with self.assertRaises(ValueError):
            resonance_metrics([.7,.8,.9], [1,4,1])
    def test_nonmonotone(self):
        with self.assertRaises(ValueError):
            resonance_metrics([.7,.72,.74,.74,.78,.8,.82], [1,2,4,6,4,2,1])
    def test_two_peaks(self):
        with self.assertRaises(ValueError):
            resonance_metrics([.7,.72,.74,.76,.78,.8,.82,.84,.86], [1,3,1,1,4,1,1,3,1])
    def test_flat(self):
        with self.assertRaises(ValueError):
            resonance_metrics([.7,.72,.74,.76,.78,.8,.82], [1]*7)
    def test_sensitivity(self):
        self.assertAlmostEqual(finite_difference_sensitivity(1.35,.8,1.36,.82), 2000)
    def test_duplicate_ri(self):
        with self.assertRaises(ValueError):
            finite_difference_sensitivity(1.35,.8,1.35,.82)


if __name__ == "__main__":
    unittest.main()
