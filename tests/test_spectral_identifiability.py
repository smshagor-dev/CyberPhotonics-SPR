import unittest
from scripts.spectral_identifiability import inspect_spectrum

class TestSpectrum(unittest.TestCase):
    def test_bracketed(self):
        r=inspect_spectrum([1,2,3,4,5,6,7],[0,1,3,5,3,1,0])
        self.assertTrue(r["identified"])
        self.assertAlmostEqual(r["width_nm"],2.5)
        self.assertFalse(r["fwhm_verified"])
    def test_three_samples(self):
        self.assertFalse(inspect_spectrum([750,800,850],[1,3,1])["identified"])
    def test_edge_peak(self):
        self.assertFalse(inspect_spectrum([1,2,3,4,5,6],[2,5,4,3,2,1])["identified"])
    def test_plateau(self):
        self.assertFalse(inspect_spectrum([1,2,3,4,5,6],[0,1,3,3,1,0])["identified"])
    def test_nonfinite(self):
        self.assertFalse(inspect_spectrum([1,2,3,4,5],[0,1,float("nan"),1,0])["identified"])
    def test_negative(self):
        self.assertFalse(inspect_spectrum([1,2,3,4,5],[0,1,-1,1,0])["identified"])
    def test_nonmonotonic(self):
        self.assertFalse(inspect_spectrum([1,2,2,4,5],[0,1,3,1,0])["identified"])
