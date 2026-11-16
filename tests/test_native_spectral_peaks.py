"""Constructed spectra test software only; never qualified Maxwell evidence."""
import unittest
from research_validation.native_spectral_peaks import isolated_peak, fixed_geometry_sensitivity

class TestNativeSpectralPeaks(unittest.TestCase):
    def setUp(self):
        self.x=list(range(20))
        self.y=[1+10/(1+((i-10)/2)**2) for i in self.x]
    def test_peak(self):
        p=isolated_peak(self.x,self.y)
        self.assertAlmostEqual(p.resonance_nm,10)
        self.assertGreater(p.fwhm_nm,0)
    def test_multipeak(self):
        y=self.y.copy();y[3]=8
        with self.assertRaises(ValueError):isolated_peak(self.x,y)
    def test_clipped(self):
        with self.assertRaises(ValueError):isolated_peak(self.x[:11],self.y[:11])
    def test_nonfinite(self):
        y=self.y.copy();y[3]=float("nan")
        with self.assertRaises(ValueError):isolated_peak(self.x,y)
    def test_negative(self):
        y=self.y.copy();y[3]=-1
        with self.assertRaises(ValueError):isolated_peak(self.x,y)
    def test_unsorted(self):
        with self.assertRaises(ValueError):isolated_peak(self.x[::-1],self.y)
    def test_geometry(self):
        with self.assertRaises(ValueError):fixed_geometry_sensitivity([1.3,1.4,1.5],[700,710,720],["a","b","a"])
    def test_sensitivity(self):
        self.assertAlmostEqual(fixed_geometry_sensitivity([1.3,1.4,1.5],[700,710,720],["a"]*3)[0],100)
