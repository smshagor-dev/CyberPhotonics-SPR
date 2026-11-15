"""Only constructed curves: these tests are NOT photonic evidence."""
import unittest
from sprpcf.simulation.spectral_resolution_gate import inspect_spectrum

class SpectralGateTests(unittest.TestCase):
    def setUp(self):
        self.x=[0.7+i*.005 for i in range(41)]
        self.y=[1+5/(1+((v-.8)/.025)**2) for v in self.x]
    def test_software_diagnostic(self):
        r=inspect_spectrum(self.x,self.y)
        self.assertTrue(r["resolved"])
        self.assertFalse(r["publication_qualified"])
        self.assertAlmostEqual(r["resonance_um"],.8)
        self.assertGreater(r["fwhm_um"],0)
    def test_three_points(self):
        self.assertFalse(inspect_spectrum(self.x[:3],self.y[:3])["resolved"])
    def test_edge_peak(self):
        self.assertFalse(inspect_spectrum(self.x,list(range(41)))["resolved"])
    def test_nan(self):
        y=self.y.copy();y[4]=float("nan")
        self.assertFalse(inspect_spectrum(self.x,y)["resolved"])
    def test_duplicate_wavelength(self):
        x=self.x.copy();x[4]=x[3]
        self.assertFalse(inspect_spectrum(x,self.y)["resolved"])
    def test_multiple_peaks(self):
        y=self.y.copy();y[6]=100
        self.assertFalse(inspect_spectrum(self.x,y)["resolved"])
    def test_flat_curve(self):
        self.assertFalse(inspect_spectrum(self.x,[1.0]*41)["resolved"])
if __name__=="__main__":
    unittest.main()
