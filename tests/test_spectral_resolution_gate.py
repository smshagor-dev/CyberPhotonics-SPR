"""Software-only synthetic regression cases; never physical evidence."""
import math
import unittest
from research.spectral_resolution_gate import assess_spectrum

class TestSpectralResolution(unittest.TestCase):
    def test_three_sample_real_fem_grid_is_insufficient(self):
        r=assess_spectrum([.75,.80,.85],[1,5,1])
        self.assertFalse(r.resolved)
        self.assertIn("sparse_spectrum",r.reasons)
        self.assertFalse(r.physics_qualified)
        self.assertFalse(r.publication_ready)
        self.assertFalse(r.comsol_verified)

    def test_dense_constructed_gaussian_is_software_only(self):
        x=[.7+i*.001 for i in range(201)]
        y=[2+8*math.exp(-.5*((z-.8)/.025)**2) for z in x]
        r=assess_spectrum(x,y)
        self.assertTrue(r.resolved)
        self.assertAlmostEqual(r.fwhm_um,2.35482*.025,delta=.0002)
        self.assertFalse(r.physics_qualified)

    def test_monotone(self):
        self.assertFalse(assess_spectrum([.7+i*.001 for i in range(30)],list(range(30))).resolved)

    def test_nan(self):
        self.assertFalse(assess_spectrum([.7,.8,.9],[1,float("nan"),2]).resolved)

    def test_duplicate_wavelength(self):
        self.assertFalse(assess_spectrum([.7,.7,.9],[1,2,1]).resolved)

    def test_unbracketed_peak(self):
        x=[.7+i*.001 for i in range(101)]
        y=[1+4*i/50 if i<=50 else 5-.01*(i-50) for i in range(101)]
        self.assertFalse(assess_spectrum(x,y).resolved)

    def test_multiple_peaks(self):
        x=[.7+i*.001 for i in range(201)]
        y=[2+math.exp(-((z-.75)/.005)**2)+math.exp(-((z-.85)/.005)**2) for z in x]
        self.assertFalse(assess_spectrum(x,y).resolved)

if __name__=="__main__":
    unittest.main()
