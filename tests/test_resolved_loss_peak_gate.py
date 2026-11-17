import unittest
import math
from research.resolved_loss_peak_gate import analyze, sensitivity_nm_per_riu

class TestLossPeak(unittest.TestCase):
    def spectrum(self, center=.8):
        x=[.7+i*.001 for i in range(201)]
        return x,[2+10/(1+((v-center)/.02)**2) for v in x]
    def test_resolved(self):
        x,y=self.spectrum()
        p=analyze(x,y,max_step_um=.002)
        self.assertAlmostEqual(p.resonance_um,.8,places=5)
        self.assertAlmostEqual(p.fwhm_um,.039,delta=.003)
    def test_coarse(self):
        x,y=self.spectrum()
        with self.assertRaises(ValueError): analyze(x,y,max_step_um=.0005)
    def test_nonmonotone(self):
        x,y=self.spectrum();x[10]=x[9]
        with self.assertRaises(ValueError): analyze(x,y,max_step_um=.002)
    def test_nonfinite(self):
        x,y=self.spectrum();y[20]=math.nan
        with self.assertRaises(ValueError): analyze(x,y,max_step_um=.002)
    def test_flat(self):
        x,y=self.spectrum()
        with self.assertRaises(ValueError): analyze(x,[2]*len(x),max_step_um=.002)
    def test_unresolved(self):
        x,y=self.spectrum()
        with self.assertRaises(ValueError): analyze(x,y,max_step_um=.002,min_samples_per_width=100)
    def test_sensitivity(self):
        x,y=self.spectrum();a=analyze(x,y,max_step_um=.002)
        x,y=self.spectrum(.81);b=analyze(x,y,max_step_um=.002)
        self.assertAlmostEqual(sensitivity_nm_per_riu(a,b,1.35,1.36),1000,places=5)
    def test_invalid_ri(self):
        x,y=self.spectrum();a=analyze(x,y,max_step_um=.002)
        with self.assertRaises(ValueError): sensitivity_nm_per_riu(a,a,1.35,1.35)
