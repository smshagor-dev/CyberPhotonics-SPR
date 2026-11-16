"""Constructed regression data only; never physics evidence."""
import unittest
from research_spectral_resolution_gate import audit_spectrum
D="a"*64
class TestSpectrum(unittest.TestCase):
    def test_three_point_real_pilot_grid_cannot_yield_fwhm(self):
        r=audit_spectrum([.75,.8,.85],[1,3,1],source_kind="solver_native",geometry_sha256=D)
        self.assertFalse(r["fwhm_verified"]);self.assertFalse(r["publication_ready"])
    def test_constructed_narrow_peak_rejected(self):
        r=audit_spectrum([.75,.775,.8,.825,.85],[1,2,8,2,1],source_kind="solver_native",geometry_sha256=D,pml_verified=True,branch_verified=True,spectral_convergence_verified=True)
        self.assertIn("FWHM insufficiently sampled relative to crossing intervals",r["errors"])
    def test_constructed_dense_peak_diagnostic_only(self):
        x=[.7+i*.001 for i in range(201)]
        y=[1+5/(1+((v-.8)/.02)**2) for v in x]
        r=audit_spectrum(x,y,source_kind="solver_native",geometry_sha256=D,pml_verified=True,branch_verified=True,spectral_convergence_verified=True)
        self.assertTrue(r["fwhm_verified"],r);self.assertFalse(r["physics_qualified"]);self.assertFalse(r["publication_ready"])
    def test_source_rejected(self):
        x=[.7+i*.001 for i in range(201)];y=[1+5/(1+((v-.8)/.02)**2) for v in x]
        self.assertIn("source is not solver-native",audit_spectrum(x,y,source_kind="synthetic",geometry_sha256=D)["errors"])
    def test_unbracketed_rejected(self):
        self.assertIn("half-maximum not bracketed on both sides",audit_spectrum([1,2,3,4,5],[1,3,5,4.8,4.7],source_kind="solver_native",geometry_sha256=D)["errors"])
if __name__=="__main__": unittest.main()
