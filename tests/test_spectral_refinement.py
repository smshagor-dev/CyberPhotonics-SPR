"""Software-only regression inputs; not FEM or COMSOL evidence."""
import unittest
from scripts.spectral_refinement import compare_spectra

P = dict(geometry_sha256="a"*64, au_dataset_sha256="b"*64,
         analyte_ri=1.36, mode_branch="core0",
         boundary_condition="pml", solver="femwell")
C = [1,2,3,4,5,6,7]
F = [1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7]

def loss(x):
    return [max(0, 5-abs(v-4)*2) for v in x]

class SpectralRefinementTests(unittest.TestCase):
    def check(self, **kw):
        return compare_spectra(C,loss(C),F,loss(F),
                               coarse_provenance=P,fine_provenance=P,**kw)
    def test_stable_diagnostic_never_qualified(self):
        r=self.check()
        self.assertTrue(r["stable"])
        self.assertFalse(r["physics_qualified"])
        self.assertFalse(r["fwhm_verified"])
    def test_changed_au_rejected(self):
        q=dict(P,au_dataset_sha256="c"*64)
        r=compare_spectra(C,loss(C),F,loss(F),
                          coarse_provenance=P,fine_provenance=q)
        self.assertEqual(r["reason"],"incompatible_or_missing_provenance")
    def test_non_nested_rejected(self):
        x=[1.1,2.1,3.1,4.1,5.1,6.1,7.1]
        r=compare_spectra(C,loss(C),x,loss(x),
                          coarse_provenance=P,fine_provenance=P)
        self.assertEqual(r["reason"],"not_nested_refinement")
    def test_underresolved_rejected(self):
        r=compare_spectra([1,2,3],[0,1,0],F,loss(F),
                          coarse_provenance=P,fine_provenance=P)
        self.assertEqual(r["reason"],"unidentified_spectrum")
    def test_invalid_tolerance(self):
        self.assertEqual(self.check(peak_tolerance_nm=float("nan"))["reason"],
                         "invalid_tolerance")
    def test_missing_provenance(self):
        r=compare_spectra(C,loss(C),F,loss(F),
                          coarse_provenance={},fine_provenance=P)
        self.assertFalse(r["stable"])
