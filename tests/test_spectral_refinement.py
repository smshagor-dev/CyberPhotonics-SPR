"""Constructed Gaussian tests: software regressions, not Maxwell physics."""
import math
import unittest
from research.spectral_refinement import compare_spectra

HASH = "a" * 64

def record(step, *, geom=HASH, ri=1.36, source=None, center=.8):
    count = round(.2 / step)
    x = [.7 + i * .2 / count for i in range(count + 1)]
    y = [2 + 8 * math.exp(-.5 * ((v-center)/.025)**2) for v in x]
    return dict(wavelength_um=x, loss_db_cm=y, geometry_sha256=geom,
                ri=ri, source_id=source or str(step))

class TestRefinement(unittest.TestCase):
    def setUp(self):
        self.coarse = record(.002)
        self.fine = record(.001)

    def test_constructed_stable_not_physics(self):
        r = compare_spectra(self.coarse, self.fine)
        self.assertTrue(r.stable, r.reasons)
        self.assertFalse(r.physics_qualified)
        self.assertFalse(r.publication_ready)
        self.assertFalse(r.comsol_verified)

    def test_geometry_mismatch(self):
        self.assertFalse(compare_spectra(self.coarse, record(.001, geom="b"*64)).stable)

    def test_ri_mismatch(self):
        self.assertFalse(compare_spectra(self.coarse, record(.001, ri=1.37)).stable)

    def test_duplicate_source(self):
        self.assertFalse(compare_spectra(self.coarse, record(.001, source="0.002")).stable)

    def test_insufficient_refinement(self):
        self.assertFalse(compare_spectra(self.coarse, record(.0018)).stable)

    def test_shifted_peak(self):
        self.assertFalse(compare_spectra(self.coarse, record(.001, center=.81)).stable)

    def test_missing_source(self):
        r = self.fine.copy()
        del r["source_id"]
        self.assertFalse(compare_spectra(self.coarse, r).stable)

    def test_sparse_grid(self):
        self.assertFalse(compare_spectra(record(.05), self.fine).stable)

    def test_invalid_tolerance(self):
        self.assertFalse(compare_spectra(self.coarse, self.fine, resonance_tol_um=-1).stable)

if __name__ == "__main__":
    unittest.main()
