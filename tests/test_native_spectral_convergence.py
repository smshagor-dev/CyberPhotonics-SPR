"""Constructed software fixtures: never qualify as numerical-physics evidence."""
import unittest
from research.native_spectral_convergence import Spectrum, assess_spectral_refinement

def samples():
    out = []
    for n, char in ((17, 'a'), (33, 'b'), (65, 'c')):
        x = tuple(700+200*i/(n-1) for i in range(n))
        y = tuple(1+12/(1+((v-800)/22)**2) for v in x)
        out.append(Spectrum(x, y, '1'*64, '2'*64, 'core-mode', char*64))
    return out

class TestSpectralConvergence(unittest.TestCase):
    def test_valid(self):
        result = assess_spectral_refinement(samples())
        self.assertTrue(result['spectral_grid_pass'])
        self.assertFalse(result['qualified_physics'])
        self.assertFalse(result['publication_ready'])

    def test_reused_archive(self):
        a = samples()
        s = a[1]
        a[1] = Spectrum(s.wavelength_nm, s.loss_db_per_cm, s.geometry_sha256,
                        s.material_sha256, s.mode_id, a[0].source_archive_sha256)
        with self.assertRaises(ValueError):
            assess_spectral_refinement(a)

    def test_geometry_change(self):
        a = samples()
        s = a[1]
        a[1] = Spectrum(s.wavelength_nm, s.loss_db_per_cm, '3'*64,
                        s.material_sha256, s.mode_id, s.source_archive_sha256)
        with self.assertRaises(ValueError):
            assess_spectral_refinement(a)

    def test_no_peak(self):
        a = samples()
        s = a[2]
        a[2] = Spectrum(s.wavelength_nm, tuple(range(len(s.wavelength_nm))),
                        s.geometry_sha256, s.material_sha256, s.mode_id, s.source_archive_sha256)
        with self.assertRaises(ValueError):
            assess_spectral_refinement(a)

    def test_insufficient_grids(self):
        with self.assertRaises(ValueError):
            assess_spectral_refinement(samples()[:2])

    def test_negative_loss(self):
        a = samples()
        s = a[0]
        a[0] = Spectrum(s.wavelength_nm, tuple(-v for v in s.loss_db_per_cm),
                        s.geometry_sha256, s.material_sha256, s.mode_id, s.source_archive_sha256)
        with self.assertRaises(ValueError):
            assess_spectral_refinement(a)

    def test_non_nested(self):
        a = samples()
        s = a[1]
        x = list(s.wavelength_nm)
        x[1] += 0.1
        a[1] = Spectrum(tuple(x), s.loss_db_per_cm, s.geometry_sha256,
                        s.material_sha256, s.mode_id, s.source_archive_sha256)
        with self.assertRaises(ValueError):
            assess_spectral_refinement(a)

if __name__ == '__main__':
    unittest.main()
