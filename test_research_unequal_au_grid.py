"""Constructed numerical fixtures only; not Maxwell physics evidence."""
import unittest
from research_unequal_au_grid import audit, observed_order

H=[.030,.025,.020]
def check(z, **kw):
    defaults=dict(geometry_ids=['a'*64]*3, material_ids=['b'*64]*3,
                  pml_ids=['c'*64]*3, archive_hashes=[str(i)*64 for i in (1,2,3)],
                  mode_tracking_verified=True)
    defaults.update(kw)
    return audit(H,z,**defaults)

class TestUnequalGrid(unittest.TestCase):
    def test_quadratic_reference(self):
        r=check([complex(1.33+h*h,.01+h*h/10) for h in H])
        self.assertTrue(r['convergence_diagnostic_passed'],r)
        self.assertAlmostEqual(r['observed_order'],2,places=8)
        self.assertAlmostEqual(r['richardson_neff'][0],1.33,places=9)
        self.assertFalse(r['physics_qualified'])
        self.assertFalse(r['publication_ready'])
    def test_direction_reversal(self):
        r=check([1.35+.01j,1.34+.01j,1.345+.01j])
        self.assertFalse(r['convergence_diagnostic_passed'])
    def test_nonpassive_extrapolate(self):
        r=check([complex(1.4+h*h,-.00001+.1*h*h) for h in H])
        self.assertTrue(r['integrity_passed'])
        self.assertFalse(r['convergence_diagnostic_passed'])
    def test_unverified_mode(self):
        self.assertFalse(check([1.3+.01j]*3,mode_tracking_verified=False)['integrity_passed'])
    def test_changed_material(self):
        self.assertFalse(check([1.3+.01j]*3,material_ids=['b'*64,'b'*64,'d'*64])['integrity_passed'])
    def test_extreme_drift(self):
        with self.assertRaises(ValueError):
            observed_order(H,1e300)
    def test_invalid_spacing(self):
        with self.assertRaises(ValueError):
            observed_order([.03,.03,.02],2)

if __name__=='__main__':
    unittest.main()
