import unittest
import numpy as np
from research.poynting_absorption_audit import audit_transverse_fields as audit

class PoyntingAuditTests(unittest.TestCase):
    def setUp(self):
        self.xy=np.array([[0.,0.],[1e-6,0.],[0.,1e-6]])
        self.tri=np.array([[0,1,2]])
        self.e=np.tile([1.+0j,0,0],(3,1))
        self.h=np.tile([0,1.+0j,0],(3,1))
    def check(self,loss=0.,e=None,h=None,tri=None):
        return audit(self.xy,self.tri if tri is None else tri,
                     self.e if e is None else e,self.h if h is None else h,
                     np.array([loss]),800e-9)
    def test_forward_flux(self):
        self.assertAlmostEqual(self.check()["forward_power_w"],2.5e-13)
    def test_lossless(self):
        self.assertEqual(self.check()["absorption_loss_db_per_cm"],0)
    def test_passive(self):
        self.assertGreater(self.check(.5)["absorption_w_per_m"],0)
        self.assertFalse(self.check(.5)["physics_qualified"])
    def test_phase_invariance(self):
        phase=np.exp(.75j)
        self.assertAlmostEqual(self.check(.5)["absorption_alpha_np_per_m"],
                               self.check(.5,self.e*phase,self.h*phase)["absorption_alpha_np_per_m"])
    def test_scale_invariance(self):
        self.assertAlmostEqual(self.check(.5)["absorption_alpha_np_per_m"],
                               self.check(.5,self.e*3,self.h*3)["absorption_alpha_np_per_m"])
    def test_gain_rejected(self):
        with self.assertRaises(ValueError): self.check(-.1)
    def test_reverse_flux_rejected(self):
        with self.assertRaises(ValueError): self.check(0,h=-self.h)
    def test_clockwise_rejected(self):
        with self.assertRaises(ValueError): self.check(0,tri=np.array([[0,2,1]]))
    def test_quadratic_quadrature(self):
        e=self.e.copy();e[:,0]=self.xy[:,0]/1e-6
        got=self.check(1.,e=e)
        self.assertAlmostEqual(got["forward_power_w"],.25e-12/3)
        expected=.5*(2*np.pi*299792458/800e-9)*8.8541878128e-12*.5e-12/6
        self.assertAlmostEqual(got["absorption_w_per_m"]/expected,1.)
if __name__=="__main__": unittest.main()
