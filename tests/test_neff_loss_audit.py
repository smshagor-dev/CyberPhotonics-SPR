import math
import unittest
from sprpcf.simulation.neff_loss_audit import loss_from_neff

class LossAuditTests(unittest.TestCase):
    def test_zero(self):
        p=loss_from_neff(800, 1.45+0j)
        self.assertEqual(p.loss_db_per_cm,0)
        self.assertEqual(p.power_1cm_fraction,1)
    def test_power_ratio(self):
        p=loss_from_neff(800, 1.45+1e-7j)
        self.assertAlmostEqual(-10*math.log10(p.power_1cm_fraction),p.loss_db_per_cm,places=12)
    def test_wavelength_scaling(self):
        a=loss_from_neff(800,1.4+1e-7j)
        b=loss_from_neff(1600,1.4+1e-7j)
        self.assertAlmostEqual(a.loss_db_per_cm,2*b.loss_db_per_cm)
    def test_negative_loss(self):
        with self.assertRaises(ValueError): loss_from_neff(800,1.4-1e-7j)
    def test_invalid(self):
        for wl,n in [(0,1.4+0j),(float('nan'),1.4+0j),(800,complex(float('nan'),0)),(800,complex(1,float('inf')))]:
            with self.assertRaises(ValueError):loss_from_neff(wl,n)
    def test_no_bool(self):
        with self.assertRaises(TypeError):loss_from_neff(True,1.4+0j)
        with self.assertRaises(TypeError):loss_from_neff(800,True)
