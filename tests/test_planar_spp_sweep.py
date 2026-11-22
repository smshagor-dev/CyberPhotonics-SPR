import unittest
from sprpcf.simulation.planar_spp_sweep import planar_reference

class PlanarReferenceTests(unittest.TestCase):
    def test_boundary_and_loss(self):
        r = planar_reference(.8, 1.36, -18+1.5j)
        self.assertLess(r["normalized_tm_boundary_residual"], 1e-12)
        self.assertGreater(r["loss_db_per_cm"], 0)
        self.assertFalse(r["physics_qualified"])
    def test_ri_changes_result(self):
        a = planar_reference(.8, 1.35, -18+1.5j)
        b = planar_reference(.8, 1.37, -18+1.5j)
        self.assertNotEqual(a["n_eff_real"], b["n_eff_real"])
    def test_active_rejected(self):
        with self.assertRaises(ValueError): planar_reference(.8, 1.36, -18-1j)
    def test_unbound_rejected(self):
        with self.assertRaises(ValueError): planar_reference(.8, 1.36, -1+1j)
    def test_nonfinite_rejected(self):
        with self.assertRaises(ValueError): planar_reference(float("nan"), 1.36, -18+1j)
    def test_invalid_ri_rejected(self):
        with self.assertRaises(ValueError): planar_reference(.8, 0, -18+1j)
