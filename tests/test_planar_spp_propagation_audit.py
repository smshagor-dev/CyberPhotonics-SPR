"""Regime screens use illustrative permittivities; never physics evidence."""
import unittest
from scripts.planar_spp_propagation_audit import audit_planar_propagation


class PlanarRegimeTests(unittest.TestCase):
    def test_au_like_root_passes_screen_but_not_qualification(self):
        d = audit_planar_propagation(2.25, -20+1.5j, 0.8)
        self.assertTrue(d["planar_reference_screen_pass"])
        self.assertFalse(d["physics_qualified"])
        self.assertFalse(d["publication_ready"])
        self.assertFalse(d["comsol_verified"])

    def test_overdamped_root_not_called_propagating_spp(self):
        d = audit_planar_propagation(2.25, -0.5+0.1j, 0.8)
        self.assertTrue(d["transverse_decay_pass"])
        self.assertTrue(d["interface_residual_pass"])
        self.assertFalse(d["above_dielectric_light_line"])
        self.assertTrue(d["overdamped"])
        self.assertFalse(d["planar_reference_screen_pass"])

    def test_near_pole_overdamped_root_not_promoted(self):
        d = audit_planar_propagation(2.25, -2+0.1j, 0.8)
        self.assertFalse(d["planar_reference_screen_pass"])


if __name__ == "__main__":
    unittest.main()
