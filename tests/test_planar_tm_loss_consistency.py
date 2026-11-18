"""Constructed passive parameters only: software/numerical control, not measured Au."""
import math
import unittest
from scripts.planar_tm_loss_consistency import planar_tm_loss_consistency


class TestPlanarLossConsistency(unittest.TestCase):
    def test_loss_and_propagation_length_reciprocity(self):
        for wavelength, metal, dielectric in (
            (.8, -24+1.5j, 1.77), (.65, -12+1.2j, 1.0), (1.0, -40+3j, 2.0)
        ):
            result = planar_tm_loss_consistency(wavelength, metal, dielectric)
            self.assertAlmostEqual(result["beta_imag_per_um"] * 2 * result["intensity_1e_length_um"], 1)
            self.assertAlmostEqual(result["loss_db_per_cm"] * result["intensity_1e_length_um"] / 10000, 10/math.log(10))
            self.assertGreater(result["loss_db_per_cm"], 0)
            for key in ("physics_qualified", "publication_ready", "comsol_verified",
                        "source_material_verified", "independent_fem_verified"):
                self.assertFalse(result[key])

    def test_wavelength_scaling_for_constant_permittivity(self):
        short = planar_tm_loss_consistency(.8, -24+1.5j, 1.77)
        long = planar_tm_loss_consistency(1.6, -24+1.5j, 1.77)
        self.assertAlmostEqual(short["neff_imag"], long["neff_imag"], places=8)
        self.assertAlmostEqual(short["loss_db_per_cm"]/long["loss_db_per_cm"], 2, places=7)

    def test_reject_active_and_unbound(self):
        for eps_m in (-24-1j, -.5+1j):
            with self.assertRaises(ValueError):
                planar_tm_loss_consistency(.8, eps_m, 1.77)


if __name__ == "__main__":
    unittest.main()
