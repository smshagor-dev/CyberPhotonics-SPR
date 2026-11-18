"""Software-only regression tests, never physical sensor evidence."""
import unittest
import numpy as np
from scripts.measured_au_planar_spp_reference import planar_spp_spectrum


class TestPlanarMeasuredAu(unittest.TestCase):
    def test_passive_reference_and_attenuation_reciprocity(self):
        rows = planar_spp_spectrum(np.linspace(0.70, 0.90, 9), 1.36)
        self.assertEqual(len(rows), 9)
        for row in rows:
            self.assertGreater(row["au_epsilon_imag"], 0)
            self.assertGreater(row["spp_neff_imag"], 0)
            self.assertAlmostEqual(
                row["attenuation_db_cm"] * row["intensity_1e_length_um"] * 1e-4,
                10 / np.log(10), places=10,
            )

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            planar_spp_spectrum([0.1, 0.2, 0.3])

    def test_nonmonotonic_grid_rejected(self):
        with self.assertRaises(ValueError):
            planar_spp_spectrum([0.8, 0.7, 0.9])

    def test_invalid_ri_rejected(self):
        with self.assertRaises(ValueError):
            planar_spp_spectrum([0.7, 0.8, 0.9], float("nan"))


if __name__ == "__main__":
    unittest.main()
