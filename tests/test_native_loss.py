"""Software-only regression checks for native complex-index loss conversion."""
import math
import unittest
from scripts.native_loss import confinement_loss_db_per_cm, spectrum_loss


class TestNativeLoss(unittest.TestCase):
    def test_analytic(self):
        expected = 20 / math.log(10) * 2 * math.pi * 1e6 * 1e-6 / 100
        self.assertAlmostEqual(confinement_loss_db_per_cm(1.0, 1.45 + 1e-6j), expected)

    def test_zero(self):
        self.assertEqual(confinement_loss_db_per_cm(0.8, 1.5 + 0j), 0)

    def test_inverse_wavelength(self):
        a = confinement_loss_db_per_cm(0.5, 1.4 + 1e-6j)
        b = confinement_loss_db_per_cm(1.0, 1.4 + 1e-6j)
        self.assertAlmostEqual(a, 2*b)

    def test_negative(self):
        with self.assertRaises(ValueError):
            confinement_loss_db_per_cm(0.8, 1.4 - 1e-6j)

    def test_convention(self):
        with self.assertRaises(ValueError):
            confinement_loss_db_per_cm(0.8, 1.4 + 1e-6j, convention="unknown")

    def test_nan(self):
        with self.assertRaises(ValueError):
            confinement_loss_db_per_cm(float("nan"), 1.4 + 1e-6j)

    def test_noncomplex(self):
        with self.assertRaises(ValueError):
            confinement_loss_db_per_cm(0.8, 1.4)

    def test_unsorted(self):
        with self.assertRaises(ValueError):
            spectrum_loss([{"wavelength_um": .8, "neff": 1+1e-6j},
                           {"wavelength_um": .7, "neff": 1+1e-6j}])

    def test_duplicate(self):
        with self.assertRaises(ValueError):
            spectrum_loss([{"wavelength_um": .8, "neff": 1+1e-6j},
                           {"wavelength_um": .8, "neff": 1+1e-6j}])

    def test_valid(self):
        self.assertEqual(len(spectrum_loss([{"wavelength_um": .7, "neff": 1+1e-6j},
                                            {"wavelength_um": .8, "neff": 1+2e-6j}])), 2)

    def test_boolean_wavelength(self):
        with self.assertRaises(ValueError):
            confinement_loss_db_per_cm(True, 1.4+1e-6j)

    def test_infinite_loss(self):
        with self.assertRaises(ValueError):
            confinement_loss_db_per_cm(1e-320, 1.4+1e-6j)


if __name__ == "__main__":
    unittest.main()
