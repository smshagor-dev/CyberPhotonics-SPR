"""Measured Au reference: planar Newton-vs-closed-form audit, NOT PCF evidence."""
import cmath
import math
import unittest

from sprpcf.simulation.materials_reference import (
    au_johnson_christy_permittivity,
    au_reference_manifest,
)
from sprpcf.simulation.planar_tm_newton import solve_planar_tm_root


class TestMeasuredAuPlanarNewton(unittest.TestCase):
    def test_measured_au_nine_point_grid(self):
        manifest = au_reference_manifest()
        self.assertEqual(manifest["row_count"], 49)
        self.assertEqual(manifest["source_type"], "measured_bulk_film_reference")
        self.assertEqual(len(manifest["dataset_sha256"]), 64)
        for wavelength in (0.75, 0.80, 0.85):
            epsilon_m = complex(au_johnson_christy_permittivity(wavelength))
            self.assertGreater(epsilon_m.imag, 0)
            for ri in (1.35, 1.36, 1.37):
                with self.subTest(wavelength=wavelength, ri=ri):
                    epsilon_d = ri * ri
                    numerical = solve_planar_tm_root(wavelength, epsilon_d, epsilon_m)
                    analytical = cmath.sqrt(epsilon_d * epsilon_m / (epsilon_d + epsilon_m))
                    actual = complex(numerical["n_eff_real"], numerical["n_eff_imag"])
                    self.assertLess(abs(actual - analytical), 1e-8)
                    self.assertLess(numerical["normalized_tm_boundary_residual"], 1e-10)
                    self.assertGreater(actual.imag, 0)
                    self.assertFalse(numerical["physics_qualified"])
                    self.assertFalse(numerical["publication_ready"])
                    self.assertFalse(numerical["comsol_verified"])

    def test_source_extrapolation_is_rejected(self):
        for wavelength in (0.18, 2.0, float("nan")):
            with self.subTest(wavelength=wavelength):
                with self.assertRaises(ValueError):
                    au_johnson_christy_permittivity(wavelength)


if __name__ == "__main__":
    unittest.main()
