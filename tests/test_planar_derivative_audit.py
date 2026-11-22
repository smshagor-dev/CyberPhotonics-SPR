"""Independent derivative and attenuation-unit audits for planar TM-SPP reference.

These are mathematical reference checks, NOT PCF sensitivity or sensor evidence.
"""
import cmath
import math
import unittest

from sprpcf.simulation.planar_spp_sweep import planar_reference
from sprpcf.simulation.materials_reference import au_johnson_christy_permittivity


class PlanarDerivativeAudit(unittest.TestCase):
    def test_complex_ri_derivative_against_centered_difference(self):
        # Differentiate n_eff = sqrt(eps_m*n^2/(eps_m+n^2)) analytically
        # at fixed wavelength and fixed dispersive gold permittivity.
        for wavelength in (0.75, 0.80, 0.85):
            gold = complex(au_johnson_christy_permittivity(wavelength))
            for ri in (1.35, 1.36, 1.37):
                h = 1e-5
                neff = cmath.sqrt(gold * ri**2 / (gold + ri**2))
                analytic = gold**2 * ri / ((gold + ri**2)**2 * neff)
                plus = planar_reference(wavelength, ri + h, gold)
                minus = planar_reference(wavelength, ri - h, gold)
                numeric = complex(
                    (plus["n_eff_real"] - minus["n_eff_real"]) / (2*h),
                    (plus["n_eff_imag"] - minus["n_eff_imag"]) / (2*h),
                )
                self.assertLess(abs(numeric - analytic), 2e-7)
                self.assertTrue(math.isfinite(analytic.real))
                self.assertTrue(math.isfinite(analytic.imag))
                self.assertFalse(plus["physics_qualified"])
                self.assertFalse(minus["publication_ready"])

    def test_attenuation_units_are_independently_reconstructed(self):
        # exp(i beta z) power decay -> 10 log10(P0/Pz) per cm.
        # k0 [cm^-1] = 2*pi/(wavelength_um*1e-4).
        for wavelength in (0.75, 0.80, 0.85):
            gold = complex(au_johnson_christy_permittivity(wavelength))
            row = planar_reference(wavelength, 1.36, gold)
            alpha_power_cm = 2 * (2*math.pi/(wavelength*1e-4)) * row["n_eff_imag"]
            independent_db_cm = 10/math.log(10) * alpha_power_cm
            self.assertAlmostEqual(row["loss_db_per_cm"], independent_db_cm, places=9)
            self.assertLess(row["normalized_tm_boundary_residual"], 1e-9)
            self.assertFalse(row["comsol_verified"])


if __name__ == "__main__":
    unittest.main()
