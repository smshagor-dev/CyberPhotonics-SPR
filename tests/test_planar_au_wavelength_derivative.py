"""Source-backed dispersive-Au wavelength derivative audit (analytical planar model only).

A wavelength derivative MUST re-evaluate epsilon_Au(lambda); holding gold fixed
silently measures a different partial derivative. This is not PCF resonance evidence.
"""
import math
import unittest

from sprpcf.simulation.materials_reference import au_johnson_christy_permittivity
from sprpcf.simulation.planar_spp_sweep import planar_reference


def dispersive_loss_derivative(wavelength_um, ri, step_um):
    """Centered finite difference of planar loss with re-evaluated Au dispersion."""
    wl, h = float(wavelength_um), float(step_um)
    if not all(map(math.isfinite, (wl, h, float(ri)))) or h <= 0 or wl <= h:
        raise ValueError("invalid wavelength differentiation interval")
    lo = planar_reference(wl-h, ri, au_johnson_christy_permittivity(wl-h))
    hi = planar_reference(wl+h, ri, au_johnson_christy_permittivity(wl+h))
    return (hi["loss_db_per_cm"]-lo["loss_db_per_cm"])/(2*h)


class SourceBoundWavelengthAudit(unittest.TestCase):
    def test_source_dispersion_is_not_held_fixed(self):
        for wl in (0.75, 0.80, 0.85):
            h = 1e-4
            ri = 1.36
            derivative = dispersive_loss_derivative(wl, ri, h)
            em = au_johnson_christy_permittivity(wl)
            fixed = (planar_reference(wl+h, ri, em)["loss_db_per_cm"]
                     -planar_reference(wl-h, ri, em)["loss_db_per_cm"])/(2*h)
            self.assertTrue(math.isfinite(derivative))
            self.assertGreater(abs(derivative-fixed), 1e-5)

    def test_centered_difference_refines_with_step(self):
        # Away from source-table knots, interpolated Au dispersion is smooth.
        for wl in (0.751, 0.801, 0.851):
            d1 = dispersive_loss_derivative(wl, 1.36, 2e-4)
            d2 = dispersive_loss_derivative(wl, 1.36, 1e-4)
            d3 = dispersive_loss_derivative(wl, 1.36, 5e-5)
            self.assertTrue(all(map(math.isfinite, (d1,d2,d3))))
            self.assertLess(abs(d3-d2), abs(d2-d1)*0.4 + 1e-7)

    def test_invalid_intervals_rejected(self):
        for h in (0, -1e-4, float("nan"), float("inf"), 1):
            with self.assertRaises(ValueError):
                dispersive_loss_derivative(0.8, 1.36, h)


if __name__ == "__main__":
    unittest.main()
