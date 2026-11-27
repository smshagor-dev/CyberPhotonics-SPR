"""Software regression only; illustrative material values are not measured Au."""
import cmath
import unittest
from scripts.planar_tm_root_audit import numerical_planar_tm_root


class TestPlanarRoot(unittest.TestCase):
    def test_numerical_vs_analytic(self):
        for wavelength, metal, dielectric in [
            (0.8, -24+1.5j, 1.77),
            (0.65, -12+1.2j, 1.0),
            (1.0, -40+3j, 2.0),
        ]:
            result = numerical_planar_tm_root(wavelength, metal, dielectric)
            expected = cmath.sqrt(metal*dielectric/(metal+dielectric))
            self.assertLess(abs(complex(result["neff_real"], result["neff_imag"])-expected), 1e-8)
            self.assertLess(result["boundary_relative_residual"], 1e-9)
            self.assertFalse(result["physics_qualified"])

    def test_bad_inputs(self):
        for args in [(0, -24+1j, 1), (.8, -.5+1j, 1),
                     (.8, -24-1j, 1), (.8, -24+1j, 0)]:
            with self.assertRaises(ValueError):
                numerical_planar_tm_root(*args)


if __name__ == "__main__":
    unittest.main()
