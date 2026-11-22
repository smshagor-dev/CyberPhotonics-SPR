"""Analytical planar TM reference cross-check only; no PCF physics qualification."""
import cmath
import unittest
from sprpcf.simulation.planar_tm_newton import solve_planar_tm_root

class TestIndependentPlanarTMRoot(unittest.TestCase):
    def test_passive_cases(self):
        for ed, em in [(1.35**2,-20+1.5j),(1.37**2,-25+2j),(1.4**2,-12+0.8j)]:
            with self.subTest(ed=ed, em=em):
                result = solve_planar_tm_root(.8, ed, em)
                expected = cmath.sqrt(ed*em/(ed+em))
                actual = complex(result["n_eff_real"],result["n_eff_imag"])
                self.assertLess(abs(actual-expected),1e-9)
                self.assertLess(result["normalized_tm_boundary_residual"],1e-11)
                self.assertFalse(result["physics_qualified"])
                self.assertFalse(result["publication_ready"])
                self.assertFalse(result["comsol_verified"])
    def test_invalid_metal(self):
        for metal in (-20-1j, 2+1j):
            with self.assertRaises(ValueError):
                solve_planar_tm_root(.8,1.8,metal)
    def test_nonfinite_wavelength(self):
        with self.assertRaises(ValueError):
            solve_planar_tm_root(float("nan"),1.8,-20+1j)

if __name__ == "__main__":
    unittest.main()
