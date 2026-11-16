import unittest
from complex_richardson_direction import extrapolate

class TestComplexRichardson(unittest.TestCase):
    def test_exact_complex_second_order(self):
        z = 1.35 + .002j
        h = [.04,.02,.01]
        vals = [z + (.2+.01j)*x*x for x in h]
        out = extrapolate(h,vals)
        self.assertTrue(out["valid"],out["reason"])
        self.assertAlmostEqual(out["observed_order"],2,places=8)
        self.assertAlmostEqual(out["limit_real"],z.real,places=10)
        self.assertAlmostEqual(out["limit_imag"],z.imag,places=10)
        self.assertFalse(out["physics_qualified"])
        self.assertFalse(out["publication_ready"])
    def test_oscillation(self):
        self.assertFalse(extrapolate([4,2,1],[1+.1j,1.1+.1j,1.05+.1j])["valid"])
    def test_orthogonal_drift(self):
        self.assertIn("direction",extrapolate([4,2,1],[1+.1j,1.1+.1j,1.1+.05j])["reason"])
    def test_nonuniform(self):
        self.assertFalse(extrapolate([4,2,1.5],[1.2,1.1,1.05])["valid"])
    def test_zero_drift(self):
        self.assertFalse(extrapolate([4,2,1],[1.2,1.1,1.1])["valid"])
    def test_nonfinite(self):
        self.assertFalse(extrapolate([4,2,1],[1.2,float("nan"),1.0])["valid"])
    def test_bool(self):
        self.assertFalse(extrapolate([4,2,True],[1.2,1.1,1.0])["valid"])
