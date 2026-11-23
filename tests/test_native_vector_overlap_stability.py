"""Numerical-stability regressions; synthetic diagnostics, not Maxwell evidence."""
import unittest
import numpy as np
from scripts.native_vector_mode_overlap import overlap, match_modes

class StableOverlapTests(unittest.TestCase):
    def setUp(self):
        self.a=np.eye(4,3,dtype=complex)
        self.b=np.roll(self.a,1,axis=0)
        self.w=np.ones(4)

    def test_large_amplitude_ratio(self):
        self.assertAlmostEqual(overlap(self.a*1e200,self.a*1e200,self.a*1e-200,self.a*1e-200,self.w),1)

    def test_large_quadrature(self):
        self.assertAlmostEqual(overlap(self.a,self.a,self.a,self.a,self.w*1e300),1)

    def test_nonfinite_minimum(self):
        with self.assertRaises(ValueError):
            match_modes([(self.a,self.a)],[(self.a,self.a)],self.w,minimum=float("nan"))

    def test_nonfinite_margin(self):
        with self.assertRaises(ValueError):
            match_modes([(self.a,self.a)],[(self.a,self.a)],self.w,margin=float("inf"))

    def test_assignment_and_flags(self):
        result=match_modes([(self.a,self.a),(self.b,self.b)],[(self.b,self.b),(self.a,self.a)],self.w)
        self.assertEqual(result["assignment"],[1,0])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

if __name__=="__main__":
    unittest.main()
