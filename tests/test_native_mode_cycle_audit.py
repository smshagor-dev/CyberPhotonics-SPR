"""Synthetic software-only regressions; never physical PCF evidence."""
import unittest
import numpy as np
from scripts.native_mode_cycle_audit import overlap_matrix, assign, cycle_audit


class ModeCycleTests(unittest.TestCase):
    def setUp(self):
        self.a = np.eye(3, dtype=complex)
        self.w = np.ones(3)

    def test_phase_invariant(self):
        b = self.a * np.exp(1j * np.array([.3, 1.4, 2.1]))[:, None]
        np.testing.assert_allclose(overlap_matrix(self.a, b, self.w), np.eye(3), atol=1e-14)

    def test_permutation(self):
        self.assertEqual(assign(overlap_matrix(self.a, self.a[[2, 0, 1]], self.w)), (1, 2, 0))

    def test_cycle(self):
        result = cycle_audit([self.a, self.a[[2, 0, 1]], self.a[[1, 2, 0]]], self.w)
        self.assertTrue(result["accepted"])
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_ambiguous(self):
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            assign(np.full((2, 2), .8))

    def test_low_confidence(self):
        with self.assertRaisesRegex(ValueError, "low-confidence"):
            assign(np.eye(2) * .4)

    def test_invalid_threshold(self):
        with self.assertRaises(ValueError):
            assign(np.eye(2), gap=float("nan"))

    def test_nonfinite_field(self):
        a = self.a.copy()
        a[0, 0] = np.nan
        with self.assertRaises(ValueError):
            overlap_matrix(a, self.a, self.w)

    def test_zero_norm(self):
        a = self.a.copy()
        a[0] = 0
        with self.assertRaises(ValueError):
            overlap_matrix(a, self.a, self.w)

    def test_nonpositive_weights(self):
        with self.assertRaises(ValueError):
            overlap_matrix(self.a, self.a, [1, 0, 1])

    def test_dynamic_range(self):
        np.testing.assert_allclose(overlap_matrix(self.a * 1e200, self.a * 1e-200, self.w), np.eye(3))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            overlap_matrix(self.a, self.a[:2], self.w)


if __name__ == "__main__":
    unittest.main()
