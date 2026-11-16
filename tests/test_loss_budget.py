import math
import unittest
from research.loss_budget import audit_loss_budget as audit


class LossBudgetTests(unittest.TestCase):
    def test_exact_closure(self):
        a = 10.
        r = audit(a*800e-9/(4*math.pi), 800e-9, 7., leakage_alpha_np_per_m=3.)
        self.assertTrue(r["budget_verified"])
        self.assertFalse(r["publication_ready"])

    def test_unmeasured_leakage_not_verified(self):
        self.assertFalse(audit(0., 800e-9, 0.)["budget_verified"])

    def test_missing_leakage_is_not_zero(self):
        self.assertIn("not_independently_measured", audit(1e-6, 800e-9, 10.)["reason"])

    def test_nonclosing_budget(self):
        self.assertFalse(audit(1e-6, 800e-9, 1., leakage_alpha_np_per_m=0.)["budget_verified"])

    def test_negative_residual_not_assumed_gain(self):
        r = audit(0., 800e-9, 1., leakage_alpha_np_per_m=0.)
        self.assertLess(r["unexplained_alpha_np_per_m"], 0)
        self.assertFalse(r["budget_verified"])

    def test_reject_negative(self):
        for v in [-1., float("nan"), float("inf"), True]:
            with self.assertRaises(ValueError):
                audit(v, 800e-9, 0.)

    def test_reject_bad_leakage(self):
        for v in [-1., float("nan"), True]:
            with self.assertRaises(ValueError):
                audit(0., 800e-9, 0., leakage_alpha_np_per_m=v)

    def test_units(self):
        r = audit(1e-6, 1e-6, 0.)
        self.assertAlmostEqual(r["eigenmode_alpha_np_per_m"], 4*math.pi)
        self.assertAlmostEqual(r["eigenmode_loss_db_per_cm"], 4*math.pi*10/math.log(10)/100)


if __name__ == "__main__":
    unittest.main()
