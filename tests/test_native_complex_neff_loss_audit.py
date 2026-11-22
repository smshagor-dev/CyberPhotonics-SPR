import math
import unittest
from scripts.native_complex_neff_loss_audit import attenuation_db_per_cm, audit


class NativeLossAuditTests(unittest.TestCase):
    def test_analytic_conversion(self):
        self.assertAlmostEqual(attenuation_db_per_cm(1.0, 1.45, 1e-6), 0.545750, places=5)

    def test_zero_loss(self):
        self.assertEqual(attenuation_db_per_cm(0.8, 1.4, 0), 0)

    def test_bad_values(self):
        for w, n, im in [(True, 1.4, 0.1), (0, 1.4, 0.1),
                         (0.8, 1.4, -0.1), (0.8, math.nan, 0.1),
                         (0.8, 1.4, float("inf"))]:
            with self.subTest(values=(w, n, im)), self.assertRaises(ValueError):
                attenuation_db_per_cm(w, n, im)

    def test_source_and_publication_gates(self):
        base = {"source_type": "open_source_fem_native",
                "field_time_convention": "exp(-i omega t)",
                "propagation_convention": "exp(+i beta z)",
                "records": [{"wavelength_um": 0.8, "neff_real": 1.4,
                             "neff_imag": 1e-5}]}
        result = audit(base)
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])
        for source in ("comsol", "fixture", "synthetic", None):
            with self.subTest(source=source), self.assertRaises(ValueError):
                audit({**base, "source_type": source})

    def test_convention_required(self):
        with self.assertRaises(ValueError):
            audit({"source_type": "open_source_fem_native", "records": [{}]})
