"""Analytical plane-wave regression; these are NOT PCF sensor measurements."""
import unittest
import numpy as np
from research_audits.maxwell_residual_audit import audit_uniform_fields, C0, MU0


class TestMaxwellResidual(unittest.TestCase):
    def setUp(self):
        self.lam = 0.8e-6
        self.n = 1.45
        self.beta = 2 * np.pi * self.n / self.lam
        self.er = np.full((16, 18), self.n**2, dtype=complex)
        self.E = np.zeros((3, 16, 18), dtype=complex)
        self.H = np.zeros_like(self.E)
        self.E[0] = 1
        self.H[1] = self.n / (MU0 * C0)

    def run_audit(self, **kw):
        return audit_uniform_fields(self.E, self.H, self.er,
                                    dx_m=2e-8, dy_m=2e-8,
                                    wavelength_m=self.lam,
                                    beta_per_m=self.beta, **kw)

    def test_analytical_plane_wave(self):
        r = self.run_audit()
        self.assertLess(r["curl_E_residual"], 1e-12)
        self.assertLess(r["curl_H_residual"], 1e-12)
        self.assertFalse(r["physics_qualified"])

    def test_corrupt_magnetic_field(self):
        self.H[1] *= 0.7
        self.assertGreater(self.run_audit()["curl_E_residual"], 0.1)

    def test_wrong_propagation_constant(self):
        r = audit_uniform_fields(self.E, self.H, self.er, dx_m=2e-8,
                                 dy_m=2e-8, wavelength_m=self.lam,
                                 beta_per_m=self.beta * 1.2)
        self.assertGreater(r["curl_E_residual"], 0.05)

    def test_active_material_rejected(self):
        self.er[:] = 1 - 1j
        with self.assertRaisesRegex(ValueError, "nonpassive"):
            self.run_audit()

    def test_nan_rejected(self):
        self.E[0, 4, 4] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.run_audit()

    def test_insufficient_interior_rejected(self):
        with self.assertRaisesRegex(ValueError, "interior"):
            self.run_audit(trim=7)

    def test_zero_fields_rejected(self):
        self.E[:] = 0
        self.H[:] = 0
        with self.assertRaisesRegex(ValueError, "zero electromagnetic"):
            self.run_audit()
