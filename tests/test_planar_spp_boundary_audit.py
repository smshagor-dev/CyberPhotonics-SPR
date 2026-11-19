"""Illustrative material values only; never measured Au physics evidence."""
import cmath
import unittest
from sprpcf.simulation.planar_spp_boundary_audit import audit_tm_boundary


class TestBoundaryAudit(unittest.TestCase):
    def setUp(self):
        self.kw = dict(wavelength_um=.8, dielectric_ri=1.35,
                       epsilon_m_real=-24., epsilon_m_imag=1.5)

    def test_exact_dispersion_residual(self):
        ed = self.kw["dielectric_ri"]**2
        em = complex(-24, 1.5)
        n = cmath.sqrt(ed*em/(ed+em))
        r = audit_tm_boundary(**self.kw, neff_real=n.real, neff_imag=n.imag)
        self.assertLess(r["normalized_tm_boundary_residual"], 1e-12)
        self.assertFalse(r["publication_ready"])

    def test_perturbed_mode_detected(self):
        ed = self.kw["dielectric_ri"]**2
        em = complex(-24, 1.5)
        n = cmath.sqrt(ed*em/(ed+em))
        r = audit_tm_boundary(**self.kw, neff_real=n.real+.03, neff_imag=n.imag)
        self.assertGreater(r["normalized_tm_boundary_residual"], 1e-3)

    def test_invalid_and_unbound(self):
        for patch in ({"neff_imag": -.01}, {"neff_real": float("nan")},
                      {"wavelength_um": 0}, {"epsilon_m_real": -1},
                      {"dielectric_ri": True}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                audit_tm_boundary(**{**self.kw, "neff_real": 1.4,
                                     "neff_imag": .01, **patch})

    def test_lossless_limit_not_misrepresented(self):
        with self.assertRaises(ValueError):
            audit_tm_boundary(**{**self.kw, "epsilon_m_imag": 0},
                              neff_real=1.4, neff_imag=.01)
