"""Constructed records exercise validation only; they are NOT Maxwell evidence."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research"))
from independent_solver_gate import compare


def row(solver, implementation, run, field):
    return dict(solver=solver, implementation=implementation,
                geometry_sha256="a"*64, au_data_sha256="b"*64,
                run_sha256=run*64, field_sha256=field*64,
                mode_id="core-1", wavelength_nm=800.0, analyte_ri=1.36,
                neff_real=1.44, neff_imag=0.0001, overlap_verified=True)


class IndependentSolverGateTests(unittest.TestCase):
    def setUp(self):
        self.a = row("femwell", "skfem", "c", "d")
        self.b = row("fenicsx", "dolfinx", "e", "f")

    def test_diagnostic_never_qualifies_publication(self):
        r = compare(self.a, self.b)
        self.assertTrue(r["comparison_pass"])
        self.assertFalse(r["publication_qualified"])
        self.assertFalse(r["comsol_verified"])

    def test_same_implementation(self):
        self.b["implementation"] = "skfem"
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_reused_archive(self):
        self.b["field_sha256"] = self.a["field_sha256"]
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_different_au_data(self):
        self.b["au_data_sha256"] = "9"*64
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_different_wavelength(self):
        self.b["wavelength_nm"] = 801.0
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_untracked_mode(self):
        self.b["overlap_verified"] = False
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_disagreeing_loss(self):
        self.b["neff_imag"] *= 2
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_nonfinite_eigenvalue(self):
        self.b["neff_real"] = float("nan")
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_comsol_provenance(self):
        self.b["solver"] = "comsol"
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])

    def test_negative_attenuation(self):
        self.b["neff_imag"] = -0.0001
        self.assertFalse(compare(self.a, self.b)["comparison_pass"])


if __name__ == "__main__":
    unittest.main()
