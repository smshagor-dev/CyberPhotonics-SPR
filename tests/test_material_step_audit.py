"""Constructed provenance tests; never numerical Maxwell evidence."""
import unittest
from research_audits.material_step_audit import authorize_material_step


class MaterialStepAuditTests(unittest.TestCase):
    def setUp(self):
        self.kw = dict(mesh_before="m", mesh_after="m",
                       geometry_before="g", geometry_after="g",
                       material_before="a", material_after="a")

    def run_audit(self, **changes):
        return authorize_material_step(**dict(self.kw, **changes))

    def test_same_material(self):
        self.assertTrue(self.run_audit()["diagnostic_authorized"])

    def test_ri_change_explicit(self):
        r = self.run_audit(material_after="b", kind="analyte_ri",
                           before=1.35, after=1.36)
        self.assertTrue(r["diagnostic_authorized"])
        self.assertFalse(r["mode_identity_physically_verified"])
        self.assertFalse(r["publication_ready"])

    def test_wavelength_change_explicit(self):
        self.assertTrue(self.run_audit(material_after="b",
                        kind="wavelength_um", before=.75, after=.8)["diagnostic_authorized"])

    def test_unexplained_material_change(self):
        self.assertFalse(self.run_audit(material_after="b")["diagnostic_authorized"])

    def test_geometry_change(self):
        self.assertFalse(self.run_audit(geometry_after="other")["diagnostic_authorized"])

    def test_mesh_change(self):
        self.assertFalse(self.run_audit(mesh_after="other")["diagnostic_authorized"])

    def test_nonfinite_and_zero_steps(self):
        for value in (float("nan"), float("inf"), 0, -1):
            self.assertFalse(self.run_audit(material_after="b", kind="analyte_ri",
                             before=value, after=1.36)["diagnostic_authorized"])

    def test_unchanged_step(self):
        self.assertFalse(self.run_audit(material_after="b", kind="analyte_ri",
                         before=1.36, after=1.36)["diagnostic_authorized"])

    def test_undeclared_change(self):
        self.assertFalse(self.run_audit(kind="analyte_ri",
                         before=1.35, after=1.36)["diagnostic_authorized"])


if __name__ == "__main__":
    unittest.main()
