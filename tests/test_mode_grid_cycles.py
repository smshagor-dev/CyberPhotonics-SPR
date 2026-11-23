"""Adversarial tests for mode transport around RI/wavelength plaquettes."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("mode_grid_cycles", Path(__file__).resolve().parents[1] / "scripts" / "audit_mode_grid_cycles.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
audit_grid = module.audit_grid


def fixture():
    rows, edges = [], []
    for r in range(2):
        for w in range(2):
            for m in range(2):
                rows.append(dict(ri_index=r, wavelength_index=w, mode_index_native_untracked=m,
                                 n_eff_real=1.4, n_eff_imag=.001))
    for r in range(2):
        for m in range(2):
            edges.append(dict(varying="wavelength", ri_index=r, previous_wl_index=0,
                              current_wl_index=1, previous_mode_index=m, current_mode_index=m,
                              bidirectional_min_overlap_margin=.4, vector_eh_poynting_overlap_abs=.8))
    for w in range(2):
        for m in range(2):
            edges.append(dict(varying="analyte_ri", wavelength_index=w, previous_ri_index=0,
                              current_ri_index=1, previous_mode_index=m, current_mode_index=m,
                              bidirectional_min_overlap_margin=.4, vector_eh_poynting_overlap_abs=.8))
    return rows, edges


class ModeCycleAuditTests(unittest.TestCase):
    def test_closed_cycle_does_not_promote_publication(self):
        rows, edges = fixture()
        result = audit_grid(rows, edges)
        self.assertTrue(result["cycle_consistent"])
        self.assertEqual(result["cycles_checked"], 1)
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["physical_core_spp_identity_verified"])

    def test_inconsistent_cycle(self):
        rows, edges = fixture()
        for edge in edges:
            if edge["varying"] == "analyte_ri" and edge["wavelength_index"] == 1:
                edge["current_mode_index"] = 1 - edge["current_mode_index"]
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_ambiguous_edge(self):
        rows, edges = fixture()
        edges[0]["bidirectional_min_overlap_margin"] = 0
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_missing_edge(self):
        rows, edges = fixture()
        self.assertFalse(audit_grid(rows, edges[:-1])["cycle_consistent"])

    def test_duplicate_eigenmode(self):
        rows, edges = fixture()
        self.assertFalse(audit_grid(rows + [rows[0]], edges)["cycle_consistent"])

    def test_fractional_mode_index_rejected(self):
        rows, edges = fixture()
        edges[0]["previous_mode_index"] = 0.5
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_out_of_range_overlap_rejected(self):
        rows, edges = fixture()
        edges[0]["vector_eh_poynting_overlap_abs"] = 1.2
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_boolean_grid_index_rejected(self):
        rows, edges = fixture()
        rows[0]["ri_index"] = True
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_noncanonical_grid_index_rejected(self):
        rows, edges = fixture()
        rows[0]["wavelength_index"] = "00"
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_margin_cannot_exceed_best_overlap(self):
        rows, edges = fixture()
        edges[0]["vector_eh_poynting_overlap_abs"] = .2
        edges[0]["bidirectional_min_overlap_margin"] = .4
        result = audit_grid(rows, edges)
        self.assertFalse(result["cycle_consistent"])
        self.assertTrue(any("invalid overlap" in x for x in result["failures"]))

    def test_boolean_overlap_rejected(self):
        rows, edges = fixture()
        edges[0]["vector_eh_poynting_overlap_abs"] = True
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_boolean_margin_rejected(self):
        rows, edges = fixture()
        edges[0]["bidirectional_min_overlap_margin"] = True
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_negative_margin_rejected(self):
        rows, edges = fixture()
        edges[0]["bidirectional_min_overlap_margin"] = -0.1
        self.assertFalse(audit_grid(rows, edges)["cycle_consistent"])

    def test_out_of_range_source_mode_rejected(self):
        rows, edges = fixture()
        edges[0]["previous_mode_index"] = 2
        result = audit_grid(rows, edges)
        self.assertFalse(result["cycle_consistent"])
        self.assertTrue(any("out of range" in error for error in result["failures"]))

    def test_out_of_range_target_mode_rejected(self):
        rows, edges = fixture()
        edges[0]["current_mode_index"] = 2
        result = audit_grid(rows, edges)
        self.assertFalse(result["cycle_consistent"])
        self.assertTrue(any("out of range" in error for error in result["failures"]))

    def test_stricter_margin(self):
        rows, edges = fixture()
        self.assertFalse(audit_grid(rows, edges, min_margin=.5)["cycle_consistent"])


if __name__ == "__main__":
    unittest.main()
