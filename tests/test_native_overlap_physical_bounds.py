"""Adversarial tests for native E/H mode-transport evidence validation."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "audit_mode_grid_cycles",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_mode_grid_cycles.py",
)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


def example():
    rows = [
        dict(ri_index=r, wavelength_index=w, mode_index_native_untracked=0,
             n_eff_real="1.44", n_eff_imag="0.001")
        for r in range(2) for w in range(2)
    ]
    edges = []
    for r in range(2):
        edges.append(dict(varying="wavelength", ri_index=r,
                          previous_wl_index=0, current_wl_index=1,
                          previous_mode_index=0, current_mode_index=0,
                          bidirectional_min_overlap_margin=0.4,
                          vector_eh_poynting_overlap_abs=0.8))
    for w in range(2):
        edges.append(dict(varying="analyte_ri", wavelength_index=w,
                          previous_ri_index=0, current_ri_index=1,
                          previous_mode_index=0, current_mode_index=0,
                          bidirectional_min_overlap_margin=0.4,
                          vector_eh_poynting_overlap_abs=0.8))
    return rows, edges


class NativeOverlapTests(unittest.TestCase):
    def test_valid_cycle_is_only_diagnostic(self):
        rows, edges = example()
        result = module.audit_grid(rows, edges)
        self.assertTrue(result["cycle_consistent"])
        self.assertFalse(result["physical_core_spp_identity_verified"])
        self.assertFalse(result["publication_ready"])
        self.assertFalse(result["comsol_verified"])

    def test_margin_exceeding_overlap_rejected(self):
        rows, edges = example()
        edges[0]["bidirectional_min_overlap_margin"] = 0.9
        self.assertFalse(module.audit_grid(rows, edges)["cycle_consistent"])

    def test_boolean_margin_rejected(self):
        rows, edges = example()
        edges[0]["bidirectional_min_overlap_margin"] = True
        self.assertFalse(module.audit_grid(rows, edges)["cycle_consistent"])

    def test_boolean_overlap_rejected(self):
        rows, edges = example()
        edges[0]["vector_eh_poynting_overlap_abs"] = True
        self.assertFalse(module.audit_grid(rows, edges)["cycle_consistent"])


if __name__ == "__main__":
    unittest.main()
