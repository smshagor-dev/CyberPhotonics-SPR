"""Constructed regression fixtures only; no optical-physics evidence."""
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from research_audits.native_grid_audit import audit_grid


class NativeGridAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        rows = [
            dict(ri_index=i, wavelength_index=j, mode_index_native_untracked=0,
                 analyte_ri=ri, wavelength_um=wl, n_eff_real=1.4, n_eff_imag=.001)
            for i, ri in enumerate((1.35, 1.36))
            for j, wl in enumerate((.75, .8))
        ]
        path = self.root / "raw_eigenmode_grid.csv"
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        self.report = dict(
            experiment="unqualified_femwell_full_vector_wavelength_ri_grid",
            completed=True, failures=[], solver_calls_completed=4,
            wavelengths_um=[.75, .8], analyte_ris=[1.35, 1.36],
            num_modes_requested=1,
            artifacts_sha256={"raw_eigenmode_grid.csv": hashlib.sha256(path.read_bytes()).hexdigest()})
        self.save()

    def save(self):
        (self.root / "GRID_READINESS.json").write_text(json.dumps(self.report))

    def test_valid_still_unqualified(self):
        result = audit_grid(self.root)
        self.assertTrue(result["integrity_passed"])
        self.assertEqual(len(result["loss_rows"]), 4)
        self.assertFalse(result["physics_qualified"])
        self.assertFalse(result["publication_ready"])

    def test_tampered_archive(self):
        with (self.root / "raw_eigenmode_grid.csv").open("a") as f:
            f.write("tamper")
        self.assertFalse(audit_grid(self.root)["integrity_passed"])

    def test_missing_solver_call(self):
        self.report["solver_calls_completed"] = 3
        self.save()
        self.assertFalse(audit_grid(self.root)["integrity_passed"])

    def test_missing_mode_cell(self):
        self.report["num_modes_requested"] = 2
        self.save()
        self.assertFalse(audit_grid(self.root)["integrity_passed"])

    def test_path_traversal(self):
        self.report["artifacts_sha256"]["../outside"] = "a" * 64
        self.save()
        self.assertFalse(audit_grid(self.root)["integrity_passed"])

    def test_duplicate_row(self):
        path = self.root / "raw_eigenmode_grid.csv"
        with path.open("a") as f:
            f.write("0,0,0,1.35,0.75,1.4,0.001\n")
        self.report["artifacts_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        self.save()
        self.assertFalse(audit_grid(self.root)["integrity_passed"])


if __name__ == "__main__":
    unittest.main()
