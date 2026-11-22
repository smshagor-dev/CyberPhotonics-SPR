"""Integrity-only fixtures: never physics evidence."""
import csv
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from sprpcf.simulation.native_vector_audit import audit_native_vector_grid
from sprpcf.simulation.attenuation_identity import loss_db_per_cm


class NativeVectorAuditTests(unittest.TestCase):
    def test_integrity_and_tamper_detection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "raw_vector_eh_ri0_wl0.npz"
            np.savez_compressed(root / name, mode_0_E=np.array([1+1j]),
                                mode_0_H=np.array([2+1j]))
            csv_name = "raw_eigenmode_grid.csv"
            with (root / csv_name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "ri_index", "wavelength_index", "mode_index_native_untracked",
                    "wavelength_um", "analyte_ri", "n_eff_real", "n_eff_imag",
                    "gold_epsilon_real", "gold_epsilon_imag",
                    "raw_attenuation_db_per_cm_unqualified"])
                writer.writeheader()
                writer.writerow(dict(ri_index=0, wavelength_index=0,
                    mode_index_native_untracked=0, wavelength_um=0.8,
                    analyte_ri=1.36, n_eff_real=1.4, n_eff_imag=0.001,
                    gold_epsilon_real=-20, gold_epsilon_imag=1,
                    raw_attenuation_db_per_cm_unqualified=loss_db_per_cm(0.8, 0.001)))
            flags = dict.fromkeys((
                "sensor_geometry_peer_reviewed_and_frozen", "pml_validated",
                "mesh_convergence_verified", "mode_tracking_physically_verified",
                "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
                "scientific_submission_ready"), False)
            report = dict(flags, completed=True, failures=[],
                wavelengths_um=[0.8], analyte_ris=[1.36],
                num_modes_requested=1, raw_eigenmode_rows=1,
                artifacts_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                  for p in (name, csv_name)})
            (root / "GRID_READINESS.json").write_text(json.dumps(report))
            good = audit_native_vector_grid(root)
            self.assertTrue(good["native_integrity_pass"], good["failures"])
            self.assertFalse(good["physics_qualified"])
            self.assertTrue(good["attenuation_dimensional_identity_pass"])
            with (root / csv_name).open("a") as stream:
                stream.write("tamper")
            bad = audit_native_vector_grid(root)
            self.assertFalse(bad["native_integrity_pass"])
            self.assertTrue(any("checksum mismatch" in s for s in bad["failures"]))


    def test_hashed_but_physically_inconsistent_loss_is_rejected(self):
        """A matching SHA256 cannot make an incorrect attenuation formula valid."""
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = "raw_vector_eh_ri0_wl0.npz"
            np.savez_compressed(root / archive, mode_0_E=np.array([1+0j]),
                                mode_0_H=np.array([1+0j]))
            csv_name = "raw_eigenmode_grid.csv"
            with (root / csv_name).open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "ri_index", "wavelength_index", "mode_index_native_untracked",
                    "wavelength_um", "analyte_ri", "n_eff_real", "n_eff_imag",
                    "gold_epsilon_real", "gold_epsilon_imag",
                    "raw_attenuation_db_per_cm_unqualified"])
                writer.writeheader()
                writer.writerow(dict(ri_index=0, wavelength_index=0,
                    mode_index_native_untracked=0, wavelength_um=0.8,
                    analyte_ri=1.36, n_eff_real=1.4, n_eff_imag=0.001,
                    gold_epsilon_real=-20, gold_epsilon_imag=1,
                    raw_attenuation_db_per_cm_unqualified=100))
            flags = dict.fromkeys((
                "sensor_geometry_peer_reviewed_and_frozen", "pml_validated",
                "mesh_convergence_verified", "mode_tracking_physically_verified",
                "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
                "scientific_submission_ready"), False)
            report = dict(flags, completed=True, failures=[],
                wavelengths_um=[0.8], analyte_ris=[1.36],
                num_modes_requested=1, raw_eigenmode_rows=1,
                artifacts_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                  for p in (archive, csv_name)})
            (root / "GRID_READINESS.json").write_text(json.dumps(report))
            result = audit_native_vector_grid(root)
            self.assertFalse(result["native_integrity_pass"])
            self.assertFalse(result["attenuation_dimensional_identity_pass"])
            self.assertEqual(result["attenuation_rows_checked"], 1)
            self.assertTrue(any("attenuation identity row 1" in x
                                for x in result["failures"]))
            self.assertFalse(result["publication_ready"])

    def test_empty_grid_cannot_pass_vacuously(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw_eigenmode_grid.csv").write_text("ri_index,wavelength_index,mode_index_native_untracked\\n")
            digest = hashlib.sha256((root / "raw_eigenmode_grid.csv").read_bytes()).hexdigest()
            (root / "GRID_READINESS.json").write_text(json.dumps({
                "completed": True, "failures": [], "wavelengths_um": [],
                "analyte_ris": [], "num_modes_requested": 0,
                "raw_eigenmode_rows": 0,
                "artifacts_sha256": {"raw_eigenmode_grid.csv": digest},
            }))
            result = audit_native_vector_grid(root)
            self.assertFalse(result["native_integrity_pass"])
            self.assertTrue(any("invalid or empty" in x for x in result["failures"]))

    def test_unhashed_csv_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw_eigenmode_grid.csv").write_text("ri_index,wavelength_index,mode_index_native_untracked\\n")
            (root / "dummy.txt").write_text("native")
            (root / "GRID_READINESS.json").write_text(json.dumps({
                "completed": True, "failures": [], "wavelengths_um": [0.8],
                "analyte_ris": [1.36], "num_modes_requested": 1,
                "raw_eigenmode_rows": 1,
                "artifacts_sha256": {"dummy.txt": hashlib.sha256(b"native").hexdigest()},
            }))
            result = audit_native_vector_grid(root)
            self.assertFalse(result["native_integrity_pass"])
            self.assertIn("unhashed eigenmode CSV", result["failures"])

    def test_missing_report_is_not_silent(self):
        with TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                audit_native_vector_grid(tmp)


if __name__ == "__main__":
    unittest.main()
