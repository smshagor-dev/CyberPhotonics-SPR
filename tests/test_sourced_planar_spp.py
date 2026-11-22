"""Source-backed planar Au sweep: numerical reference, never PCF qualification."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from sprpcf.simulation.planar_spp_sweep import run_sourced_sweep

class SourcedPlanarSweepTests(unittest.TestCase):
    def test_measured_source_native_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "sweep.json"
            digest = run_sourced_sweep([0.75, 0.80, 0.85], [1.35, 1.36, 1.37], target)
            raw = target.read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            data = json.loads(raw)
            self.assertEqual(len(data["rows"]), 9)
            self.assertEqual(data["gold"]["row_count"], 49)
            self.assertEqual(len(data["gold"]["dataset_sha256"]), 64)
            self.assertEqual(data["failure_count"], 0)
            for flag in ("physics_qualified", "publication_ready", "comsol_verified"):
                self.assertIs(data[flag], False)
            for row in data["rows"]:
                self.assertGreater(row["loss_db_per_cm"], 0)
                self.assertLess(row["normalized_tm_boundary_residual"], 1e-9)
                self.assertIs(row["physics_qualified"], False)
            with self.assertRaises(FileExistsError):
                run_sourced_sweep([0.75], [1.35], target)

    def test_out_of_source_range_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "outside.json"
            with self.assertRaises(ValueError):
                run_sourced_sweep([2.0], [1.36], target)
            self.assertFalse(target.exists())

    def test_duplicate_grid_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                run_sourced_sweep([0.8, 0.8], [1.36], Path(tmp) / "dup.json")

if __name__ == "__main__":
    unittest.main()
