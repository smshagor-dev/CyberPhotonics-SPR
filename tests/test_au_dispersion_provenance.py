"""Constructed constants test software only; never physics evidence."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.au_dispersion_provenance import load_au_table, epsilon_at_nm

CSV = b"wavelength_nm,n,k\n700,0.15,4.0\n800,0.16,4.5\n900,0.17,5.0\n"


class TestAuDispersion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "au.csv"
        self.path.write_bytes(CSV)
        self.sha = hashlib.sha256(CSV).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def load(self):
        return load_au_table(self.path, expected_sha256=self.sha,
                             source_doi="10.0000/constructed-test",
                             source_table="CONSTRUCTED-TEST")

    def test_interpolation_and_flags(self):
        table = self.load()
        self.assertAlmostEqual(epsilon_at_nm(table, 750).imag, 2 * 0.155 * 4.25)
        self.assertFalse(table["physics_qualified"])
        self.assertFalse(table["source_verified"])

    def test_no_extrapolation(self):
        with self.assertRaises(ValueError):
            epsilon_at_nm(self.load(), 699)

    def test_hash_tamper(self):
        self.path.write_bytes(CSV + b"\n")
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_wavelength(self):
        raw = CSV.replace(b"900,0.17", b"800,0.17")
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError):
            load_au_table(self.path, expected_sha256=hashlib.sha256(raw).hexdigest(),
                          source_doi="10.0000/test", source_table="fixture")

    def test_missing_source(self):
        with self.assertRaises(ValueError):
            load_au_table(self.path, expected_sha256=self.sha,
                          source_doi="", source_table="fixture")


if __name__ == "__main__":
    unittest.main()
