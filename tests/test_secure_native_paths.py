"""Adversarial tests for descriptor-relative native artifact access."""
import os
import tempfile
import unittest
from pathlib import Path
from research.secure_native_paths import open_native_file


@unittest.skipUnless(os.name == "posix", "POSIX descriptor traversal")
class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "dir").mkdir()
        (self.root / "dir" / "data").write_bytes(b"native")

    def tearDown(self):
        self.tmp.cleanup()

    def test_nested_file(self):
        fd = open_native_file(self.root, "dir/data")
        try:
            self.assertEqual(os.read(fd, 6), b"native")
        finally:
            os.close(fd)

    def test_symlinked_file(self):
        (self.root / "alias").symlink_to("dir/data")
        with self.assertRaises(OSError):
            open_native_file(self.root, "alias")

    def test_symlinked_parent(self):
        (self.root / "aliasdir").symlink_to("dir")
        with self.assertRaises(OSError):
            open_native_file(self.root, "aliasdir/data")

    def test_traversal(self):
        for path in ("../data", "dir/../data", "/etc/passwd", "dir//data", "dir/./data", "dir\\data"):
            with self.subTest(path=path), self.assertRaises(ValueError):
                open_native_file(self.root, path)

    def test_directory(self):
        with self.assertRaises(ValueError):
            open_native_file(self.root, "dir")

    def test_missing(self):
        with self.assertRaises(FileNotFoundError):
            open_native_file(self.root, "missing")

    def test_root_symlink(self):
        (self.root / "rootalias").symlink_to(self.root / "dir")
        with self.assertRaises(OSError):
            open_native_file(self.root / "rootalias", "data")


if __name__ == "__main__":
    unittest.main()
