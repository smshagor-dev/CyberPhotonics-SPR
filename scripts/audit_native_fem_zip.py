"""Structural ZIP audit of exploratory native FEM artifacts; NOT physics validation."""
import argparse
import csv
import hashlib
import io
import json
import re
import zipfile
from pathlib import PurePosixPath

FALSE_FLAGS = ("scientific_submission_ready", "pml_validated", "mode_tracking_physically_verified", "resonance_fwhm_verified", "physical_sensor_sensitivity_verified")

def audit_zip(path, ri=2, wavelengths=8, modes=2):
    errors = []
    with zipfile.ZipFile(path) as archive:
        entries = [i for i in archive.infolist() if not i.is_dir()]
        names = [i.filename for i in entries]
        if len(names) != len(set(names)):
            errors.append("duplicate members")
        if any(n.startswith("/") or ".." in PurePosixPath(n).parts or "\\" in n for n in names):
            errors.append("unsafe member path")
        if any(i.flag_bits & 1 for i in entries):
            errors.append("encrypted member")
        def read_one(name):
            found = [n for n in names if n == name or n.endswith("/" + name)]
            if len(found) != 1:
                errors.append("expected one " + name)
                return None
            return archive.read(found[0])
        readiness = read_one("GRID_READINESS.json")
        csv_bytes = read_one("raw_eigenmode_grid.csv")
        if readiness is None or csv_bytes is None:
            return {"passed": False, "errors": errors}
        try:
            status = json.loads(readiness)
            rows = list(csv.DictReader(io.StringIO(csv_bytes.decode("utf-8-sig"))))
        except (ValueError, UnicodeDecodeError, csv.Error):
            return {"passed": False, "errors": errors + ["invalid JSON/CSV"]}
        for flag in FALSE_FLAGS:
            if status.get(flag) is not False:
                errors.append(flag + " must be explicitly false")
        calls = ri * wavelengths
        if status.get("completed") is not True or status.get("solver_calls_completed") != calls:
            errors.append("solver call count mismatch")
        if status.get("raw_eigenmode_rows") != calls * modes or len(rows) != calls * modes:
            errors.append("mode row count mismatch")
        found = set()
        for name in names:
            match = re.search(r"(?:^|/)raw_vector_eh_ri(\d+)_wl(\d+)\.npz$", name)
            if match:
                found.add((int(match.group(1)), int(match.group(2))))
            if name.endswith(".npz"):
                try:
                    with zipfile.ZipFile(io.BytesIO(archive.read(name))) as nested:
                        if not nested.namelist() or nested.testzip() is not None:
                            errors.append("invalid NPZ: " + name)
                except (zipfile.BadZipFile, RuntimeError):
                    errors.append("invalid NPZ: " + name)
        expected = {(i, j) for i in range(ri) for j in range(wavelengths)}
        if found != expected:
            errors.append("E/H NPZ coverage mismatch")
        if archive.testzip() is not None:
            errors.append("ZIP CRC mismatch")
    with open(path, "rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"passed": not errors, "errors": errors, "sha256": digest, "rows": len(rows),
            "field_archives": len(found), "qualification": "unqualified_structural_integrity_only"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip")
    parser.add_argument("--ri", type=int, default=2)
    parser.add_argument("--wavelengths", type=int, default=8)
    parser.add_argument("--modes", type=int, default=2)
    args = parser.parse_args()
    result = audit_zip(args.zip, args.ri, args.wavelengths, args.modes)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
