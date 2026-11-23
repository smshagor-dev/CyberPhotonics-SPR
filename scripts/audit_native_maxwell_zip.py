"""Read-only, fail-closed structural audit of a real native Maxwell artifact ZIP.

This does NOT validate PCF physics, PML, geometry, or mode identity.
Usage: python scripts/audit_native_maxwell_zip.py artifact.zip
"""
import csv
import hashlib
import io
import json
import sys
import zipfile


def audit(path):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive entries")
        manifest = json.loads(archive.read("GRID_READINESS.json"))
        mismatches = [
            name for name, expected in manifest["artifacts_sha256"].items()
            if name not in names
            or hashlib.sha256(archive.read(name)).hexdigest() != expected
        ]
        rows = list(csv.DictReader(io.StringIO(
            archive.read("raw_eigenmode_grid.csv").decode("utf-8")
        )))
        nonfinite = 0
        invalid_shapes = []
        field_archives = [n for n in names if n.endswith(".npz")]
        for name in field_archives:
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as fields:
                expected_keys = {"mode_0_E", "mode_0_H", "mode_1_E", "mode_1_H"}
                if set(fields.files) != expected_keys:
                    invalid_shapes.append(name)
                for key in fields.files:
                    arr = fields[key]
                    if arr.ndim != 1 or arr.size == 0 or not np.iscomplexobj(arr):
                        invalid_shapes.append(name + ":" + key)
                    nonfinite += int(arr.size - np.count_nonzero(np.isfinite(arr)))
        invalid_csv = []
        for index, row in enumerate(rows):
            for key in ("n_eff_real", "n_eff_imag", "wavelength_um", "analyte_ri"):
                if not np.isfinite(float(row[key])):
                    invalid_csv.append([index, key])
        return {
            "archive_sha256": hashlib.sha256(open(path, "rb").read()).hexdigest(),
            "checked_hashes": len(manifest["artifacts_sha256"]),
            "mismatched_hashes": mismatches,
            "raw_rows": len(rows),
            "field_archives": len(field_archives),
            "nonfinite_field_coefficients": nonfinite,
            "invalid_field_shapes": invalid_shapes,
            "invalid_csv": invalid_csv,
            "physics_qualified": False,
            "publication_ready": False,
            "comsol_verified": False,
            "note": "Byte/structure checks cannot establish Maxwell or sensor validity.",
        }


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2))
