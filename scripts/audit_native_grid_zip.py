"""Independent read-only integrity audit of a native Femwell RI/wavelength ZIP.

Usage: python scripts/audit_native_grid_zip.py artifact.zip
Requires numpy. Checks integrity and finite native arrays, NOT Maxwell accuracy.
"""
import csv
import hashlib
import io
import json
import math
import sys
import zipfile

import numpy as np


def audit(path):
    errors = []
    with zipfile.ZipFile(path) as archive:
        names = set(archive.namelist())
        if archive.testzip() is not None:
            errors.append("ZIP CRC failure")
        readiness = json.loads(archive.read("GRID_READINESS.json"))
        protocol = json.loads(archive.read("experiment_protocol.json"))
        hashes = readiness["artifacts_sha256"]
        for name, expected in hashes.items():
            if name not in names or hashlib.sha256(archive.read(name)).hexdigest() != expected:
                errors.append("missing or modified: " + name)
        rows = list(csv.DictReader(io.StringIO(archive.read("raw_eigenmode_grid.csv").decode())))
        ris = protocol["analyte_ris"]
        wavelengths = protocol["wavelengths_um"]
        modes = protocol["num_modes"]
        expected = {(i, j, k) for i in range(len(ris)) for j in range(len(wavelengths)) for k in range(modes)}
        actual = set()
        for row in rows:
            key = tuple(int(row[x]) for x in ("ri_index", "wavelength_index", "mode_index_native_untracked"))
            if key in actual:
                errors.append("duplicate native mode: " + str(key))
            actual.add(key)
            if key not in expected:
                errors.append("unexpected native mode: " + str(key))
                continue
            if float(row["analyte_ri"]) != ris[key[0]] or float(row["wavelength_um"]) != wavelengths[key[1]]:
                errors.append("grid coordinate mismatch: " + str(key))
            for field in ("n_eff_real", "n_eff_imag", "gold_epsilon_real", "gold_epsilon_imag"):
                if not math.isfinite(float(row[field])):
                    errors.append("nonfinite " + field + ": " + str(key))
        if actual != expected or len(rows) != len(expected):
            errors.append("incomplete native mode grid")
        field_archives = 0
        for i in range(len(ris)):
            for j in range(len(wavelengths)):
                name = f"raw_vector_eh_ri{i}_wl{j}.npz"
                if name not in names:
                    errors.append("missing native fields: " + name)
                    continue
                field_archives += 1
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as fields:
                    if not fields.files:
                        errors.append("empty native fields: " + name)
                    for key in fields.files:
                        if not np.all(np.isfinite(fields[key])):
                            errors.append("nonfinite native fields: " + name + ":" + key)
        forbidden = ("mesh_convergence_verified", "mode_tracking_physically_verified",
                     "physical_sensor_sensitivity_verified", "pml_validated",
                     "resonance_fwhm_verified", "scientific_submission_ready",
                     "sensor_geometry_peer_reviewed_and_frozen")
        for flag in forbidden:
            if readiness.get(flag) is not False:
                errors.append("unqualified readiness flag: " + flag)
        if readiness.get("solver_calls_completed") != len(ris) * len(wavelengths):
            errors.append("solver-call count mismatch")
        return {"pass": not errors, "errors": errors, "manifest_hashes_checked": len(hashes),
                "native_rows": len(rows), "native_field_archives": field_archives,
                "evidence_class": "archive_integrity_only_not_physics_validation"}


if __name__ == "__main__":
    result = audit(sys.argv[1])
    print(json.dumps(result, indent=2, sort_keys=True))
    sys.exit(0 if result["pass"] else 1)
