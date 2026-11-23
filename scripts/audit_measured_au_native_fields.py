"""Byte-level source-specific audit of the unqualified measured-Au FEM ZIP.

This is an integrity and dimensional-consistency check, NOT PML/mesh convergence,
mode identity, sensor sensitivity, or COMSOL verification.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import zipfile

import numpy as np

from audit_native_fem_zip import audit_zip


def audit_native_fields(path):
    structural = audit_zip(path)
    errors = list(structural["errors"])
    field_count = 0
    row_count = 0
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("GRID_READINESS.json"))
        rows = list(csv.DictReader(io.StringIO(
            archive.read("raw_eigenmode_grid.csv").decode("utf-8-sig")
        )))
        expected = manifest.get("artifacts_sha256", {})
        for name, digest in expected.items():
            if name not in archive.namelist():
                errors.append("missing hashed native member: " + name)
            elif hashlib.sha256(archive.read(name)).hexdigest() != digest:
                errors.append("native SHA256 mismatch: " + name)
        if set(expected) != set(archive.namelist()) - {"GRID_READINESS.json"}:
            errors.append("native member hash coverage incomplete")
        for row in rows:
            row_count += 1
            try:
                ri = int(row["ri_index"])
                wi = int(row["wavelength_index"])
                mode = int(row["mode_index_native_untracked"])
                wavelength = float(row["wavelength_um"])
                neff_imag = float(row["n_eff_imag"])
                reported_loss = float(row["raw_attenuation_db_per_cm_unqualified"])
                expected_loss = 8.685889638065037 * 2 * math.pi * neff_imag / wavelength * 1e4
                if not all(map(math.isfinite, (wavelength, neff_imag, reported_loss, expected_loss))):
                    raise ValueError("nonfinite numerical value")
                if wavelength <= 0 or neff_imag < 0 or not math.isclose(
                    expected_loss, reported_loss, rel_tol=1e-11, abs_tol=1e-12
                ):
                    raise ValueError("passivity or attenuation identity failure")
                if not (0 <= ri < 2 and 0 <= wi < 8 and 0 <= mode < 2):
                    raise ValueError("out-of-grid mode index")
            except (KeyError, ValueError, OverflowError, ZeroDivisionError) as exc:
                errors.append(f"row {row_count}: {exc}")
        for ri in range(2):
            for wi in range(8):
                name = f"raw_vector_eh_ri{ri}_wl{wi}.npz"
                if name not in archive.namelist():
                    errors.append("missing field archive: " + name)
                    continue
                with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as fields:
                    if set(fields.files) != {f"mode_{m}_{field}" for m in range(2) for field in ("E", "H")}:
                        errors.append("field key mismatch: " + name)
                    for key in fields.files:
                        field_count += 1
                        values = fields[key]
                        if values.ndim != 1 or values.size == 0 or not np.iscomplexobj(values):
                            errors.append("invalid vector field dimensions: " + name + ":" + key)
                        elif not np.isfinite(values).all() or np.linalg.norm(values) <= 0:
                            errors.append("nonfinite or zero vector field: " + name + ":" + key)
    return {
        "passed": not errors, "errors": errors, "rows_checked": row_count,
        "field_arrays_checked": field_count, "native_hashes_checked": len(expected),
        "source": "github_actions_measured_au_exploratory_fem",
        "physics_qualified": False, "publication_ready": False, "comsol_verified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive")
    args = parser.parse_args()
    result = audit_native_fields(args.archive)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
