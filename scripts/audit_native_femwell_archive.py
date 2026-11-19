"""Fail-closed, read-only audit of an actual Femwell native ZIP archive.

This checks numerical data integrity, not electromagnetic qualification.
Never convert its result into a publication-ready claim.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np


def audit(path: Path) -> dict:
    archive_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as z:
        if z.testzip() is not None:
            raise ValueError("ZIP CRC failure")
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive member names")
        required = {"GRID_READINESS.json", "experiment_protocol.json",
                    "topology.json", "raw_eigenmode_grid.csv",
                    "vector_mode_overlap_diagnostics.json"}
        if not required.issubset(names):
            raise ValueError("missing required native archive members")
        readiness = json.loads(z.read("GRID_READINESS.json"))
        if not readiness.get("completed"):
            raise ValueError("solver run incomplete")
        hashes = readiness["artifacts_sha256"]
        if set(hashes) != set(names) - {"GRID_READINESS.json"}:
            raise ValueError("manifest does not cover every native member")
        for name, expected in hashes.items():
            if hashlib.sha256(z.read(name)).hexdigest() != expected:
                raise ValueError(f"SHA256 mismatch: {name}")
        protocol = json.loads(z.read("experiment_protocol.json"))
        ris = readiness["analyte_ris"]
        wavelengths = readiness["wavelengths_um"]
        if (not isinstance(ris, list) or not isinstance(wavelengths, list)
                or not ris or not wavelengths
                or any(type(x) not in (int, float) or not math.isfinite(x) for x in ris + wavelengths)
                or any(x <= 0 for x in wavelengths)
                or len(set(ris)) != len(ris) or len(set(wavelengths)) != len(wavelengths)
                or any(b <= a for a, b in zip(wavelengths, wavelengths[1:]))
                or protocol.get("analyte_ris") != ris
                or protocol.get("wavelengths_um") != wavelengths
                or protocol.get("num_modes") != readiness["num_modes_requested"]):
            raise ValueError("invalid or inconsistent frozen RI/wavelength protocol")
        rows = list(csv.DictReader(io.StringIO(z.read("raw_eigenmode_grid.csv").decode())))
        grid = {}
        loss_diffs = []
        for row in rows:
            point = (int(row["ri_index"]), int(row["wavelength_index"]))
            mode = int(row["mode_index_native_untracked"])
            if (point[0] not in range(len(ris)) or point[1] not in range(len(wavelengths))
                    or float(row["analyte_ri"]) != ris[point[0]]
                    or float(row["wavelength_um"]) != wavelengths[point[1]]):
                raise ValueError("native eigenmode coordinates differ from frozen protocol")
            grid.setdefault(point, []).append(mode)
            lam = float(row["wavelength_um"])
            imag = float(row["n_eff_imag"])
            loss = float(row["raw_attenuation_db_per_cm_unqualified"])
            values = (lam, imag, loss, float(row["n_eff_real"]))
            if not all(math.isfinite(v) for v in values) or lam <= 0 or imag < 0 or loss < 0:
                raise ValueError("invalid native eigenmode values")
            calculated = (20 / math.log(10)) * (2 * math.pi / lam) * imag * 1e4
            loss_diffs.append(abs(calculated - loss))
        expected_grid = {(i, j) for i in range(len(readiness["analyte_ris"]))
                         for j in range(len(readiness["wavelengths_um"]))}
        if set(grid) != expected_grid:
            raise ValueError("missing/extra RI-wavelength samples")
        expected_modes = list(range(int(readiness["num_modes_requested"])))
        if any(sorted(modes) != expected_modes for modes in grid.values()):
            raise ValueError("missing/duplicate native modes")
        vector_files = sorted(n for n in names if n.endswith(".npz"))
        if len(vector_files) != len(expected_grid):
            raise ValueError("missing native vector-field archive")
        for i, j in expected_grid:
            name = f"raw_vector_eh_ri{i}_wl{j}.npz"
            if name not in vector_files:
                raise ValueError(f"missing vector field: {name}")
            with np.load(io.BytesIO(z.read(name)), allow_pickle=False) as arrays:
                required_keys = {f"mode_{m}_{field}" for m in expected_modes
                                 for field in ("E", "H")}
                if set(arrays.files) != required_keys:
                    raise ValueError(f"missing E/H arrays: {name}")
                field_sizes = set()
                for key in required_keys:
                    a = arrays[key]
                    field_sizes.add(a.size)
                    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
                        raise ValueError(f"invalid native field: {name}/{key}")
                if len(field_sizes) != 1:
                    raise ValueError(f"inconsistent E/H coefficient lengths: {name}")
        overlaps = json.loads(z.read("vector_mode_overlap_diagnostics.json"))
        if len(overlaps) != readiness["vector_overlap_matched_edges"]:
            raise ValueError("overlap edge count mismatch")
        if not all(math.isfinite(float(o["vector_eh_poynting_overlap_abs"]))
                   and 0 <= float(o["vector_eh_poynting_overlap_abs"]) <= 1 + 1e-9
                   for o in overlaps):
            raise ValueError("invalid overlap magnitude")
        if len(rows) != readiness["raw_eigenmode_rows"] or len(grid) != readiness["solver_calls_completed"]:
            raise ValueError("solver accounting mismatch")
        if max(loss_diffs) > 1e-8:
            raise ValueError("native loss formula mismatch")
        flags = ("scientific_submission_ready", "mode_tracking_physically_verified",
                 "mesh_convergence_verified", "pml_validated",
                 "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
                 "Au_reference_transcription_independently_verified")
        if any(readiness.get(flag) is not False for flag in flags):
            raise ValueError("unexpected physics qualification flag")
        return {"archive_sha256": archive_sha256, "manifest_sha256_verified": True,
                "zip_members": len(names), "solver_calls": len(grid),
                "raw_eigenmode_rows": len(rows), "native_eh_archives": len(vector_files),
                "overlap_edges": len(overlaps),
                "max_loss_formula_error_db_per_cm": max(loss_diffs),
                "native_integrity_verified": True, "physics_qualified": False,
                "publication_ready": False, "comsol_verified": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.archive)
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as output:
            output.write(rendered)
    else:
        print(rendered, end="")
