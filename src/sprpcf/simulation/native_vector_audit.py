"""Fail-closed audit of native, unqualified Femwell E/H archives.

This checks archive integrity and bookkeeping, NOT PML, mode physics or sensor
qualification. It never changes publication gates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sprpcf.simulation.attenuation_identity import audit_attenuation_rows


def audit_native_vector_grid(directory: str | Path) -> dict:
    root = Path(directory)
    report = json.loads((root / "GRID_READINESS.json").read_text(encoding="utf-8"))
    failures: list[str] = []
    hashes = report.get("artifacts_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        failures.append("missing native artifact hashes")
        hashes = {}
    for name, expected in hashes.items():
        if Path(name).name != name or not isinstance(expected, str) or len(expected) != 64:
            failures.append(f"invalid artifact manifest entry: {name}")
            continue
        path = root / name
        if not path.is_file():
            failures.append(f"missing artifact: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(f"checksum mismatch: {name}")

    if "raw_eigenmode_grid.csv" not in hashes:
        failures.append("unhashed eigenmode CSV")
    rows_path = root / "raw_eigenmode_grid.csv"
    rows = []
    if rows_path.is_file():
        with rows_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    else:
        failures.append("missing eigenmode CSV")
    wavelengths = report.get("wavelengths_um", [])
    ris = report.get("analyte_ris", [])
    modes = report.get("num_modes_requested", 0)
    if (not isinstance(wavelengths, list) or not isinstance(ris, list)
            or not isinstance(modes, int) or isinstance(modes, bool)
            or modes < 1 or not wavelengths or not ris
            or any(not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
                   for x in wavelengths + ris)
            or len(set(wavelengths)) != len(wavelengths)
            or len(set(ris)) != len(ris)):
        failures.append("invalid or empty wavelength/RI/mode grid")
        wavelengths, ris, modes = [], [], 0
    expected_count = len(wavelengths) * len(ris) * modes
    if not report.get("completed") or report.get("failures"):
        failures.append("solver reported incomplete or failed run")
    if len(rows) != expected_count or report.get("raw_eigenmode_rows") != expected_count:
        failures.append("incomplete or duplicated eigenmode grid")
    seen = set()
    for row in rows:
        try:
            ri, wi, mi = (int(row[k]) for k in (
                "ri_index", "wavelength_index", "mode_index_native_untracked"))
            key = (ri, wi, mi)
            if key in seen or not (0 <= ri < len(ris) and 0 <= wi < len(wavelengths)
                                   and 0 <= mi < modes):
                raise ValueError("duplicate or out-of-range mode index")
            seen.add(key)
            if (abs(float(row["wavelength_um"]) - wavelengths[wi]) > 1e-12
                    or abs(float(row["analyte_ri"]) - ris[ri]) > 1e-12):
                raise ValueError("grid coordinate mismatch")
            if not all(math.isfinite(float(row[k])) for k in (
                    "n_eff_real", "n_eff_imag", "gold_epsilon_real",
                    "gold_epsilon_imag", "raw_attenuation_db_per_cm_unqualified")):
                raise ValueError("nonfinite eigenmode value")
            if (float(row["n_eff_imag"]) < 0
                    or float(row["raw_attenuation_db_per_cm_unqualified"]) < 0):
                raise ValueError("negative attenuation or gain-mode index")
            if float(row["gold_epsilon_imag"]) <= 0:
                raise ValueError("nonpassive Au")
        except (KeyError, ValueError, IndexError, TypeError) as exc:
            failures.append(f"invalid CSV eigenmode row: {exc}")
    attenuation = audit_attenuation_rows(rows)
    if not attenuation["dimensional_identity_pass"]:
        for failure in attenuation["failures"]:
            failures.append(f"attenuation identity row {failure['row']}: {failure['reason']}")
    if len(seen) != expected_count:
        failures.append("missing distinct eigenmode grid keys")
    for ri in range(len(ris)):
        for wi in range(len(wavelengths)):
            name = f"raw_vector_eh_ri{ri}_wl{wi}.npz"
            if name not in hashes:
                failures.append(f"unhashed eigenfield archive: {name}")
                continue
            path = root / name
            if not path.is_file():
                continue
            try:
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != {
                            f"mode_{i}_{field}" for i in range(modes)
                            for field in ("E", "H")}:
                        raise ValueError("missing/extra E/H mode arrays")
                    for field in archive.files:
                        arr = archive[field]
                        if arr.size == 0 or not np.iscomplexobj(arr) or not np.isfinite(arr).all():
                            raise ValueError(f"invalid field: {field}")
            except (ValueError, OSError, KeyError) as exc:
                failures.append(f"invalid eigenfield archive {name}: {exc}")
    for flag in ("sensor_geometry_peer_reviewed_and_frozen", "pml_validated",
                 "mesh_convergence_verified", "mode_tracking_physically_verified",
                 "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
                 "scientific_submission_ready"):
        if report.get(flag) is not False:
            failures.append(f"unqualified evidence must not assert {flag}")
    return {
        "schema_version": 1,
        "audit": "native_femwell_vector_grid_integrity_not_physics_validation",
        "native_integrity_pass": not failures,
        "attenuation_dimensional_identity_pass": attenuation["dimensional_identity_pass"],
        "attenuation_rows_checked": attenuation["rows_checked"],
        "physics_qualified": False,
        "publication_ready": False,
        "comsol_verified": False,
        "rows_checked": len(rows),
        "expected_rows": expected_count,
        "field_archives_expected": len(ris) * len(wavelengths),
        "failures": failures,
    }
