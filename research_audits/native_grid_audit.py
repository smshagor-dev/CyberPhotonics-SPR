"""Fail-closed integrity audit of exploratory Femwell wavelength/RI eigenmode grids.

This does not establish PML convergence, core/SPP mode identity, or sensor metrics.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


def audit_grid(folder):
    folder = Path(folder)
    report = json.loads((folder / "GRID_READINESS.json").read_text())
    failures = []
    wavelengths = report.get("wavelengths_um", [])
    ris = report.get("analyte_ris", [])
    modes = report.get("num_modes_requested", 0)
    if report.get("experiment") != "unqualified_femwell_full_vector_wavelength_ri_grid":
        failures.append("unexpected experiment identity")
    if (not report.get("completed") or report.get("failures")
            or report.get("solver_calls_completed") != len(wavelengths) * len(ris)):
        failures.append("incomplete solver campaign")
    if not wavelengths or not ris or not isinstance(modes, int) or modes <= 0:
        failures.append("invalid grid dimensions")
    artifacts = report.get("artifacts_sha256", {})
    for name, expected in artifacts.items():
        if (Path(name).name != name or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            failures.append("invalid artifact manifest entry")
            continue
        path = folder / name
        if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            failures.append("artifact missing or hash mismatch: " + name)
    if not artifacts or "raw_eigenmode_grid.csv" not in artifacts:
        failures.append("missing native eigenvalue manifest")
    rows = []
    if (folder / "raw_eigenmode_grid.csv").is_file():
        with (folder / "raw_eigenmode_grid.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
    seen = set()
    loss_rows = []
    for row in rows:
        try:
            ri_i, wl_i, mode_i = (int(row[k]) for k in
                ("ri_index", "wavelength_index", "mode_index_native_untracked"))
            wl, ri = float(row["wavelength_um"]), float(row["analyte_ri"])
            real, imag = float(row["n_eff_real"]), float(row["n_eff_imag"])
            key = (ri_i, wl_i, mode_i)
            if key in seen or not (0 <= ri_i < len(ris) and 0 <= wl_i < len(wavelengths) and 0 <= mode_i < modes):
                raise ValueError("duplicate or invalid grid coordinate")
            seen.add(key)
            if not all(map(math.isfinite, (wl, ri, real, imag))) or imag < 0 or wl <= 0:
                raise ValueError("non-passive or nonfinite eigenmode")
            if abs(wl - float(wavelengths[wl_i])) > 1e-10 or abs(ri - float(ris[ri_i])) > 1e-10:
                raise ValueError("grid coordinate inconsistent with protocol")
            loss_rows.append({"ri_index": ri_i, "wavelength_index": wl_i,
                "mode_index_native_untracked": mode_i,
                "attenuation_db_per_cm_unqualified": 8.685889638 * 4 * math.pi * imag / wl * 1e4})
        except (ValueError, TypeError, KeyError, IndexError) as exc:
            failures.append("invalid eigenmode row: " + str(exc))
    if len(seen) != len(ris) * len(wavelengths) * modes:
        failures.append("missing eigenmode grid cells")
    return {"schema_version": 1, "integrity_passed": not failures,
        "failures": failures, "loss_rows": loss_rows if not failures else [],
        "loss_is_physical_sensor_evidence": False, "mode_tracking_verified": False,
        "pml_validated": False, "physics_qualified": False,
        "publication_ready": False, "resonance_fwhm_sensitivity_reportable": False}
