"""Fail-closed consistency audit for exploratory native Femwell grid archives.

This is an integrity check, NOT electromagnetic validation or COMSOL evidence.
"""
import csv
import hashlib
import json
import math
from pathlib import Path


def audit(directory):
    root = Path(directory)
    report = json.loads((root / "GRID_READINESS.json").read_text(encoding="utf-8"))
    errors = []
    hashes = report.get("artifacts_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        errors.append("missing artifact hash registry")
        hashes = {}
    for name, digest in hashes.items():
        if not isinstance(name, str) or Path(name).name != name or name in (".", ".."):
            errors.append("unsafe artifact filename")
            continue
        path = root / name
        if path.is_symlink():
            errors.append(f"symlink artifact forbidden: {name}")
        elif not path.is_file():
            errors.append(f"missing artifact: {name}")
        elif not isinstance(digest, str) or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"hash mismatch: {name}")

    for required in ("experiment_protocol.json", "raw_eigenmode_grid.csv",
                     "pcf_frozen_exploratory_grid_mesh.msh", "topology.json",
                     "vector_mode_overlap_diagnostics.json"):
        if required not in hashes:
            errors.append(f"unregistered required artifact: {required}")

    protocol_path = root / "experiment_protocol.json"
    rows_path = root / "raw_eigenmode_grid.csv"
    if not protocol_path.is_file() or not rows_path.is_file():
        errors.append("missing protocol or eigenmode grid")
    else:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        wavelengths = protocol.get("wavelengths_um", [])
        ris = protocol.get("analyte_ris", [])
        modes = protocol.get("num_modes")
        if (not isinstance(modes, int) or isinstance(modes, bool) or modes < 1
                or not isinstance(wavelengths, list) or not isinstance(ris, list)
                or not wavelengths or not ris
                or any(a >= b for values in (wavelengths, ris)
                       for a, b in zip(values, values[1:]))
                or not all(isinstance(x, (float, int)) and not isinstance(x, bool)
                           and math.isfinite(x) for x in wavelengths + ris)):
            errors.append("invalid experiment grid")
        else:
            with rows_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            expected = {(i, j, k) for i in range(len(ris))
                        for j in range(len(wavelengths)) for k in range(modes)}
            seen = set()
            for row in rows:
                try:
                    key = (int(row["ri_index"]), int(row["wavelength_index"]),
                           int(row["mode_index_native_untracked"]))
                    if key not in expected or key in seen:
                        errors.append("duplicate or unexpected eigenmode coordinate")
                    seen.add(key)
                    ri, wl = float(row["analyte_ri"]), float(row["wavelength_um"])
                    if (not math.isclose(ri, ris[key[0]], rel_tol=0, abs_tol=1e-12)
                            or not math.isclose(wl, wavelengths[key[1]], rel_tol=0, abs_tol=1e-12)):
                        errors.append("row protocol coordinate mismatch")
                    for field in ("n_eff_real", "n_eff_imag",
                                  "raw_attenuation_db_per_cm_unqualified"):
                        if not math.isfinite(float(row[field])):
                            errors.append("nonfinite eigenmode output")
                except (KeyError, ValueError, IndexError, TypeError):
                    errors.append("malformed eigenmode row")
            if seen != expected:
                errors.append("incomplete eigenmode coordinate coverage")
            for i in range(len(ris)):
                for j in range(len(wavelengths)):
                    name = f"raw_vector_eh_ri{i}_wl{j}.npz"
                    if name not in hashes:
                        errors.append(f"unregistered E/H archive: {name}")

    forbidden = ("sensor_geometry_peer_reviewed_and_frozen", "pml_validated",
                 "mesh_convergence_verified", "mode_tracking_physically_verified",
                 "resonance_fwhm_verified", "physical_sensor_sensitivity_verified",
                 "scientific_submission_ready")
    for flag in forbidden:
        if report.get(flag) is not False:
            errors.append(f"unsafe or absent qualification flag: {flag}")
    return {"integrity_passed": not errors, "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False,
            "errors": sorted(set(errors)), "registered_artifacts": len(hashes)}
