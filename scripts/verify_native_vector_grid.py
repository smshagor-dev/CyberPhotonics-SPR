"""Fail-closed native Femwell grid integrity gate; never qualifies physics."""
import csv
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    root = Path(root)
    errors = []
    try:
        report = json.loads((root / "GRID_READINESS.json").read_text())
        protocol = json.loads((root / "experiment_protocol.json").read_text())
        with (root / "raw_eigenmode_grid.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, ValueError, csv.Error) as exc:
        return {"integrity_verified": False, "physics_qualified": False,
                "publication_ready": False, "comsol_verified": False,
                "errors": [f"input: {exc}"]}
    if not isinstance(protocol, dict) or not isinstance(report, dict):
        return {"integrity_verified": False, "physics_qualified": False,
                "publication_ready": False, "comsol_verified": False,
                "errors": ["invalid JSON document shape"]}
    wl, ri = protocol.get("wavelengths_um", []), protocol.get("analyte_ris", [])
    if not isinstance(wl, list) or not isinstance(ri, list):
        wl, ri = [], []
        errors.append("invalid protocol grid shape")
    for name, values in (("wavelengths_um", wl), ("analyte_ris", ri)):
        if not all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in values):
            errors.append(f"invalid {name} values")
        if len(set(map(str, values))) != len(values):
            errors.append(f"duplicate {name} values")
    modes = protocol.get("num_modes")
    expected = len(wl) * len(ri)
    if type(modes) is not int or modes < 1 or len(wl) < 2 or len(ri) < 2:
        errors.append("invalid protocol dimensions")
    if report.get("solver_calls_completed") != expected or report.get("completed") is not True:
        errors.append("incomplete solver campaign")
    if type(modes) is not int or modes < 1 or len(rows) != expected * modes:
        errors.append("wrong eigenmode row count")
    identities = set()
    for row in rows:
        try:
            if type(modes) is not int or modes < 1:
                raise ValueError("invalid mode count")
            raw_indices = [row[x] for x in ("ri_index", "wavelength_index", "mode_index_native_untracked")]
            if any(not v.isascii() or not v.isdecimal() for v in raw_indices):
                raise ValueError("noncanonical integer index")
            i, j, k = map(int, raw_indices)
            if not (0 <= i < len(ri) and 0 <= j < len(wl) and 0 <= k < modes):
                raise ValueError("index out of range")
            row_ri, row_wl = float(row["analyte_ri"]), float(row["wavelength_um"])
            if not math.isfinite(row_ri) or not math.isfinite(row_wl):
                raise ValueError("nonfinite row coordinate")
            if row_ri != float(ri[i]) or row_wl != float(wl[j]):
                raise ValueError("protocol mismatch")
            identities.add((i, j, k))
        except (ValueError, KeyError, TypeError):
            errors.append("invalid row identity")
    if len(identities) != len(rows):
        errors.append("duplicate mode identities")
    required = {"pcf_frozen_exploratory_grid_mesh.msh", "topology.json",
                "experiment_protocol.json", "raw_eigenmode_grid.csv",
                "vector_mode_overlap_diagnostics.json"}
    required.update(f"raw_vector_eh_ri{i}_wl{j}.npz"
                    for i in range(len(ri)) for j in range(len(wl)))
    digests = report.get("artifacts_sha256", {})
    if not isinstance(digests, dict):
        errors.append("invalid artifact digest mapping")
        digests = {}
    for name in sorted(required):
        if name not in digests:
            errors.append(f"missing digest: {name}")
        elif ((root / name).is_symlink() or not (root / name).is_file()
              or not isinstance(digests[name], str)
              or len(digests[name]) != 64
              or any(c not in "0123456789abcdef" for c in digests[name])
              or hashlib.sha256((root / name).read_bytes()).hexdigest() != digests[name]):
            errors.append(f"file absent or hash mismatch: {name}")
    if any(report.get(flag) is True for flag in ("physics_qualified", "publication_ready", "comsol_verified")):
        errors.append("unsafe publication or COMSOL claim")
    if report.get("scientific_submission_ready") is not False:
        errors.append("unsafe readiness flag")
    return {"integrity_verified": not errors, "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False,
            "expected_solver_calls": expected, "observed_rows": len(rows),
            "errors": errors}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    result = audit(parser.parse_args().directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["integrity_verified"] else 2)
