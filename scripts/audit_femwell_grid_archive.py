"""Fail-closed archive integrity audit; not a physics validation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def audit(root):
    root = Path(root)
    errors = []
    try:
        report = json.loads((root / "GRID_READINESS.json").read_text())
    except (OSError, ValueError) as exc:
        return {"integrity_pass": False, "errors": [str(exc)], "scientific_submission_ready": False}
    ris, wavelengths = report.get("analyte_ris", []), report.get("wavelengths_um", [])
    n = report.get("num_modes_requested", 0)
    planned = len(ris) * len(wavelengths)
    if not planned or type(n) is not int or n < 1:
        errors.append("invalid grid dimensions")
        n = 0
    if report.get("solver_calls_completed") != planned or report.get("raw_eigenmode_rows") != planned * n:
        errors.append("incomplete solver calls or mode rows")
    if report.get("completed") is not True or report.get("failures"):
        errors.append("experiment incomplete or failed")
    if report.get("scientific_submission_ready") is not False:
        errors.append("unsafe publication flag")
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict):
        hashes = {}
        errors.append("missing hash manifest")
    required = {"topology.json", "experiment_protocol.json",
                "pcf_frozen_exploratory_grid_mesh.msh", "raw_eigenmode_grid.csv",
                "vector_mode_overlap_diagnostics.json"}
    required |= {f"raw_vector_eh_ri{i}_wl{j}.npz"
                 for i in range(len(ris)) for j in range(len(wavelengths))}
    for name in sorted(required - hashes.keys()):
        errors.append(f"required file not hashed: {name}")
    for name, digest in hashes.items():
        if Path(name).name != name or not isinstance(digest, str) or len(digest) != 64:
            errors.append(f"invalid artifact entry: {name}")
            continue
        path = root / name
        if not path.is_file():
            errors.append(f"missing artifact: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"hash mismatch: {name}")
    try:
        with (root / "raw_eigenmode_grid.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        observed = [(int(row["ri_index"]), int(row["wavelength_index"]),
                     int(row["mode_index_native_untracked"])) for row in rows]
        expected = {(i, j, k) for i in range(len(ris))
                    for j in range(len(wavelengths)) for k in range(n)}
        if len(observed) != len(expected) or set(observed) != expected:
            errors.append("missing, duplicate or unexpected mode tuples")
    except (OSError, ValueError, KeyError) as exc:
        errors.append(f"cannot inspect eigenmode tuples: {exc}")
    return {"integrity_pass": not errors, "errors": errors,
            "planned_solver_calls": planned, "scientific_submission_ready": False,
            "physics_validated": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.archive)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["integrity_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
