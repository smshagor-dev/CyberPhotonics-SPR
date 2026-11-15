"""Fail-closed audit of native exploratory Femwell grid artifacts.

This validates archive integrity and numerical bookkeeping, NOT physical accuracy.
No synthetic/fixture evidence may pass a publication gate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


def audit_grid(directory: Path) -> dict:
    root = Path(directory)
    errors = []
    try:
        report = json.loads((root / "GRID_READINESS.json").read_text(encoding="utf-8"))
        protocol = json.loads((root / "experiment_protocol.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"passed": False, "errors": [f"missing/invalid manifest: {exc}"],
                "publication_qualified": False}
    if report.get("evidence_class") != "unqualified_solver_feasibility":
        errors.append("unexpected evidence class")
    if report.get("scientific_submission_ready") is not False:
        errors.append("publication gate must remain false")
    if report.get("completed") is not True or report.get("failures"):
        errors.append("incomplete solver campaign or recorded failures")
    hashes = report.get("artifacts_sha256")
    if not isinstance(hashes, dict) or not hashes:
        errors.append("missing artifact hashes")
        hashes = {}
    for name, digest in hashes.items():
        path = root / name
        if (not isinstance(name, str) or not name or Path(name).name != name
                or not isinstance(digest, str) or len(digest) != 64):
            errors.append(f"unsafe or invalid artifact record: {name!r}")
            continue
        if not path.is_file() or path.is_symlink():
            errors.append(f"missing or linked artifact: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            errors.append(f"SHA256 mismatch: {name}")
    required = {"pcf_frozen_exploratory_grid_mesh.msh", "topology.json",
                "experiment_protocol.json", "raw_eigenmode_grid.csv",
                "vector_mode_overlap_diagnostics.json"}
    if not required.issubset(hashes):
        errors.append("required native artifact missing from manifest")
    try:
        wavelengths = protocol["wavelengths_um"]
        ris = protocol["analyte_ris"]
        modes = protocol["num_modes"]
        if not isinstance(modes, int) or isinstance(modes, bool) or modes < 1:
            raise ValueError("invalid mode count")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in wavelengths + ris):
            raise ValueError("nonfinite grid coordinate")
        if len(wavelengths) < 2 or len(ris) < 2:
            raise ValueError("insufficient grid points")
        with (root / "raw_eigenmode_grid.csv").open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        expected = {(i, j, k) for i in range(len(ris))
                    for j in range(len(wavelengths)) for k in range(modes)}
        observed = set()
        for row in rows:
            i, j, k = (int(row[key]) for key in
                       ("ri_index", "wavelength_index", "mode_index_native_untracked"))
            if (i, j, k) not in expected or (i, j, k) in observed:
                raise ValueError("duplicate or unexpected grid eigenmode")
            observed.add((i, j, k))
            if (float(row["analyte_ri"]) != ris[i]
                    or float(row["wavelength_um"]) != wavelengths[j]):
                raise ValueError("grid coordinate differs from frozen protocol")
            for key in ("n_eff_real", "n_eff_imag", "gold_epsilon_real",
                        "gold_epsilon_imag", "silica_ri"):
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"nonfinite {key}")
        if observed != expected:
            raise ValueError(f"incomplete eigenmode grid: {len(observed)}/{len(expected)}")
        if report.get("solver_calls_completed") != len(ris) * len(wavelengths):
            raise ValueError("solver call count mismatch")
        if report.get("raw_eigenmode_rows") != len(expected):
            raise ValueError("eigenmode row count mismatch")
        for i in range(len(ris)):
            for j in range(len(wavelengths)):
                name = f"raw_vector_eh_ri{i}_wl{j}.npz"
                if name not in hashes:
                    raise ValueError(f"untracked native E/H field: {name}")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        errors.append(f"numerical bookkeeping: {exc}")
    return {"passed": not errors, "errors": errors,
            "publication_qualified": False,
            "audit_scope": "integrity_and_completeness_only_not_Maxwell_validation"}


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = audit_grid(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
