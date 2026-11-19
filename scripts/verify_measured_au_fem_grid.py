"""Fail-closed fingerprint audit of a native Johnson-Christy-Au Femwell grid.

Checks real numerical output consistency, not physical validity or COMSOL equivalence.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from zipfile import BadZipFile
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.simulation.materials_reference import au_johnson_christy_permittivity, au_reference_manifest
from audit_native_vector_grid import audit_grid


def verify_measured_au_grid(root: Path) -> dict:
    root = Path(root)
    errors = []
    base = audit_grid(root)
    if not base["passed"]:
        errors.extend(base["errors"])
    try:
        protocol = json.loads((root / "experiment_protocol.json").read_text(encoding="utf-8"))
        report = json.loads((root / "GRID_READINESS.json").read_text(encoding="utf-8"))
        source = au_reference_manifest()
        if protocol.get("Au") != "Johnson_Christy_1972_measured_bulk_reference":
            errors.append("not a Johnson-Christy Au solver campaign")
        if protocol.get("Au_source_manifest") != source:
            errors.append("Au source manifest differs from installed reference data")
        if report.get("Au_reference_dataset_selected") is not True:
            errors.append("Johnson-Christy reference selection was not recorded")
        if report.get("Au_material_measured_data_provenance_verified") is not False:
            errors.append("unverified Au transcription must not claim verified provenance")
        if report.get("Au_reference_transcription_independently_verified") is not False:
            errors.append("independent Au transcription verification must remain false")
        if report.get("scientific_submission_ready") is not False:
            errors.append("scientific readiness must remain false")
        if report.get("pml_validated") is not False or report.get("mode_tracking_physically_verified") is not False:
            errors.append("unqualified physics flags were unexpectedly promoted")
        with (root / "raw_eigenmode_grid.csv").open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        if not rows:
            errors.append("no native eigenmode records")
        groups = {}
        for line, row in enumerate(rows, start=2):
            wl = float(row["wavelength_um"])
            pair = (int(row["ri_index"]), int(row["wavelength_index"]))
            groups.setdefault(pair, []).append(row)
            eps = complex(au_johnson_christy_permittivity(wl))
            got = complex(float(row["gold_epsilon_real"]), float(row["gold_epsilon_imag"]))
            if not (math.isclose(got.real, eps.real, rel_tol=1e-12, abs_tol=1e-12)
                    and math.isclose(got.imag, eps.imag, rel_tol=1e-12, abs_tol=1e-12)):
                errors.append(f"row {line} Au epsilon does not reproduce source material")
            if not (got.imag > 0 and math.isfinite(got.real) and math.isfinite(got.imag)):
                errors.append(f"row {line} Au epsilon is not finite/passive")
        for (ri_index, wl_index), group in groups.items():
            native = root / f"raw_vector_eh_ri{ri_index}_wl{wl_index}.npz"
            with np.load(native, allow_pickle=False) as archive:
                keys = {f"mode_{int(row['mode_index_native_untracked'])}_{field}"
                        for row in group for field in ("E", "H")}
                if set(archive.files) != keys:
                    errors.append(f"{native.name}: unexpected/missing modal E/H keys")
                    continue
                for row in group:
                    idx = int(row["mode_index_native_untracked"])
                    for field in ("E", "H"):
                        arr = archive[f"mode_{idx}_{field}"]
                        expected_dofs = int(row[f"vector_field_{field.lower()}_dofs"])
                        if (arr.size != expected_dofs or arr.size == 0 or not np.issubdtype(arr.dtype, np.number)
                                or not np.isfinite(arr).all() or not np.any(arr != 0)):
                            errors.append(f"{native.name}: invalid mode {idx} {field} archive")
    except (OSError, ValueError, TypeError, KeyError, OverflowError, BadZipFile) as exc:
        errors.append(f"measured Au check incomplete: {exc}")
    return {
        "passed": not errors,
        "errors": sorted(set(errors)),
        "source": "Johnson-Christy 1972 measured bulk Au reference",
        "scope": "native archive integrity + material fingerprint, not independent Maxwell validation",
        "scientific_submission_ready": False,
        "publication_qualified": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify_measured_au_grid(args.directory)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
