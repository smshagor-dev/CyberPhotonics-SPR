"""Read-only provenance and coverage audit for genuine native FEM ZIP archives.

This does NOT qualify optical sensor performance, COMSOL, or publication evidence.
Run: python scripts/audit_native_fem_archive.py <artifact.zip>
"""
import csv
import hashlib
import io
import json
import math
import sys
import zipfile
from collections import Counter
from pathlib import Path


def audit(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            n.startswith("/") or ".." in Path(n).parts for n in names
        ):
            raise ValueError("unsafe or duplicate archive paths")
        readiness = json.loads(archive.read("GRID_READINESS.json"))
        hashes = readiness["artifacts_sha256"]
        if set(hashes) != set(names) - {"GRID_READINESS.json"}:
            raise ValueError("unregistered or missing native artifacts")
        for name, expected in hashes.items():
            digest = hashlib.sha256()
            with archive.open(name) as source:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != expected:
                raise ValueError(f"native artifact checksum mismatch: {name}")
        protocol = json.loads(archive.read("experiment_protocol.json"))
        rows = list(csv.DictReader(io.StringIO(
            archive.read("raw_eigenmode_grid.csv").decode("utf-8")
        )))
        ri = len(protocol["analyte_ris"])
        wl = len(protocol["wavelengths_um"])
        modes = protocol["num_modes"]
        expected = {(i, j, k) for i in range(ri)
                    for j in range(wl) for k in range(modes)}
        actual = Counter((int(r["ri_index"]), int(r["wavelength_index"]),
                          int(r["mode_index_native_untracked"])) for r in rows)
        if set(actual) != expected or any(count != 1 for count in actual.values()):
            raise ValueError("missing or duplicated eigenmode grid cell")
        for i in range(ri):
            for j in range(wl):
                name = f"raw_vector_eh_ri{i}_wl{j}.npz"
                if name not in hashes:
                    raise ValueError(f"unregistered native E/H field: {name}")
        loss = []
        for row in rows:
            wavelength = float(row["wavelength_um"])
            imaginary = float(row["n_eff_imag"])
            if not math.isfinite(wavelength) or wavelength <= 0 or not math.isfinite(imaginary):
                raise ValueError("nonfinite wavelength or complex effective index")
            loss.append(8.685889638065037 * 2 * math.pi * imaginary
                        / wavelength * 1e4)
        overlaps = json.loads(archive.read("vector_mode_overlap_diagnostics.json"))
        if len(overlaps) != modes * ((wl - 1) * ri + (ri - 1) * wl):
            raise ValueError("incomplete mode overlap edges")
        return {
            "archive_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "registered_files_verified": len(hashes),
            "eigenpairs_verified": len(rows),
            "mode_overlap_edges": len(overlaps),
            "raw_attenuation_range_db_per_cm_unqualified": [min(loss), max(loss)],
            "gold_source": protocol["Au"],
            "physics_qualified": False,
            "publication_ready": False,
            "comsol_verified": False,
            "limitations": [
                "exploratory geometry", "finite-box without PML",
                "no independently verified core/SPP branch",
                "raw attenuation is not validated confinement loss",
            ],
        }


if __name__ == "__main__":
    print(json.dumps(audit(sys.argv[1]), indent=2, sort_keys=True))
