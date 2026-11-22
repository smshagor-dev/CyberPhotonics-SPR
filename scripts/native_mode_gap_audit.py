"""Fail-closed native eigenpair separation audit. This is NOT a PCF-SPR qualification."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile


def audit_native_mode_gaps(source):
    source = Path(source)
    with zipfile.ZipFile(source) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith("/") or ".." in PurePosixPath(name).parts for name in names
        ):
            raise ValueError("unsafe ZIP members")
        registry = json.loads(z.read("GRID_READINESS.json"))
        registered = registry["artifacts_sha256"]
        for name, digest in registered.items():
            if name not in names or hashlib.sha256(z.read(name)).hexdigest() != digest:
                raise ValueError("source checksum mismatch: " + name)
        for required in ("experiment_protocol.json", "raw_eigenmode_grid.csv"):
            if required not in registered:
                raise ValueError("unregistered physics input: " + required)
        protocol = json.loads(z.read("experiment_protocol.json"))
        ris = protocol["analyte_ris"]
        wavelengths = protocol["wavelengths_um"]
        modes = protocol["num_modes"]
        for grid in (ris, wavelengths):
            if not grid or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in grid
            ) or any(b <= a for a, b in zip(grid, grid[1:])):
                raise ValueError("invalid numerical grid")
        if isinstance(modes, bool) or not isinstance(modes, int) or modes < 2:
            raise ValueError("invalid native mode count")
        rows = csv.DictReader(io.StringIO(z.read("raw_eigenmode_grid.csv").decode("utf-8")))
        observed = {}
        for row in rows:
            i, j, k = (int(row[field]) for field in (
                "ri_index", "wavelength_index", "mode_index_native_untracked"
            ))
            key = (i, j, k)
            if key in observed or not (0 <= i < len(ris) and 0 <= j < len(wavelengths) and 0 <= k < modes):
                raise ValueError("duplicate or out-of-range eigenpair")
            if abs(float(row["analyte_ri"]) - ris[i]) > 1e-10 or abs(float(row["wavelength_um"]) - wavelengths[j]) > 1e-10:
                raise ValueError("inconsistent grid coordinates")
            re, im = float(row["n_eff_real"]), float(row["n_eff_imag"])
            if not math.isfinite(re) or not math.isfinite(im) or im < 0:
                raise ValueError("invalid complex eigenvalue")
            observed[key] = complex(re, im)
        expected = {(i, j, k) for i in range(len(ris)) for j in range(len(wavelengths)) for k in range(modes)}
        if set(observed) != expected:
            raise ValueError("incomplete native eigenpair grid")
        diagnostics = []
        for i, ri in enumerate(ris):
            for j, wl in enumerate(wavelengths):
                vals = [observed[i, j, k] for k in range(modes)]
                gaps = [abs(vals[a] - vals[b]) for a in range(modes) for b in range(a + 1, modes)]
                diagnostics.append({
                    "analyte_ri": ri, "wavelength_um": wl,
                    "minimum_complex_neff_gap": min(gaps),
                    "maximum_complex_neff_gap": max(gaps),
                    "physical_core_spp_identity_verified": False,
                })
    return {
        "source_archive_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "native_eigenpairs": len(observed), "diagnostics": diagnostics,
        "physics_qualified": False, "publication_ready": False, "comsol_verified": False,
        "limitations": "Untracked native eigenpairs; no authenticated Au, PML, core/SPP identity, spectral convergence or COMSOL verification.",
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = audit_native_mode_gaps(args.archive)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(report, f, sort_keys=True, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
