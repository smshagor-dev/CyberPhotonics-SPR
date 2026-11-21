"""Fail-closed spectral gate for *unqualified* native FEM data; never infer sensor metrics."""
import csv
import hashlib
import io
import json
import math
import pathlib
import sys
import zipfile


def audit(path):
    path = pathlib.Path(path)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(
            name.startswith("/") or "\\" in name or ".." in pathlib.PurePosixPath(name).parts
            for name in names
        ):
            raise ValueError("unsafe archive paths")
        readiness = json.loads(archive.read("GRID_READINESS.json"))
        protocol = json.loads(archive.read("experiment_protocol.json"))
        hashes = readiness["artifacts_sha256"]
        if set(hashes) != set(names) - {"GRID_READINESS.json"}:
            raise ValueError("unregistered or missing artifact")
        for flag in ("scientific_submission_ready", "pml_validated",
                     "mode_tracking_physically_verified", "resonance_fwhm_verified",
                     "physical_sensor_sensitivity_verified"):
            if readiness.get(flag) is not False:
                raise ValueError("unsafe or absent nonpublication flag: " + flag)
        ri, wl, modes = protocol["analyte_ris"], protocol["wavelengths_um"], protocol["num_modes"]
        if (type(modes) is not int or not 1 <= modes <= 100 or not ri or len(wl) < 3
            or any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in ri + wl)
            or any(b <= a for grid in (ri, wl) for a, b in zip(grid, grid[1:]))):
            raise ValueError("invalid RI/wavelength/mode grid")
        for name, digest in hashes.items():
            if hashlib.sha256(archive.read(name)).hexdigest() != digest:
                raise ValueError("SHA256 mismatch: " + name)
        rows = list(csv.DictReader(io.StringIO(archive.read("raw_eigenmode_grid.csv").decode())))
        expected = {(i, j, k) for i in range(len(ri)) for j in range(len(wl)) for k in range(modes)}
        found, losses = set(), {}
        for row in rows:
            key = tuple(int(row[col]) for col in
                        ("ri_index", "wavelength_index", "mode_index_native_untracked"))
            if key in found or key not in expected:
                raise ValueError("duplicate or unexpected eigenpair")
            found.add(key)
            i, j, k = key
            if (abs(float(row["analyte_ri"]) - ri[i]) > 1e-10
                or abs(float(row["wavelength_um"]) - wl[j]) > 1e-10):
                raise ValueError("coordinate mismatch")
            real, imag = float(row["n_eff_real"]), float(row["n_eff_imag"])
            if not math.isfinite(real) or not math.isfinite(imag) or imag < 0:
                raise ValueError("invalid passive eigenvalue")
            loss = 8.686 * (2 * math.pi / (wl[j] * 1e-4)) * imag
            if not math.isfinite(loss):
                raise ValueError("invalid attenuation")
            losses[key] = loss
        if found != expected:
            raise ValueError("incomplete eigenpair grid")
        spectra = []
        for i, n in enumerate(ri):
            for k in range(modes):
                values = [losses[i, j, k] for j in range(len(wl))]
                sampled_max = any(values[j] > values[j-1] and values[j] > values[j+1]
                                  for j in range(1, len(wl)-1))
                spectra.append({"analyte_ri": n, "native_mode_index_untracked": k,
                                "loss_db_cm": values, "internal_sampled_maximum": sampled_max,
                                "resonance_verified": False, "fwhm_verified": False,
                                "sensitivity_verified": False})
        return {"source_archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "eigenpairs": len(rows), "wavelength_um": wl, "analyte_ri": ri,
                "spectra": spectra, "reasons": [
                    "sparse spectrum cannot establish resonance/FWHM",
                    "native mode labels are not physically core/SPP tracked",
                    "no verified PML, independently sourced Au, frozen topology or solver cross-check"],
                "physics_qualified": False, "publication_ready": False,
                "comsol_verified": False}


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: native_spectral_gate.py ARCHIVE NEW_REPORT.json")
    result = audit(sys.argv[1])
    with pathlib.Path(sys.argv[2]).open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True)
