"""Source-traceable optical-constant interpolation; NOT a PCF Maxwell solver.

CSV columns: wavelength_nm,n,k.  n+ik convention exp(-i omega t).
Never extrapolate optical constants or treat a fitted curve as measured data.
"""
import csv
import hashlib
import math
from pathlib import Path


def load_au_table(csv_path, *, expected_sha256, source_doi, source_table):
    if not isinstance(source_doi, str) or not source_doi.startswith("10.") or not source_table:
        raise ValueError("independent source DOI and table identifier required")
    path = Path(csv_path)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("optical constants SHA256 mismatch")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    if len(rows) < 3:
        raise ValueError("at least three independently tabulated wavelengths required")
    points = []
    for row in rows:
        try:
            w, n, k = (float(row[key]) for key in ("wavelength_nm", "n", "k"))
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError("invalid n,k optical constants table") from exc
        if not all(map(math.isfinite, (w, n, k))) or w <= 0 or n <= 0 or k < 0:
            raise ValueError("nonphysical optical constant")
        if points and w <= points[-1][0]:
            raise ValueError("wavelengths must be strictly increasing and unique")
        points.append((w, n, k))
    return {"points": tuple(points), "sha256": expected_sha256,
            "source_doi": source_doi, "source_table": source_table,
            "source_verified": False, "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False}


def epsilon_at_nm(table, wavelength_nm):
    """Piecewise-linear interpolation of measured n,k; never extrapolate."""
    points = table["points"]
    if not math.isfinite(wavelength_nm) or not points[0][0] <= wavelength_nm <= points[-1][0]:
        raise ValueError("outside tabulated spectral range")
    for i in range(len(points) - 1):
        x0, n0, k0 = points[i]
        x1, n1, k1 = points[i + 1]
        if x0 <= wavelength_nm <= x1:
            t = (wavelength_nm - x0) / (x1 - x0)
            n, k = n0 + t * (n1 - n0), k0 + t * (k1 - k0)
            epsilon = complex(n, k) ** 2
            if epsilon.imag < 0:
                raise ValueError("passivity violation")
            return epsilon
    raise ValueError("unbracketed wavelength")
