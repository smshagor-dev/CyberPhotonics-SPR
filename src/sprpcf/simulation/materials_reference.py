"""Measured-reference optical n,k interpolation for gold; no extrapolation.

Source: P. B. Johnson & R. W. Christy, Phys. Rev. B 6, 4370 (1972).
CC0 public-domain refractiveindex.info database transcription:
https://github.com/polyanskiy/refractiveindex.info-database/
The measured bulk-film reference constants do NOT independently certify
nanostructured thin-film fabrication, roughness, temperature or sensor physics.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
from importlib import resources
import io
import math

import numpy as np

JOHNSON_CHRISTY_DOI = "https://doi.org/10.1103/PhysRevB.6.4370"
JOHNSON_CHRISTY_SOURCE = (
    "https://github.com/polyanskiy/refractiveindex.info-database/"
    "blob/master/database/data/main/Au/nk/Johnson.yml"
)
JOHNSON_CHRISTY_FILE = "au_johnson_christy_1972.csv"


@lru_cache(maxsize=1)
def _load_johnson_christy() -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
    raw = resources.files("sprpcf.simulation").joinpath(
        "data", JOHNSON_CHRISTY_FILE,
    ).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    content = raw.decode("utf-8")
    lines = [line for line in content.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    if reader.fieldnames != ["wavelength_um", "n", "k"]:
        raise ValueError("unexpected Johnson-Christy reference CSV columns")
    rows = list(reader)
    if len(rows) != 49:
        raise ValueError("expected exactly 49 source-transcribed Au n,k measurements")
    wl = np.array([float(row["wavelength_um"]) for row in rows], dtype=np.float64)
    refractive = np.array([float(row["n"]) for row in rows], dtype=np.float64)
    extinction = np.array([float(row["k"]) for row in rows], dtype=np.float64)
    if (not np.isfinite(wl).all()
            or not np.isfinite(refractive).all()
            or not np.isfinite(extinction).all()
            or not np.all(np.diff(wl) > 0)
            or np.any(wl <= 0) or np.any(refractive < 0) or np.any(extinction <= 0)):
        raise ValueError("invalid or unsorted measured Au optical constants")
    return wl, refractive, extinction, digest


def au_johnson_christy_nk(wavelength_um: np.ndarray | float) -> np.ndarray:
    """Return interpolated n+i*k; valid ONLY within original 0.1879–1.937 μm data.

    Interpolate measured n and k linearly, never epsilon directly. The
    exp(-i*omega*t) convention makes Im(epsilon)>0 for passive gold.
    """
    wl, n, k, _ = _load_johnson_christy()
    wavelength = np.asarray(wavelength_um, dtype=np.float64)
    if (wavelength.size == 0 or not np.isfinite(wavelength).all()
            or np.any(wavelength < wl[0]) or np.any(wavelength > wl[-1])):
        raise ValueError(
            "Au optical wavelength outside source data [0.1879,1.9370] μm; "
            "no extrapolation permitted"
        )
    return np.interp(wavelength, wl, n) + 1j * np.interp(wavelength, wl, k)


def au_johnson_christy_permittivity(wavelength_um: np.ndarray | float) -> np.ndarray:
    """Complex relative permittivity epsilon=(n+i*k)^2, exp(-i omega t)."""
    n_complex = au_johnson_christy_nk(wavelength_um)
    epsilon = np.asarray(n_complex**2, dtype=np.complex128)
    if not np.isfinite(epsilon).all() or np.any(epsilon.imag <= 0):
        raise ValueError("measured-reference Au permittivity must remain passive")
    return epsilon


def au_reference_manifest() -> dict[str, object]:
    """DOI, exact archive hash, source range and reference limitations."""
    wl, _, _, digest = _load_johnson_christy()
    return {
        "material": "Au",
        "source_type": "measured_bulk_film_reference",
        "source_doi": JOHNSON_CHRISTY_DOI,
        "database_source": JOHNSON_CHRISTY_SOURCE,
        "database_license": "CC0-1.0",
        "dataset_file": f"sprpcf/simulation/data/{JOHNSON_CHRISTY_FILE}",
        "dataset_sha256": digest,
        "row_count": len(wl),
        "wavelength_min_um": float(wl[0]),
        "wavelength_max_um": float(wl[-1]),
        "room_temperature": True,
        "interpolation": "linear_in_n_and_k_no_extrapolation",
        "limitations": [
            "bulk-film optical reference data are not measurements of the simulated sensor film",
            "thin-film roughness, grain size and fabrication change effective optics",
            "reference interpolation and measured data provenance do not validate FEM geometry/PML/convergence",
        ],
    }
