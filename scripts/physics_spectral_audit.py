"""Fail-closed fixed-geometry RI spectral consistency audit (software-only)."""
import math


def audit_spectra(spectra, *, min_samples=9):
    if not isinstance(spectra, list) or len(spectra) < 3:
        return {"accepted": False, "errors": ["at least three RI spectra required"],
                "publication_ready": False, "physics_qualified": False, "comsol_verified": False}
    errors, geometries, grids, ris = [], set(), [], []
    for index, record in enumerate(spectra):
        try:
            if not isinstance(record, dict):
                raise ValueError("invalid spectrum")
            digest = record["geometry_sha256"]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError("invalid geometry digest")
            geometries.add(digest)
            if record["source"] not in ("native_fem", "independent_maxwell"):
                raise ValueError("unqualified source")
            ri = record["ri"]
            if type(ri) not in (int, float) or not math.isfinite(ri) or not 1 < ri < 2:
                raise ValueError("invalid RI")
            ris.append(ri)
            wl, loss = record["wavelength_um"], record["loss_db_cm"]
            if not isinstance(wl, list) or not isinstance(loss, list) or len(wl) != len(loss) or len(wl) < min_samples:
                raise ValueError("insufficient spectral samples")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in wl):
                raise ValueError("invalid wavelengths")
            if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in loss):
                raise ValueError("invalid loss")
            if any(y <= x for x, y in zip(wl, wl[1:])):
                raise ValueError("unordered wavelengths")
            grids.append(tuple(wl))
            peak = max(range(len(loss)), key=loss.__getitem__)
            if peak in (0, len(loss) - 1) or sum(v == loss[peak] for v in loss) != 1:
                raise ValueError("unresolved or ambiguous peak")
            baseline = min(loss[0], loss[-1])
            half = baseline + (loss[peak] - baseline) / 2
            left = any(loss[j] <= half < loss[j + 1] for j in range(peak))
            right = any(loss[j] > half >= loss[j + 1] for j in range(peak, len(loss) - 1))
            if loss[0] >= half or loss[-1] >= half or not left or not right:
                raise ValueError("unbracketed FWHM")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"spectrum {index}: {exc}")
    if len(geometries) != 1:
        errors.append("geometry changed across RI spectra")
    if len(set(ris)) != len(ris):
        errors.append("duplicate RI")
    if grids and any(grid != grids[0] for grid in grids[1:]):
        errors.append("wavelength grids differ")
    return {"accepted": not errors, "errors": errors, "publication_ready": False,
            "physics_qualified": False, "comsol_verified": False}
