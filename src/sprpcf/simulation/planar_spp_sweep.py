"""Source-traceable analytical planar TM-SPP sweep; NEVER PCF or COMSOL evidence."""
from __future__ import annotations
import cmath
import hashlib
import json
import math
import os
from pathlib import Path

def planar_reference(wavelength_um: float, analyte_ri: float, au_epsilon: complex) -> dict:
    """Independent half-space dispersion and boundary residual, exp(i beta z-i omega t)."""
    wl, ri, em = float(wavelength_um), float(analyte_ri), complex(au_epsilon)
    if not all(map(math.isfinite, (wl, ri, em.real, em.imag))):
        raise ValueError("nonfinite input")
    ed = ri * ri
    if wl <= 0 or ri <= 0 or em.imag <= 0 or em.real >= -ed:
        raise ValueError("outside passive bound planar TM-SPP domain")
    neff = cmath.sqrt(em * ed / (em + ed))
    if neff.real < 0:
        neff = -neff
    if neff.imag <= 0:
        raise ValueError("nonattenuating or wrong propagation branch")
    kd = cmath.sqrt(neff * neff - ed)
    km = cmath.sqrt(neff * neff - em)
    if kd.real < 0:
        kd = -kd
    if km.real < 0:
        km = -km
    if kd.real <= 0 or km.real <= 0:
        raise ValueError("no decaying half-space branches")
    left, right = kd / ed, km / em
    residual = abs(left + right) / (abs(left) + abs(right))
    loss = (20 / math.log(10)) * (2 * math.pi / (wl * 1e-4)) * neff.imag
    if not math.isfinite(loss) or not math.isfinite(residual):
        raise ValueError("nonfinite diagnostic")
    return {"wavelength_um": wl, "analyte_ri": ri,
            "n_eff_real": neff.real, "n_eff_imag": neff.imag,
            "loss_db_per_cm": loss, "normalized_tm_boundary_residual": residual,
            "decaying_half_spaces": True, "physics_qualified": False,
            "publication_ready": False, "comsol_verified": False}

def run_sourced_sweep(wavelengths_um, analyte_ris, output_path):
    """Immutable JSON from repository Johnson-Christy bulk Au data, not PCF mode data."""
    from sprpcf.simulation.materials_reference import (
        au_johnson_christy_permittivity, au_reference_manifest)
    wavelengths = [float(w) for w in wavelengths_um]
    ris = [float(r) for r in analyte_ris]
    if not wavelengths or not ris or len(set(wavelengths)) != len(wavelengths) or len(set(ris)) != len(ris):
        raise ValueError("nonempty unique wavelength/RI grids required")
    if sorted(wavelengths) != wavelengths or sorted(ris) != ris:
        raise ValueError("grids must be strictly increasing")
    rows = []
    for wl in wavelengths:
        em = complex(au_johnson_christy_permittivity(wl))
        for ri in ris:
            rows.append(planar_reference(wl, ri, em))
    result = {"model": "analytic_planar_tm_spp_half_spaces_not_pcf",
              "gold": au_reference_manifest(), "rows": rows,
              "failure_count": 0, "physics_qualified": False,
              "publication_ready": False, "comsol_verified": False}
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("refuse overwrite of immutable reference")
    # O_EXCL avoids the TOCTOU race in exists()+write_bytes().
    # fsync ensures bytes are flushed before the digest is returned.
    with output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(payload).hexdigest()
