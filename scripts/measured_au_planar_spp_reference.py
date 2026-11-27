"""Measured-reference Au planar SPP Maxwell control; NOT PCF sensor evidence."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from sprpcf.simulation.materials_reference import au_johnson_christy_nk, au_reference_manifest


def planar_spp_spectrum(wavelengths_um, dielectric_ri=1.36):
    """Analytic TM interface dispersion with measured bulk Au n,k interpolation."""
    wl = np.asarray(wavelengths_um, dtype=float)
    if (wl.ndim != 1 or wl.size < 3 or not np.isfinite(wl).all()
            or not np.all(np.diff(wl) > 0)):
        raise ValueError("require >=3 strictly increasing finite wavelengths")
    if not (math.isfinite(dielectric_ri) and 1 < dielectric_ri < 2):
        raise ValueError("invalid dielectric RI")
    nk = au_johnson_christy_nk(wl)  # source module rejects extrapolation
    em = nk**2
    ed = dielectric_ri**2
    neff = np.sqrt(ed * em / (ed + em))
    if not (np.isfinite(neff).all() and np.all(em.imag > 0) and np.all(neff.imag > 0)):
        raise ValueError("nonpassive or invalid SPP branch")
    loss = 20 / np.log(10) * 2 * np.pi / wl * neff.imag * 1e4
    length = wl / (4 * np.pi * neff.imag)
    if not np.allclose(loss * length * 1e-4, 10 / np.log(10), rtol=1e-12):
        raise ArithmeticError("attenuation/propagation-length mismatch")
    return [{
        "wavelength_um": float(w), "au_n": float(z.real), "au_k": float(z.imag),
        "au_epsilon_real": float(e.real), "au_epsilon_imag": float(e.imag),
        "spp_neff_real": float(n.real), "spp_neff_imag": float(n.imag),
        "attenuation_db_cm": float(a), "intensity_1e_length_um": float(p),
    } for w, z, e, n, a, p in zip(wl, nk, em, neff, loss, length)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--start-um", type=float, default=0.70)
    parser.add_argument("--stop-um", type=float, default=0.90)
    parser.add_argument("--points", type=int, default=41)
    parser.add_argument("--dielectric-ri", type=float, default=1.36)
    args = parser.parse_args()
    if not 3 <= args.points <= 1001:
        raise ValueError("points must be in [3,1001]")
    samples = planar_spp_spectrum(
        np.linspace(args.start_um, args.stop_um, args.points), args.dielectric_ri
    )
    result = {
        "schema_version": 1, "source_kind": "measured_bulk_Au_planar_analytic_Maxwell_control",
        "au_reference": au_reference_manifest(), "dielectric_ri": args.dielectric_ri,
        "temporal_convention": "exp(-i omega t)", "samples": samples,
        "physics_qualified": False, "publication_ready": False, "comsol_verified": False,
        "limitations": [
            "analytic planar interface, not full-vector PCF FEM or independent solver",
            "bulk reference optics do not characterize the deposited sensor Au film",
            "no PML/domain/mesh convergence, mode tracking or PCF sensitivity evidence",
        ],
    }
    payload = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(payload)
    print(json.dumps({"samples": len(samples),
                      "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                      "au_data_sha256": result["au_reference"]["dataset_sha256"]}))


if __name__ == "__main__":
    main()
