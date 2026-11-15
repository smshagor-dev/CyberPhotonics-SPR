"""Independent ANALYTIC planar Au/analyte SPP limit, NOT a PCF solver.

Numerically compares source-traceable measured-reference Au planar-interface
dispersion with unselected native Femwell grid eigenvalues. A planar interface
is not the PCF curved 50 nm film; no PML/core/SPP/mode or resonance claim.
"""
from __future__ import annotations

import argparse
import cmath
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sprpcf.simulation.materials_reference import (
    au_johnson_christy_permittivity, au_reference_manifest,
)
from verify_measured_au_fem_grid import verify_measured_au_grid


def planar_au_analyte_spp_index(wavelength_um: float, analyte_ri: float) -> complex:
    """TM SPP complex n_eff for ideal flat semi-infinite Au/dielectric interface.

    Exact planar Maxwell boundary formula for two homogeneous half-spaces,
    applied only where the lossless bound-SPP real-permittivity criterion holds.
    """
    if not (math.isfinite(wavelength_um) and math.isfinite(analyte_ri)
            and 1 < analyte_ri < 2):
        raise ValueError("invalid wavelength or analyte RI")
    metal = complex(au_johnson_christy_permittivity(wavelength_um))
    dielectric = analyte_ri**2
    if metal.real >= -dielectric or metal.imag <= 0:
        raise ValueError("not in passive planar-interface SPP reference regime")
    effective = cmath.sqrt(metal * dielectric / (metal + dielectric))
    if not (math.isfinite(effective.real) and math.isfinite(effective.imag)
            and effective.real > 0 and effective.imag > 0):
        raise ValueError("invalid passive planar SPP solution")
    return effective


def audit_native_planar_reference(root: Path) -> dict:
    root = Path(root)
    audit = verify_measured_au_grid(root)
    if not audit["passed"]:
        raise ValueError(f"native Au-FEM archive failed: {audit['errors']}")
    manifest = json.loads((root / "GRID_READINESS.json").read_text(encoding="utf-8"))
    protocol = json.loads((root / "experiment_protocol.json").read_text(encoding="utf-8"))
    with (root / "raw_eigenmode_grid.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        key = (float(row["analyte_ri"]), float(row["wavelength_um"]))
        grouped.setdefault(key, []).append(row)
    expected = {(ri, wl) for ri in protocol["analyte_ris"]
                for wl in protocol["wavelengths_um"]}
    if set(grouped) != expected:
        raise ValueError("missing or extra native grid coordinates")
    points = []
    for ri, wl in sorted(expected):
        n_spp = planar_au_analyte_spp_index(wl, ri)
        modes = grouped[ri, wl]
        if len(modes) != protocol["num_modes"]:
            raise ValueError("incomplete modal grid")
        raw_neff = [complex(float(x["n_eff_real"]), float(x["n_eff_imag"]))
                    for x in modes]
        if not all(math.isfinite(n.real) and math.isfinite(n.imag) for n in raw_neff):
            raise ValueError("nonfinite raw solver eigenvalue")
        points.append({
            "analyte_ri": ri, "wavelength_um": wl,
            "planar_reference_n_eff_real": n_spp.real,
            "planar_reference_n_eff_imag": n_spp.imag,
            "closest_raw_fem_real_neff_delta_abs": min(abs(n.real - n_spp.real) for n in raw_neff),
            "raw_fem_neff_real_min": min(n.real for n in raw_neff),
            "raw_fem_neff_real_max": max(n.real for n in raw_neff),
            "closest_mode_identity_ignored": True,
        })
    return {
        "evidence_class": "analytic_flat_interface_reference_vs_unqualified_raw_fem",
        "reference": "ideal semi-infinite planar Au/analyte TM SPP",
        "not_an_independent_full_vector_pcf_solver": True,
        "source_doi": au_reference_manifest()["source_doi"],
        "source_dataset_sha256": au_reference_manifest()["dataset_sha256"],
        "native_grid_readiness_sha256": hashlib.sha256(
            (root / "GRID_READINESS.json").read_bytes()).hexdigest(),
        "native_eigenvalues_sha256": manifest["artifacts_sha256"]["raw_eigenmode_grid.csv"],
        "native_mesh_sha256": manifest["artifacts_sha256"]["pcf_frozen_exploratory_grid_mesh.msh"],
        "solver": manifest["full_vector_maxwell_solver"],
        "sample_count": len(points),
        "points": points,
        "analytical_formula": "sqrt(eps_Au * n_analyte^2 / (eps_Au + n_analyte^2))",
        "curved_finite_Au_film_is_not_planar_interface": True,
        "core_spp_phase_matching_verified": False,
        "PML_validated": False,
        "independent_solver_verified": False,
        "spectral_convergence_verified": False,
        "physical_sensitivity_verified": False,
        "scientific_submission_ready": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_grid", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = audit_native_planar_reference(args.native_grid)
        code = 0
    except Exception as exc:
        result = {"evidence_class": "failed_analytic_reference",
                  "error_type": type(exc).__name__, "error": str(exc),
                  "scientific_submission_ready": False}
        code = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
