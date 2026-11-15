"""Compare real finite-box Maxwell eigenmode archives from two DOMAIN sizes.

A diagnostic of domain+remeshing sensitivity only: nearest-eigenvalue mode
association does NOT validate core/SPP identity; no PML or physical loss claims.
Raw provenance must come from native Femwell solver, never from fixtures.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from verify_measured_au_fem_grid import verify_measured_au_grid


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def eigenvalue_pairs(left: list[complex], right: list[complex]) -> dict:
    """Nearest complex eigenvalue pairing only; never a physical mode label."""
    a = np.asarray(left, dtype=np.complex128)
    b = np.asarray(right, dtype=np.complex128)
    if a.ndim != 1 or b.ndim != 1 or len(a) != len(b) or len(a) < 2:
        raise ValueError("need equal sets of at least two complex eigenmodes")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite eigenvalues")
    cost = np.abs(a[:, None] - b[None, :])
    i, j = linear_sum_assignment(cost)
    assignment_cost = float(cost[i, j].sum())
    # Two-mode assignment uncertainty is explicit. For >2 modes, no
    # unsupported uniqueness claim is made.
    margin = None
    if len(a) == 2:
        diagonal = float(cost[0, 0] + cost[1, 1])
        swapped = float(cost[0, 1] + cost[1, 0])
        margin = abs(diagonal - swapped)
    # Diagnostic threshold only; a tiny assignment margin means the pairing is unstable.
    ambiguous = margin is None or margin <= max(1e-9, 0.01 * assignment_cost)
    pairs = [{
        "small_mode_index_unverified": int(x),
        "large_mode_index_unverified": int(y),
        "n_eff_small_real": float(a[x].real),
        "n_eff_small_imag": float(a[x].imag),
        "n_eff_large_real": float(b[y].real),
        "n_eff_large_imag": float(b[y].imag),
        "complex_n_eff_abs_delta": float(cost[x, y]),
        "relative_imag_n_eff_delta": float(
            abs(a[x].imag - b[y].imag) / max(abs(a[x].imag), abs(b[y].imag), 1e-20)
        ),
        "core_spp_identity_verified": False,
    } for x, y in zip(i, j)]
    return {
        "pairs": pairs,
        "optimal_assignment_cost": assignment_cost,
        "two_mode_alternative_assignment_margin": margin,
        "eigenvalue_assignment_ambiguous": ambiguous,
        "assignment_margin_is_diagnostic_not_mode_proof": True,
        "assignment_physics_verified": False,
    }


def compare_domain_grids(small: Path, large: Path) -> dict:
    small, large = Path(small), Path(large)
    for root in (small, large):
        audit = verify_measured_au_grid(root)
        if not audit["passed"]:
            raise ValueError(f"{root}: invalid measured-Au native archive: {audit['errors']}")
    p = [json.loads((root / "experiment_protocol.json").read_text()) for root in (small, large)]
    t = [json.loads((root / "topology.json").read_text()) for root in (small, large)]
    r = [json.loads((root / "GRID_READINESS.json").read_text()) for root in (small, large)]
    for key in ("wavelengths_um", "analyte_ris", "num_modes", "gold_mesh_resolution_um",
                "Au", "Au_source_manifest", "silica", "temporal_convention"):
        if p[0].get(key) != p[1].get(key):
            raise ValueError(f"non-domain protocol changed: {key}")
    topology = [dict(x["geometry"]) for x in t]
    widths = [float(v.pop("domain_halfwidth_um")) for v in topology]
    if not (math.isfinite(widths[0]) and math.isfinite(widths[1]) and widths[0] < widths[1]):
        raise ValueError("domain halfwidth must increase")
    if topology[0] != topology[1]:
        raise ValueError("core/Au/air sensor geometry changed as well as domain")
    if r[0]["solver_version"] != r[1]["solver_version"]:
        raise ValueError("different solver versions")
    if r[0]["artifacts_sha256"]["pcf_frozen_exploratory_grid_mesh.msh"] == (
            r[1]["artifacts_sha256"]["pcf_frozen_exploratory_grid_mesh.msh"]):
        raise ValueError("identical mesh archive despite changed domain")
    rows = []
    for root in (small, large):
        with (root / "raw_eigenmode_grid.csv").open(newline="", encoding="utf-8") as handle:
            rows.append(list(csv.DictReader(handle)))
    def modes_for(data: list[dict], i: int, j: int) -> list[complex]:
        members = sorted((x for x in data if int(x["ri_index"]) == i and
                          int(x["wavelength_index"]) == j),
                         key=lambda x: int(x["mode_index_native_untracked"]))
        if len(members) != p[0]["num_modes"]:
            raise ValueError("missing modal sample")
        return [complex(float(x["n_eff_real"]), float(x["n_eff_imag"])) for x in members]

    entries = []
    for ri_index, ri in enumerate(p[0]["analyte_ris"]):
        for wl_index, wavelength in enumerate(p[0]["wavelengths_um"]):
            match = eigenvalue_pairs(modes_for(rows[0], ri_index, wl_index),
                                     modes_for(rows[1], ri_index, wl_index))
            entries.append({"ri_index": ri_index, "wavelength_index": wl_index,
                            "analyte_ri": ri, "wavelength_um": wavelength, **match})
    pairs = [pair for entry in entries for pair in entry["pairs"]]
    return {
        "schema_version": 1,
        "evidence_class": "unqualified_finite_box_domain_remesh_diagnostic",
        "source_solver": "femwell.maxwell.waveguide.compute_modes",
        "optical_material_reference": "Johnson-Christy 1972 bulk gold",
        "source_archive_grid_readiness_sha256": [_digest(root / "GRID_READINESS.json") for root in (small, large)],
        "source_mesh_sha256": [x["artifacts_sha256"]["pcf_frozen_exploratory_grid_mesh.msh"] for x in r],
        "source_solver_version": r[0]["solver_version"],
        "domain_halfwidths_um": widths,
        "mesh_elements": [x["mesh_elements"] for x in r],
        "mesh_resolution_near_gold_um": p[0]["gold_mesh_resolution_um"],
        "eigenvalue_comparisons": entries,
        "mode_pairs_diagnostic_only": len(pairs),
        "ambiguous_gridpoint_pairings": sum(bool(x["eigenvalue_assignment_ambiguous"]) for x in entries),
        "max_complex_n_eff_abs_delta": max(x["complex_n_eff_abs_delta"] for x in pairs),
        "max_relative_imag_n_eff_delta": max(x["relative_imag_n_eff_delta"] for x in pairs),
        "domain_and_remeshing_effects_not_separable": True,
        "PML_present": False,
        "domain_convergence_verified": False,
        "core_spp_identity_verified": False,
        "resonance_fwhm_verified": False,
        "physical_sensitivity_verified": False,
        "independent_solver_crosscheck_verified": False,
        "scientific_submission_ready": False,
        "failure_count": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("small", type=Path)
    parser.add_argument("large", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = compare_domain_grids(args.small, args.large)
        status = 0
    except Exception as exc:
        result = {"evidence_class": "failed_domain_comparison",
                  "error_type": type(exc).__name__, "error": str(exc),
                  "failure_count": 1, "scientific_submission_ready": False}
        status = 1
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
