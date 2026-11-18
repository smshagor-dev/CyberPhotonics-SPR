"""Genuine 2D vector Maxwell eigenmode grid on one frozen exploratory FEM mesh.

SCIENCE BOUNDARY: a finite-box, one-ring demonstrator with fitted Au optics,
no PML, no verified core/SPP modal branch, no resonance/linewidth/sensitivity.
Never promote these raw eigenpairs to qualified PCF-SPR sensor performance.
"""
from __future__ import annotations

from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from sprpcf.simulation.materials_reference import au_johnson_christy_permittivity, au_reference_manifest

from sprpcf.simulation.pcf_femwell_probe import (
    ExploratoryPcfTopology, fitted_gold_permittivity, silica_index_sellmeier,
)


def preflight_grid(
    wavelengths_um: tuple[float, ...], analyte_ris: tuple[float, ...], num_modes: int,
) -> None:
    """Fail before import/meshing if the planned grid is invalid."""
    for name, values in (("wavelengths", wavelengths_um), ("analyte_ris", analyte_ris)):
        if len(values) < 2 or len(values) > 8 or not all(math.isfinite(v) for v in values):
            raise ValueError(f"{name} needs 2–8 finite points")
        if any(a >= b for a, b in zip(values, values[1:])):
            raise ValueError(f"{name} must be strictly increasing")
    if not all(0.45 <= v <= 1.7 for v in wavelengths_um):
        raise ValueError("wavelengths exceed the validated demo-model range")
    if not all(1.0 < v < 2.0 for v in analyte_ris):
        raise ValueError("RI values must lie within (1,2)")
    if type(num_modes) is not int or not 1 <= num_modes <= 6:
        raise ValueError("num_modes must be an integer in [1,6]")
    if len(wavelengths_um) * len(analyte_ris) > 36:
        raise ValueError("grid is too large for CPU-only exploratory CI")


def matched_vector_overlaps(reference: Any, candidate: Any) -> list[dict[str, float | int]]:
    """Actual E/H Poynting modal overlap; diagnostic assignment only."""
    if len(reference) != len(candidate) or not len(reference):
        raise ValueError("two equal nonempty sets of vector eigenmodes required")
    matrix = np.empty((len(reference), len(candidate)), dtype=float)
    for i, mode_i in enumerate(reference):
        for j, mode_j in enumerate(candidate):
            val = complex(mode_i.calculate_overlap(mode_j))
            if not (math.isfinite(val.real) and math.isfinite(val.imag)):
                raise ValueError("complex vector-field overlap is nonfinite")
            matrix[i, j] = abs(val)
    a, b = linear_sum_assignment(-matrix)
    matches = []
    for i, j in zip(a, b):
        selected = float(matrix[i, j])
        row_rival = float(max((matrix[i, k] for k in range(len(candidate)) if k != j), default=0.0))
        col_rival = float(max((matrix[k, j] for k in range(len(reference)) if k != i), default=0.0))
        # Assignment is global; a selected edge need not be the strongest in
        # either direction. Negative margins expose this ambiguity explicitly.
        matches.append({
            "previous_mode_index": int(i),
            "current_mode_index": int(j),
            "vector_eh_poynting_overlap_abs": selected,
            "unselected_overlap_abs": row_rival,
            "reverse_competing_overlap_abs": col_rival,
            "forward_overlap_margin": selected - row_rival,
            "reverse_overlap_margin": selected - col_rival,
            "bidirectional_min_overlap_margin": min(selected - row_rival, selected - col_rival),
            "assignment_ambiguous_nonpositive_margin": (
                selected <= row_rival or selected <= col_rival
            ),
            "mode_identity_physically_verified": False,
        })
    return matches


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_vector_grid(
    out: Path,
    *,
    wavelengths_um: tuple[float, ...] = (0.75, 0.80, 0.85),
    analyte_ris: tuple[float, ...] = (1.35, 1.36, 1.37),
    topology: ExploratoryPcfTopology = ExploratoryPcfTopology(),
    gold_mesh_resolution_um: float = 0.025,
    num_modes: int = 2,
    gold_source: str = "fitted",
) -> dict[str, Any]:
    """Real independent eigenvalue solves on ONE spatial mesh, with E/H archives."""
    preflight_grid(wavelengths_um, analyte_ris, num_modes)
    if gold_source not in ("fitted", "johnson_christy_1972"):
        raise ValueError("gold_source must be fitted or johnson_christy_1972")
    if gold_source == "johnson_christy_1972":
        for wavelength_um in wavelengths_um:
            au_johnson_christy_permittivity(wavelength_um)
    if not math.isfinite(gold_mesh_resolution_um) or not 0.005 <= gold_mesh_resolution_um <= 0.03:
        raise ValueError("invalid exploratory Au mesh step")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "unqualified_femwell_full_vector_wavelength_ri_grid",
        "evidence_class": "unqualified_solver_feasibility",
        "model_description": "exploratory_2d_PCF_single_ring_finite_box_Au_shell",
        "wavelengths_um": list(wavelengths_um),
        "analyte_ris": list(analyte_ris),
        "num_modes_requested": num_modes,
        "gold_mesh_resolution_um": gold_mesh_resolution_um,
        "full_vector_maxwell_solver": "femwell.maxwell.waveguide.compute_modes",
        "mode_matching": "same_mesh_vector_EH_Poynting_overlap_diagnostic_only",
        "sensor_geometry_peer_reviewed_and_frozen": False,
        "Au_material_measured_data_provenance_verified": gold_source == "johnson_christy_1972",
        "Au_material_reference_only_not_sensor_film_measurement": True,
        "pml_validated": False,
        "mesh_convergence_verified": False,
        "mode_tracking_physically_verified": False,
        "resonance_fwhm_verified": False,
        "physical_sensor_sensitivity_verified": False,
        "scientific_submission_ready": False,
        "solver_calls_completed": 0,
        "completed": False,
        "failures": [],
        "artifacts_sha256": {},
    }
    try:
        from femwell.maxwell.waveguide import compute_modes
        from femwell.mesh import mesh_from_OrderedDict
        from skfem import Basis, ElementTriP0
        from skfem.io.meshio import from_meshio

        polygons = topology.build()
        resolutions = {
            "gold": {"resolution": gold_mesh_resolution_um, "distance": 0.15},
            "analyte": {"resolution": 0.05, "distance": 0.2},
        }
        for key in polygons:
            if key.startswith("hole_"):
                resolutions[key] = {"resolution": 0.11, "distance": 0.2}
        mesh_path = out / "pcf_frozen_exploratory_grid_mesh.msh"
        mesh_data = mesh_from_OrderedDict(
            polygons, resolutions, filename=str(mesh_path), default_resolution_max=0.5,
        )
        mesh = from_meshio(mesh_data)
        if not 100 <= mesh.nelements <= 150000:
            raise ValueError(f"unreasonable mesh element count {mesh.nelements}")
        basis0 = Basis(mesh, ElementTriP0())

        geometry_path = out / "topology.json"
        geometry_path.write_text(json.dumps({
            "geometry": asdict(topology), "regions": {key: float(x.area) for key, x in polygons.items()},
            "source": "exploratory_not_peer_reviewed", "units": "micrometre",
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "wavelengths_um": wavelengths_um, "analyte_ris": analyte_ris,
            "num_modes": num_modes, "gold_mesh_resolution_um": gold_mesh_resolution_um,
            "Au": ("Johnson_Christy_1972_measured_bulk_reference" if gold_source == "johnson_christy_1972"
                   else "repository_Rakic_style_fitted_model_not_measured_constants"),
            "Au_source_manifest": au_reference_manifest() if gold_source == "johnson_christy_1972" else None,
            "silica": "three_term_Sellmeier",
            "temporal_convention": "exp(-i*omega*t)",
        }
        freeze = out / "experiment_protocol.json"
        freeze.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        rows: list[dict[str, Any]] = []
        overlaps: list[dict[str, Any]] = []
        sampled_modes: dict[tuple[int, int], Any] = {}
        import importlib.metadata
        report["solver_version"] = importlib.metadata.version("femwell")
        report["mesh_elements"] = int(mesh.nelements)

        for ri_index, analyte_ri in enumerate(analyte_ris):
            for wavelength_index, wavelength_um in enumerate(wavelengths_um):
                epsilon = basis0.zeros(dtype=np.complex128)
                metal = (complex(au_johnson_christy_permittivity(wavelength_um))
                         if gold_source == "johnson_christy_1972"
                         else fitted_gold_permittivity(wavelength_um))
                silica = silica_index_sellmeier(wavelength_um)
                for region in polygons:
                    eps = (
                        metal if region == "gold"
                        else analyte_ri**2 if region == "analyte"
                        else 1.0 if region.startswith("hole_")
                        else silica**2
                    )
                    epsilon[basis0.get_dofs(elements=region)] = eps

                modes = compute_modes(
                    basis0, epsilon, wavelength=wavelength_um,
                    num_modes=num_modes, order=1, n_guess=silica,
                )
                if len(modes) != num_modes:
                    raise ValueError("solver produced incomplete eigenmode set")
                report["solver_calls_completed"] += 1
                sampled_modes[ri_index, wavelength_index] = modes

                fields: dict[str, np.ndarray] = {}
                for mode_index, mode in enumerate(modes):
                    neff = complex(mode.n_eff)
                    e_field = np.asarray(mode.E, dtype=np.complex128)
                    h_field = np.asarray(mode.H, dtype=np.complex128)
                    if (not math.isfinite(neff.real) or not math.isfinite(neff.imag)
                            or not np.isfinite(e_field).all()
                            or not np.isfinite(h_field).all()):
                        raise ValueError("nonfinite eigenvalue/vector field")
                    fields[f"mode_{mode_index}_E"] = e_field
                    fields[f"mode_{mode_index}_H"] = h_field
                    rows.append({
                        "ri_index": ri_index, "wavelength_index": wavelength_index,
                        "analyte_ri": analyte_ri, "wavelength_um": wavelength_um,
                        "mode_index_native_untracked": mode_index,
                        "n_eff_real": neff.real, "n_eff_imag": neff.imag,
                        # Diagnostic only: no PML, no validated confinement loss.
                        # 8.686 * (2*pi/lambda_um) * Im(neff) * 1e4 dB/cm.
                        "raw_attenuation_db_per_cm_unqualified": (
                            8.685889638065037 * 2.0 * math.pi
                            * neff.imag / wavelength_um * 1.0e4
                        ),
                        "gold_epsilon_real": metal.real, "gold_epsilon_imag": metal.imag,
                        "silica_ri": silica,
                        "mesh_elements": mesh.nelements,
                        "vector_field_e_dofs": e_field.size,
                        "vector_field_h_dofs": h_field.size,
                    })
                field_path = out / f"raw_vector_eh_ri{ri_index}_wl{wavelength_index}.npz"
                np.savez_compressed(field_path, **fields)
                report["artifacts_sha256"][field_path.name] = _sha256(field_path)

        for ri_index in range(len(analyte_ris)):
            for wi in range(1, len(wavelengths_um)):
                pairs = matched_vector_overlaps(sampled_modes[ri_index, wi-1], sampled_modes[ri_index, wi])
                overlaps.extend({"varying": "wavelength", "ri_index": ri_index, "previous_wl_index": wi-1,
                                 "current_wl_index": wi, **pair} for pair in pairs)
        for wi in range(len(wavelengths_um)):
            for ri_index in range(1, len(analyte_ris)):
                pairs = matched_vector_overlaps(sampled_modes[ri_index-1, wi], sampled_modes[ri_index, wi])
                overlaps.extend({"varying": "analyte_ri", "wavelength_index": wi, "previous_ri_index": ri_index-1,
                                 "current_ri_index": ri_index, **pair} for pair in pairs)

        records_path = out / "raw_eigenmode_grid.csv"
        with records_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        overlaps_path = out / "vector_mode_overlap_diagnostics.json"
        overlaps_path.write_text(json.dumps(overlaps, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["raw_eigenmode_rows"] = len(rows)
        report["vector_overlap_matched_edges"] = len(overlaps)
        report["ambiguous_overlap_edges"] = sum(
            bool(edge["assignment_ambiguous_nonpositive_margin"]) for edge in overlaps
        )
        report["all_overlap_edges_bidirectionally_unambiguous"] = (
            report["ambiguous_overlap_edges"] == 0
        )
        # Neither an unambiguous assignment nor an attenuation calculation
        # establishes physical core/SPP identity or PML convergence.
        for path in [mesh_path, geometry_path, freeze, records_path, overlaps_path]:
            report["artifacts_sha256"][path.name] = _sha256(path)
        report["completed"] = True
    except Exception as exc:
        report["failures"].append({"type": type(exc).__name__, "message": str(exc)})
        raise
    finally:
        (out / "GRID_READINESS.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return report
