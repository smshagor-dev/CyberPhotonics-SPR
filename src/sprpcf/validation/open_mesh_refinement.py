"""Actual multi-mesh FEM mode comparison; source-UNQUALIFIED feasibility evidence.

This experiment repeats independent Femwell eigenmode solves of the SAME
exploratory PCF geometry at multiple Au interface mesh resolutions. Matching
by nearest complex eigenvalue is a diagnostic, not identity verification.
Without PML and independent mode tracking the extracted imaginary parts
CANNOT be presented as validated confinement loss or sensor sensitivity.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from sprpcf.simulation.pcf_femwell_probe import ExploratoryPcfTopology, run_femwell_pcf_probe


def compare_mode_sets(
    old_modes: list[complex], new_modes: list[complex],
) -> list[dict[str, float | int]]:
    """Hungarian assignment by complex eigenvalue proximity (not mode-field overlap)."""
    old = np.asarray(old_modes, dtype=np.complex128)
    new = np.asarray(new_modes, dtype=np.complex128)
    if old.ndim != 1 or new.ndim != 1 or not len(old) or len(old) != len(new):
        raise ValueError("mode lists must be nonempty with identical lengths")
    if not np.isfinite(old).all() or not np.isfinite(new).all():
        raise ValueError("nonfinite complex eigenvalue cannot be matched")
    cost = np.abs(old[:, None] - new[None, :])
    row, col = linear_sum_assignment(cost)
    comparison: list[dict[str, float | int]] = []
    for a, b in zip(row, col):
        relative = float(cost[a, b] / max(abs(old[a]), np.finfo(float).tiny))
        comparison.append({
            "previous_mode_index": int(a),
            "current_mode_index": int(b),
            "previous_n_eff_real": float(old[a].real),
            "previous_n_eff_imag": float(old[a].imag),
            "current_n_eff_real": float(new[b].real),
            "current_n_eff_imag": float(new[b].imag),
            "complex_n_eff_abs_delta": float(cost[a, b]),
            "complex_n_eff_relative_delta": relative,
        })
    return comparison


def run_mesh_refinement(
    out: Path,
    *,
    gold_mesh_steps_um: tuple[float, ...] = (0.03, 0.025, 0.02),
    topology: ExploratoryPcfTopology = ExploratoryPcfTopology(),
    wavelength_um: float = 0.8,
    analyte_ri: float = 1.36,
    num_modes: int = 2,
) -> dict[str, Any]:
    """Execute independent full-vector eigenmode FEM runs with real field archives."""
    if (len(gold_mesh_steps_um) < 3
            or any(not math.isfinite(v) or not 0.005 <= v <= 0.03 for v in gold_mesh_steps_um)
            or any(a <= b for a, b in zip(gold_mesh_steps_um, gold_mesh_steps_um[1:]))):
        raise ValueError("require >=3 strictly decreasing Au mesh steps within [0.005, 0.03] μm")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "actual_exploratory_pcf_full_vector_fem_mesh_refinement",
        "source": "femwell_actual_eigenmode_runs",
        "source_evidence_class": "unqualified_solver_feasibility",
        "gold_mesh_steps_um": list(gold_mesh_steps_um),
        "wavelength_um": wavelength_um,
        "analyte_ri": analyte_ri,
        "num_modes_requested": num_modes,
        "eigenvalue_matcher": "Hungarian-nearest-complex-neff-diagnostic-not-field-overlap",
        "topology_from_peer_reviewed_sensor_design": False,
        "au_material_independently_validated": False,
        "pml_validated": False,
        "mesh_convergence_verified": False,
        "modal_branch_verified": False,
        "resolved_sensor_resonance_verified": False,
        "physical_sensor_sensitivity_verified": False,
        "scientific_submission_ready": False,
        "completed": False,
        "error": None,
        "runs": [],
        "comparisons": [],
    }
    previous_modes: list[complex] | None = None
    try:
        for position, step in enumerate(gold_mesh_steps_um):
            folder = out / f"level_{position:02d}"
            pilot = run_femwell_pcf_probe(
                folder, topology=topology,
                wavelength_um=wavelength_um, analyte_ri=analyte_ri,
                num_modes=num_modes, gold_mesh_resolution_um=step,
            )
            mode_path = folder / "raw_eigenvalues.json"
            raw = json.loads(mode_path.read_text(encoding="utf-8"))
            n_eff = [complex(row["n_eff_real"], row["n_eff_imag"]) for row in raw]
            if len(n_eff) != num_modes or not all(
                math.isfinite(z.real) and math.isfinite(z.imag) for z in n_eff
            ):
                raise ValueError("actual FEM eigenpairs are incomplete or nonfinite")
            report["runs"].append({
                "level": position, "gold_mesh_step_um": step,
                "mesh_elements": pilot["mesh_elements"],
                "raw_3d_em_field_dof_archived": pilot["full_vector_eh_dofs_archived"],
                "pilot_manifest": str((folder / "FEMWELL_PILOT.json").relative_to(out)),
                "pilot_manifest_sha256": hashlib.sha256(
                    (folder / "FEMWELL_PILOT.json").read_bytes(),
                ).hexdigest(),
            })
            if previous_modes is not None:
                pairs = compare_mode_sets(previous_modes, n_eff)
                for row in pairs:
                    report["comparisons"].append({
                        "mesh_step_previous_um": gold_mesh_steps_um[position - 1],
                        "mesh_step_current_um": step,
                        **row,
                    })
            previous_modes = n_eff

        with (out / "matched_eigenvalue_changes.csv").open("w", newline="", encoding="utf-8") as f:
            rows = report["comparisons"]
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report["max_last_step_complex_neff_relative_delta"] = max(
            float(row["complex_n_eff_relative_delta"]) for row in report["comparisons"]
            if row["mesh_step_current_um"] == gold_mesh_steps_um[-1]
        )
        report["completed"] = True
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "detail": str(exc)}
        raise
    finally:
        (out / "REFINEMENT_READINESS.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    return report
