"""Fail-closed validation of actual FEM mesh-refinement orchestration."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from sprpcf.simulation.open_mesh_refinement import compare_mode_sets, run_mesh_refinement


def test_nearest_eigenvalue_matching_permutation() -> None:
    a = [1.4 + 0.01j, 1.5 + 0.02j]
    b = [1.501 + 0.02j, 1.401 + 0.01j]
    matches = compare_mode_sets(a, b)
    assert [(x["previous_mode_index"], x["current_mode_index"]) for x in matches] == [(0, 1), (1, 0)]
    assert all(x["complex_n_eff_relative_delta"] < 0.01 for x in matches)


def test_nearest_eigenvalue_matching_invalid_fails() -> None:
    with pytest.raises(ValueError, match="identical"):
        compare_mode_sets([1.4 + 0j], [1.4 + 0j, 1.5 + 0j])
    with pytest.raises(ValueError, match="nonfinite"):
        compare_mode_sets([complex(float("nan"), 0)], [1.5 + 0j])


def test_mesh_orchestration_source_flags_never_claim_pcf_evidence(tmp_path: Path) -> None:
    neffs = [1.452 + 1e-6j, 1.451 + 2e-6j]

    def fake_solver(folder: Path, **kwargs):
        folder.mkdir(parents=True, exist_ok=True)
        step = kwargs["gold_mesh_resolution_um"]
        (folder / "raw_eigenvalues.json").write_text(
            json.dumps([
                {"n_eff_real": v.real + step * 0.002, "n_eff_imag": v.imag}
                for v in neffs
            ]), encoding="utf-8",
        )
        (folder / "FEMWELL_PILOT.json").write_text(
            json.dumps({"real_solver_executed": False, "fixture": True}), encoding="utf-8",
        )
        return {"mesh_elements": 1000, "full_vector_eh_dofs_archived": False}

    with patch("sprpcf.simulation.open_mesh_refinement.run_femwell_pcf_probe", side_effect=fake_solver):
        report = run_mesh_refinement(
            tmp_path, gold_mesh_steps_um=(0.03, 0.025, 0.02), num_modes=2,
        )
    assert report["completed"] is True
    assert report["mesh_convergence_verified"] is False
    assert report["scientific_submission_ready"] is False
    assert report["modal_branch_verified"] is False
    assert len(report["runs"]) == 3
    assert len(report["comparisons"]) == 4
    assert (tmp_path / "matched_eigenvalue_changes.csv").exists()


def test_bad_or_partial_solver_run_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="strictly decreasing"):
        run_mesh_refinement(tmp_path, gold_mesh_steps_um=(0.03, 0.03, 0.02))
    with patch("sprpcf.simulation.open_mesh_refinement.run_femwell_pcf_probe", side_effect=RuntimeError("solver failed")):
        with pytest.raises(RuntimeError, match="solver failed"):
            run_mesh_refinement(tmp_path)
    report = json.loads((tmp_path / "REFINEMENT_READINESS.json").read_text(encoding="utf-8"))
    assert report["completed"] is False
    assert report["scientific_submission_ready"] is False
    assert report["error"]["type"] == "RuntimeError"
