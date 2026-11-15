"""Tests for numerical planar-SPP FDFD control; not full PCF solver."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from sprpcf.simulation.open_physics import (
    attenuation_db_per_cm,
    fitted_gold_epsilon,
    planar_spp_exact,
    run_planar_control,
    solve_planar_tm,
)


def test_planar_spp_refinement_converges_to_independent_exact_mode(tmp_path: Path) -> None:
    report = run_planar_control(tmp_path)
    assert report["pass"] is True
    assert report["relative_error"] < 0.003
    assert report["relative_residual"] < 1e-8
    assert report["mesh_error_monotonically_decreasing"]
    assert report["scientific_submission_ready"] is False
    assert report["open_fem_physics_qualified"] is False
    assert report["physical_sensor_sensitivity_verified"] is False
    with (tmp_path / "mesh_convergence.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert all(float(row["relative_error"]) > 0 for row in rows)
    assert json.loads((tmp_path / "CONTROL_READINESS.json").read_text(encoding="utf-8"))["pass"]


def test_passive_loss_sign_and_scale() -> None:
    neff = 1.4 + 0.01j
    expected = (20 / math.log(10)) * (2 * math.pi / 0.8e-6) * 0.01 / 100
    assert math.isclose(attenuation_db_per_cm(neff, 0.8), expected)
    with pytest.raises(ValueError, match="gain"):
        attenuation_db_per_cm(1.4 - 0.01j, 0.8)


def test_gold_is_passive_and_nonmetal_control_fails() -> None:
    gold = fitted_gold_epsilon(0.8)
    assert gold.real < -(1.36**2)
    assert gold.imag > 0
    assert planar_spp_exact(0.8, 1.36, gold).imag > 0
    with pytest.raises(ValueError, match="not a passive"):
        planar_spp_exact(0.8, 1.36, 2.1 + 0.1j)


def test_invalid_settings_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="three"):
        run_planar_control(tmp_path, mesh_steps_um=(0.08, 0.04))
    with pytest.raises(ValueError, match="strictly"):
        run_planar_control(tmp_path, mesh_steps_um=(0.02, 0.08, 0.01))
    with pytest.raises(ValueError, match="mesh_step_um"):
        solve_planar_tm(mesh_step_um=float("nan"))
    with pytest.raises(ValueError, match="0.45"):
        fitted_gold_epsilon(0.1)


def test_measured_gold_not_misrepresented(tmp_path: Path) -> None:
    report = run_planar_control(tmp_path)
    assert not report["material"]["measured_constants_used"]
    assert "not full-vector PCF-SPR" in report["limitations"][1]
