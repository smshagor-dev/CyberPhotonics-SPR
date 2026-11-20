"""Regression checks for the independently discretized planar TM control."""
import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "research_controls" / "independent_tm_fd_control.py"
spec = importlib.util.spec_from_file_location("independent_tm_fd_control", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_mesh_refinement_approaches_analytic_tm_dispersion():
    errors = [module.solve(dx)["relative_beta_error"] for dx in (0.04, 0.02, 0.01)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.003


def test_surface_localization_and_passivity():
    result = module.solve(0.02)
    assert result["central_field_fraction"] > 0.95
    assert result["neff_imag"] > 0


@pytest.mark.parametrize("dx", [0, -0.01, 0.1])
def test_invalid_spacing_rejected(dx):
    with pytest.raises(ValueError):
        module.solve(dx)
