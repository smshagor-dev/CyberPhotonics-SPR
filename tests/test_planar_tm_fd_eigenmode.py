"""Numerical TM eigenmode regression tests; not PCF evidence."""
import pytest
from research.planar_tm_fd_eigenmode import solve


def test_mesh_refinement_reduces_error():
    errors = [solve(dx_um=h)["relative_beta_error"] for h in (0.08, 0.04, 0.02)]
    assert errors[0] > errors[1] > errors[2]
    assert errors[-1] < 0.01


def test_matrix_residual_and_provenance():
    result = solve()
    assert result["relative_matrix_residual"] < 1e-8
    assert result["beta_numeric_per_um"][1] > 0
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False


@pytest.mark.parametrize("params", [
    {"dx_um": 0}, {"half_domain_um": 0}, {"wavelength_um": 0},
    {"analyte_ri": 0}, {"metal_epsilon": -24-1j},
    {"dx_um": float("nan")}, {"dx_um": 0.017},
])
def test_invalid_inputs_fail_closed(params):
    with pytest.raises(ValueError):
        solve(**params)
