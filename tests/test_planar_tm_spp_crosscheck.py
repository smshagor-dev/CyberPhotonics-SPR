"""Software-only numerical verification; no claim of PCF qualification."""
import pytest
from research.planar_tm_spp_crosscheck import solve


@pytest.mark.parametrize("wavelength", [0.75, 0.80, 0.85])
@pytest.mark.parametrize("ri", [1.35, 1.36, 1.37])
def test_planar_tm_complex_root_matches_closed_form(wavelength, ri):
    result = solve(wavelength, ri, complex(-24, 1.5))
    assert result["planar_crosscheck_passed"]
    assert result["relative_beta_error"] < 1e-8
    assert result["tm_boundary_residual"] < 1e-8
    assert result["beta_numeric_per_um"][1] > 0
    assert result["physics_qualified"] is False
    assert result["publication_ready"] is False
    assert result["comsol_verified"] is False


@pytest.mark.parametrize("wavelength,ri,eps", [
    (0, 1.36, complex(-24, 1.5)),
    (0.8, 0, complex(-24, 1.5)),
    (0.8, 1.36, complex(-24, -1.5)),
])
def test_invalid_interface_rejected(wavelength, ri, eps):
    with pytest.raises(ValueError):
        solve(wavelength, ri, eps)


@pytest.mark.parametrize("wavelength", [0.75, 0.80, 0.85])
@pytest.mark.parametrize("ri", [1.35, 1.36, 1.37])
def test_independent_seeds_converge_to_same_root(wavelength, ri):
    result = solve(wavelength, ri, complex(-24, 1.5))
    assert result["independent_seed_count"] == 3
    assert result["max_seed_relative_disagreement"] < 1e-8
    assert result["root_solver_success"]
