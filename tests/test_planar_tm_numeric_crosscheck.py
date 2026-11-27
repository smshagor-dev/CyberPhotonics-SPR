"""Software-only tests with illustrative optical constants; not physics evidence."""
import pytest
from scripts.planar_tm_numeric_crosscheck import check


@pytest.mark.parametrize("epsm,epsd,wl", [
    (-12+1.2j, 2.1, 0.8),
    (-20+2j, 1.77, 0.65),
    (-8+0.4j, 1.0, 1.0),
])
def test_numeric_boundary_root_agrees_with_closed_form(epsm, epsd, wl):
    result = check(epsm, epsd, wl)
    assert result["absolute_neff_error"] < 1e-9
    assert result["normalized_boundary_residual"] < 1e-9
    assert result["power_loss_db_per_cm"] > 0
    assert not result["qualification"]["publication_ready"]


@pytest.mark.parametrize("epsm,epsd,wl", [
    (-12-1j, 2, 0.8),
    (-1+1j, 2, 0.8),
    (-12+1j, 2, -0.8),
    (-12+1j, complex(2, 0.1), 0.8),
])
def test_unqualified_inputs_rejected(epsm, epsd, wl):
    with pytest.raises(ValueError):
        check(epsm, epsd, wl)
