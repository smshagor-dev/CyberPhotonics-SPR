"""Regression checks for unqualified planar Maxwell TM reference."""
import math
import pytest
from sprpcf.simulation.planar_tm_root_reference import solve_planar_tm_root


@pytest.mark.parametrize("wavelength", [0.756, 0.8211, 0.892])
def test_measured_gold_planar_root(wavelength):
    result = solve_planar_tm_root(wavelength, 1.35)
    assert result["neff_imag"] > 0
    assert result["relative_closed_form_error"] < 1e-8
    assert result["boundary_residual"] < 1e-8
    assert math.isfinite(result["intensity_attenuation_db_per_cm"])
    assert result["intensity_attenuation_db_per_cm"] > 0
    assert result["publication_qualified"] is False
    assert result["independent_pcf_solver_verified"] is False


@pytest.mark.parametrize("wavelength,ri", [
    (float("nan"), 1.35), (0.0, 1.35), (0.8, float("inf")),
    (0.8, 1.0), (0.8, 2.0), (2.5, 1.35),
])
def test_invalid_inputs_fail_closed(wavelength, ri):
    with pytest.raises(ValueError):
        solve_planar_tm_root(wavelength, ri)


def test_ri_changes_interface_dispersion():
    a = solve_planar_tm_root(0.8211, 1.34)
    b = solve_planar_tm_root(0.8211, 1.36)
    assert b["neff_real"] > a["neff_real"]
