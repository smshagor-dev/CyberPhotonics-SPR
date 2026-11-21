"""Planar derivative controls; these are not PCF-SPR sensor claims."""
import pytest
from research.planar_tm_spp_derivative_control import planar_derivative_control, run

@pytest.mark.parametrize("w", [0.75, 0.80, 0.85])
@pytest.mark.parametrize("n", [1.35, 1.36, 1.37])
def test_measured_au_derivative(w, n):
    p = planar_derivative_control(w, n)
    assert p["relative_error"] < 1e-5
    assert p["pcf_sensor_sensitivity"] is False
    assert p["publication_ready"] is False

@pytest.mark.parametrize("n,step", [(1.36, 0), (1.36, -0.001), (1.36, float("nan")), (1.00001, 0.001), (1.99999, 0.001)])
def test_invalid_perturbations(n, step):
    with pytest.raises(ValueError):
        planar_derivative_control(0.8, n, step)

def test_immutable_report(tmp_path):
    path = tmp_path / "derivatives.json"
    result = run(path)
    assert len(result["points"]) == 9
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        run(path)
    assert path.read_bytes() == before
