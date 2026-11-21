"""Software regression and sourced-data sanity checks; not PCF qualification."""
import math

import pytest

from research.planar_tm_spp_reference import planar_tm_control, run_control


@pytest.mark.parametrize("wavelength", [0.75, 0.80, 0.85])
@pytest.mark.parametrize("ri", [1.35, 1.36, 1.37])
def test_independent_tm_root(wavelength, ri):
    point = planar_tm_control(wavelength, ri)
    assert point["root_relative_difference"] < 1e-8
    assert point["tm_boundary_residual"] < 1e-8
    assert point["beta_imag_um_inv"] > 0
    assert point["propagation_length_power_um"] > 0
    assert point["power_attenuation_db_cm"] == pytest.approx(
        (20 / math.log(10)) * point["beta_imag_um_inv"] * 1e4
    )
    assert point["qualified_pcf_physics"] is False
    assert point["publication_ready"] is False
    assert point["comsol_verified"] is False


@pytest.mark.parametrize("wavelength,ri", [
    (float("nan"), 1.36), (0.0, 1.36), (10.0, 1.36),
    (0.8, float("inf")), (0.8, 2.0),
])
def test_bad_inputs_fail_closed(wavelength, ri):
    with pytest.raises(ValueError):
        planar_tm_control(wavelength, ri)


def test_immutable_output_and_provenance(tmp_path):
    path = tmp_path / "control.json"
    report = run_control(path)
    assert len(report["points"]) == 9
    assert report["source"]["source_doi"].endswith("PhysRevB.6.4370")
    assert len(report["source"]["dataset_sha256"]) == 64
    assert report["publication_ready"] is False
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        run_control(path)
    assert path.read_bytes() == original
