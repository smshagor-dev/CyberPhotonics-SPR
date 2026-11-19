import math
from dataclasses import replace
import pytest
from research_audits.complex_mode_loss import ModeSample, attenuation_db_per_cm, fixed_geometry_spectrum

H = 'a'*64

def sample(**kwargs):
    return replace(ModeSample(0.8,1.36,1.44,1e-5,H,'femwell','core-1',H,True,True),**kwargs)

def test_physical_units():
    expected=(20/math.log(10))*(2*math.pi/(0.8e-4))*1e-5
    assert math.isclose(attenuation_db_per_cm(sample()),expected,rel_tol=1e-13)

def test_zero_loss():
    assert attenuation_db_per_cm(sample(neff_imag=0)) == 0

@pytest.mark.parametrize('change',[
    {'neff_imag':-1e-6},{'neff_imag':float('nan')},{'wavelength_um':0},
    {'native_artifact_sha256':'bad'},{'solver_source':'comsol'},
    {'solver_source':'synthetic'},{'mode_track_id':''}
])
def test_fail_closed(change):
    with pytest.raises(ValueError):
        attenuation_db_per_cm(sample(**change))

def test_valid_spectrum():
    rows=[sample(wavelength_um=x) for x in [0.7,0.75,0.8,0.85,0.9]]
    assert len(fixed_geometry_spectrum(rows)) == 5

def test_mixed_geometry_rejected():
    rows=[sample(wavelength_um=x) for x in [0.7,0.75,0.8,0.85,0.9]]
    rows[2]=replace(rows[2],geometry_sha256='b'*64)
    with pytest.raises(ValueError):
        fixed_geometry_spectrum(rows)

def test_unverified_pml_rejected():
    rows=[sample(wavelength_um=x) for x in [0.7,0.75,0.8,0.85,0.9]]
    rows[2]=replace(rows[2],pml_converged=False)
    with pytest.raises(ValueError):
        fixed_geometry_spectrum(rows)

def test_duplicate_wavelength_rejected():
    rows=[sample(wavelength_um=x) for x in [0.7,0.75,0.8,0.8,0.9]]
    with pytest.raises(ValueError):
        fixed_geometry_spectrum(rows)
