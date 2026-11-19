import numpy as np
from research.overlap_material_step_audit import audit_overlap

def sample(**changes):
    f = np.zeros((2, 3, 3), complex)
    f[0, 0, 0] = 1
    f[1, 1, 1] = 1
    kwargs = dict(mesh0="mesh", mesh1="mesh", geometry0="geometry",
                  geometry1="geometry", material0="au-a", material1="au-b")
    kwargs.update(changes)
    return audit_overlap(f, f, f, f, np.ones(3), **kwargs)

def test_ri_step_passes_diagnostic_not_identity():
    r = sample(ri0=1.35, ri1=1.36)
    assert r["overlap_tracking_passed"] and not r["mode_identity_verified"]
    assert not r["physics_qualified"] and not r["publication_ready"]

def test_wavelength_step_passes():
    assert sample(wavelength0=.80, wavelength1=.81)["overlap_tracking_passed"]

def test_unexplained_material_change_fails():
    assert not sample()["overlap_tracking_passed"]

def test_incomplete_ri_fails():
    assert not sample(ri0=1.35)["overlap_tracking_passed"]

def test_nonfinite_ri_fails():
    assert not sample(ri0=float("nan"), ri1=1.36)["overlap_tracking_passed"]

def test_mesh_mismatch_fails():
    assert not sample(ri0=1.35, ri1=1.36, mesh1="other")["overlap_tracking_passed"]

def test_geometry_mismatch_fails():
    assert not sample(ri0=1.35, ri1=1.36, geometry1="other")["overlap_tracking_passed"]

def test_invalid_threshold_fails():
    assert not sample(ri0=1.35, ri1=1.36, min_overlap=-1)["overlap_tracking_passed"]

def test_small_finite_field_amplitude_is_not_arbitrarily_rejected():
    f = np.zeros((2, 3, 3), complex)
    f[0, 0, 0] = 1e-10
    f[1, 1, 1] = 1e-10
    r = audit_overlap(f, f, f, f, np.ones(3),
                      mesh0="m", mesh1="m", geometry0="g", geometry1="g",
                      material0="a", material1="a")
    assert r["overlap_tracking_passed"]
    assert not r["physics_qualified"]


def test_overflowed_quadrature_norm_fails_closed():
    f = np.zeros((2, 3, 3), complex)
    f[0, 0, 0] = 1e200
    f[1, 1, 1] = 1e200
    r = audit_overlap(f, f, f, f, np.ones(3),
                      mesh0="m", mesh1="m", geometry0="g", geometry1="g",
                      material0="a", material1="a")
    assert not r["overlap_tracking_passed"]
