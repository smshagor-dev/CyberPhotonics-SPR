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


def test_large_finite_eigenvector_scale_is_valid():
    f = np.zeros((2, 3, 3), complex)
    f[0, 0, 0] = 1e200
    f[1, 1, 1] = 1e200
    r = audit_overlap(f, f, f, f, np.ones(3),
                      mesh0="m", mesh1="m", geometry0="g", geometry1="g",
                      material0="a", material1="a")
    assert r["overlap_tracking_passed"]
    assert not r["physics_qualified"]

def test_tiny_finite_eigenvector_scale_is_valid():
    f = np.zeros((2, 3, 3), complex)
    f[0, 0, 0] = 1e-200
    f[1, 1, 1] = 1e-200
    r = audit_overlap(f, f, f, f, np.ones(3),
                      mesh0="m", mesh1="m", geometry0="g", geometry1="g",
                      material0="a", material1="a")
    assert r["overlap_tracking_passed"]
    assert not r["mode_identity_verified"]
    assert not r["publication_ready"]


def test_overlap_invariant_to_extreme_common_quadrature_scale():
    f = np.zeros((2, 3, 3), complex)
    f[0, 0, 0] = 1
    f[1, 1, 1] = 1
    def run(weights):
        return audit_overlap(f, f, f, f, weights,
                             mesh0="m", mesh1="m", geometry0="g", geometry1="g",
                             material0="a", material1="a")
    baseline = run(np.array([1.0, 2.0, 3.0]))
    assert baseline["overlap_tracking_passed"]
    for scale in (1e-200, 1e200):
        result = run(np.array([1.0, 2.0, 3.0]) * scale)
        assert result["overlap_tracking_passed"], result["errors"]
        for a, b in zip(baseline["assignments"], result["assignments"]):
            assert a["before"] == b["before"] and a["after"] == b["after"]
            assert np.isclose(a["e_overlap"], b["e_overlap"])
            assert np.isclose(a["h_overlap"], b["h_overlap"])


def test_invalid_quadrature_weights_fail_closed():
    for weights in (np.array([1., 0., 1.]), np.array([1., np.inf, 1.]),
                    np.array([1., np.nan, 1.])):
        f = np.ones((2, 3, 3), complex)
        result = audit_overlap(f, f, f, f, weights,
                               mesh0="m", mesh1="m", geometry0="g", geometry1="g",
                               material0="a", material1="a")
        assert not result["overlap_tracking_passed"]
        assert not result["physics_qualified"]
