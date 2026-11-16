"""Software-only regression tests: constructed eigenvalues are not Maxwell evidence."""
from research.strict_complex_grid import audit

def samples():
    return [dict(h_um=h, neff_real=n, neff_imag=.001,
                 geometry_sha256="a"*64, material_sha256="b"*64,
                 pml_sha256="c"*64, archive_sha256=str(i)*64,
                 mode_tracking_verified=True)
            for i, (h, n) in enumerate(((.04, 1.36), (.02, 1.35), (.01, 1.3475)), 1)]

def test_constructed_diagnostic_is_not_physics():
    r = audit(samples())
    assert r["diagnostic_passed"]
    assert abs(r["observed_order_diagnostic"] - 2) < 1e-9
    assert r["physics_qualified"] is False and r["publication_ready"] is False

def test_phase_reversal():
    s = samples()
    s[2]["neff_real"] = 1.3525
    assert not audit(s)["diagnostic_passed"]

def test_forged_tracking_flag():
    s = samples()
    s[0]["mode_tracking_verified"] = 1
    assert not audit(s)["diagnostic_passed"]

def test_duplicate_archive():
    s = samples()
    s[2]["archive_sha256"] = s[1]["archive_sha256"]
    assert not audit(s)["diagnostic_passed"]

def test_changed_material():
    s = samples()
    s[1]["material_sha256"] = "d"*64
    assert not audit(s)["diagnostic_passed"]

def test_nonuniform_mesh():
    s = samples()
    s[0]["h_um"] = .03
    assert not audit(s)["diagnostic_passed"]

def test_nonfinite_eigenvalue():
    s = samples()
    s[0]["neff_real"] = float("nan")
    assert not audit(s)["diagnostic_passed"]

def test_invalid_hash():
    s = samples()
    s[0]["pml_sha256"] = "placeholder"
    assert not audit(s)["diagnostic_passed"]
