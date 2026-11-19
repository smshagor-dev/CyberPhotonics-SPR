from scripts.physics_spectral_audit import audit_spectra


def sample():
    return [dict(geometry_sha256="a" * 64, source="native_fem", ri=1.35 + i * .01,
                 wavelength_um=[.7 + j * .01 for j in range(11)],
                 loss_db_cm=[1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]) for i in range(3)]


def test_nominal_still_not_publication_ready():
    result = audit_spectra(sample())
    assert result["accepted"] and not result["publication_ready"]


def test_mismatched_geometry():
    x = sample()
    x[1]["geometry_sha256"] = "b" * 64
    assert not audit_spectra(x)["accepted"]


def test_shifted_grid():
    x = sample()
    x[1]["wavelength_um"][0] = .69
    assert not audit_spectra(x)["accepted"]


def test_duplicate_ri():
    x = sample()
    x[1]["ri"] = x[0]["ri"]
    assert not audit_spectra(x)["accepted"]


def test_unbracketed_peak():
    x = sample()
    x[0]["loss_db_cm"] = [4, 4, 4, 4, 5, 6, 5, 5, 5, 5, 5]
    assert not audit_spectra(x)["accepted"]


def test_synthetic():
    x = sample()
    x[0]["source"] = "synthetic"
    assert not audit_spectra(x)["accepted"]


def test_nonfinite():
    x = sample()
    x[0]["loss_db_cm"][4] = float("nan")
    assert not audit_spectra(x)["accepted"]
