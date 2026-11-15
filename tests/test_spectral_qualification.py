"""Software-only tests; synthetic arrays are never physics evidence."""
from sprpcf.simulation.spectral_qualification import PhysicsEvidence, qualify_spectrum


def test_pilot_three_points_rejected():
    result = qualify_spectrum([750, 800, 850], [1, 4, 1])
    assert not result["qualified"]
    assert "minimum_11_matched_samples" in result["failures"]
    assert result["fwhm_nm"] is None


def test_unverified_reference_shape_rejected():
    x = list(range(700, 901, 5))
    y = [1 + 8 / (1 + ((v - 800) / 25)**2) for v in x]
    result = qualify_spectrum(x, y)
    assert "unverified_independent_solver" in result["failures"]
    assert result["scientific_submission_ready"] is False


def test_unbracketed_peak_rejected():
    x = list(range(700, 901, 5))
    y = list(reversed(range(len(x))))
    result = qualify_spectrum(x, y, PhysicsEvidence(True, True, True, True, True, True))
    assert "unbracketed_peak" in result["failures"]


def test_nonfinite_rejected():
    x = list(range(700, 901, 5))
    y = [1.0] * len(x)
    y[10] = float("nan")
    assert "nonfinite_or_empty" in qualify_spectrum(x, y)["failures"]


def test_no_half_height_crossings():
    x = list(range(700, 901, 5))
    y = [10.0] * len(x)
    y[0], y[20] = 1.0, 12.0
    assert "unbracketed_fwhm" in qualify_spectrum(x, y)["failures"]
