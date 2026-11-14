"""Numerical controls for COMSOL spectra, using an INJECTED FAKE MPh client.

These are software falsification tests, NEVER real electromagnetic evidence.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from sprpcf.simulation.comsol_sweep import (
    _normalize_evaluated_spectrum, run_comsol_geometries,
)
from sprpcf.simulation.schema import Geometry
from sprpcf.simulation.spectrum_quality import (
    normalize_comsol_export, validate_optical_spectrum,
)


def _geometry(ri: float = 1.33) -> Geometry:
    return Geometry(
        d_over_lambda=0.5, pitch_um=2.0, metal_thickness_nm=40.0,
        channel_radius_um=0.6, analyte_ri=ri,
    )


def _lorentzian(peak: float = 600.0):
    w = np.linspace(550.0, 650.0, 101)
    loss = 1 + 10 / (1 + ((w - peak) / 9.0) ** 2)
    return w, loss


@pytest.mark.parametrize("parameter", [
    "d_over_lambda", "pitch_um", "metal_thickness_nm",
    "channel_radius_um", "analyte_ri",
])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_geometry_rejects_nonfinite_comsol_parameters(parameter, bad):
    params = dict(_geometry().__dict__)
    params[parameter] = bad
    with pytest.raises(ValueError, match="finite"):
        Geometry(**params).validate()


def test_valid_numpy_float32_geometry_still_works():
    params = {key: np.float32(value) for key, value in _geometry().__dict__.items()}
    Geometry(**params).validate()


def test_singleton_column_spectrum_is_supported_without_changing_units():
    w, loss = _lorentzian()
    v, l = normalize_comsol_export(
        w.reshape(-1, 1), loss.reshape(-1, 1),
        {"wavelength_scale_to_nm": 1.0, "loss_scale_to_db_per_cm": 1.0,
         "expected_wavelength_nm": [500, 800]},
    )
    assert np.array_equal(v, w)
    assert np.array_equal(l, loss)
    assert validate_optical_spectrum(v, l).fwhm_nm > 10


def test_true_multimode_columns_are_not_concatenated_into_one_fictitious_peak():
    w, loss = _lorentzian()
    config = {"expected_wavelength_nm": [500, 800]}
    with pytest.raises(ValueError, match="multiple modal/solution columns"):
        _normalize_evaluated_spectrum(
            np.column_stack((w, w + 0.1)),
            np.column_stack((loss, loss * 1.5)),
            config,
        )
    with pytest.raises(ValueError, match="multiple modal/solution columns"):
        _normalize_evaluated_spectrum(
            w, np.column_stack((loss, loss * 1.5)), config
        )


def test_complex_nonfinite_and_bad_conversion_are_rejected():
    w, loss = _lorentzian()
    with pytest.raises(ValueError, match="complex-valued"):
        normalize_comsol_export(w, loss + 1j * 0.01, {})
    with pytest.raises(ValueError, match="positive finite"):
        normalize_comsol_export(w, loss, {"loss_scale_to_db_per_cm": float("nan")})
    with pytest.raises(ValueError, match="positive finite"):
        normalize_comsol_export(w, loss, {"wavelength_scale_to_nm": -1})
    with pytest.raises(ValueError, match="nonfinite"):
        normalize_comsol_export(w, np.full_like(loss, float("nan")), {})


@pytest.mark.parametrize("defect,reason", [
    ("flat", "contrast"),
    ("negative", "negative"),
    ("boundary", "boundary"),
    ("narrow", "under-resolved"),
    ("too_short", "16"),
    ("nonmonotonic", "strictly increasing"),
])
def test_invalid_spectrum_cannot_produce_publishable_metrics(defect, reason):
    w, loss = _lorentzian()
    if defect == "flat":
        loss = np.full_like(w, 3.0)
    elif defect == "negative":
        loss -= 10
    elif defect == "boundary":
        _, loss = _lorentzian(551.0)
    elif defect == "narrow":
        loss = 1 + 10 * np.exp(-0.5 * ((w - 600) / 0.5) ** 2)
    elif defect == "too_short":
        w, loss = w[:12], loss[:12]
    elif defect == "nonmonotonic":
        w[30] = w[29]
    with pytest.raises(ValueError, match=reason):
        validate_optical_spectrum(w, loss)


class FakeModel:
    def __init__(self, modes: list[str]):
        self.modes = modes
        self.calls = 0
        self.current_ri = 1.33

    def parameter(self, name, value):
        if name == "analyte_ri":
            self.current_ri = float(value)

    def solve(self, study):
        assert study == "std1"
        self.calls += 1

    def evaluate(self, expression):
        w, l = _lorentzian(600 + 400 * (self.current_ri - 1.33))
        mode = self.modes[self.calls - 1]
        if mode == "multimode":
            w = np.column_stack((w, w + 1))
            l = np.column_stack((l, l + 2))
        elif mode == "flat":
            l[:] = 5.0
        elif mode == "complex":
            l = l + 1j * 0.2
        elif mode == "narrow":
            l = 1 + 10 * np.exp(-0.5 * ((w - 600) / 0.5) ** 2)
        return w if expression == "lambda" else l


def test_real_backend_failure_isolation_with_fake_client(monkeypatch, tmp_path):
    """One corrupted mode cannot become 'ok' or discard following successes."""
    import sprpcf.simulation.comsol_sweep as backend
    model = FakeModel(["multimode", "flat", "ok", "complex"])
    monkeypatch.setattr(
        backend, "_load_mph",
        lambda: SimpleNamespace(start=lambda: SimpleNamespace(load=lambda _: model)),
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "study: std1\nwavelength_expression: lambda\nloss_expression: loss\n"
        "wavelength_scale_to_nm: 1\nloss_scale_to_db_per_cm: 1\n"
        "expected_wavelength_nm: [500, 800]\n",
        encoding="utf-8",
    )
    frame = run_comsol_geometries(
        tmp_path / "FAKE_NOT_COMSOL.mph", cfg,
        [_geometry(1.33), _geometry(1.34), _geometry(1.35), _geometry(1.36)],
    )
    assert len(frame) == 4
    assert model.calls == 4
    assert frame["sample_id"].tolist() == [0, 1, 2, 3]
    assert frame["status"].str.startswith("failed:").tolist() == [
        True, True, False, True
    ]
    assert "multiple modal" in frame.iloc[0]["status"]
    assert "contrast" in frame.iloc[1]["status"]
    assert frame.iloc[2]["status"] == "ok"
    assert frame.iloc[2]["fwhm_nm"] > 0
    assert frame.iloc[2]["wavelength_nm"]
    assert "complex-valued" in frame.iloc[3]["status"]


def test_serialize_raw_comsol_export_with_no_six_decimal_truncation(
    monkeypatch, tmp_path,
):
    import sprpcf.simulation.comsol_sweep as backend
    model = FakeModel(["ok"])
    monkeypatch.setattr(
        backend, "_load_mph",
        lambda: SimpleNamespace(start=lambda: SimpleNamespace(load=lambda _: model)),
    )
    cfg = tmp_path / "config.yaml"
    cfg.write_text("study: std1\nwavelength_expression: lambda\n"
                   "loss_expression: loss\n", encoding="utf-8")
    frame = run_comsol_geometries(tmp_path / "fake.mph", cfg, [_geometry()])
    assert frame.iloc[0]["status"] == "ok"
    optical = np.fromstring(frame.iloc[0]["loss_db_per_cm"], sep=",")
    original_loss = _lorentzian()[1]
    assert np.array_equal(optical, original_loss)
    assert frame.iloc[0]["source"] if "source" in frame else True
