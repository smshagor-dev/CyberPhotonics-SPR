"""COMSOL request-bound derivative leakage regressions (FAKE MPh backend).

No real electromagnetic solver, model, materials or measured device is used.
These tests verify that raw solver spectra are preserved and that duplicate
RI requests or incomplete sweeps cannot cause invented sensitivity/FOM.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from sprpcf.simulation.comsol_sweep import run_comsol_geometries, run_comsol_sweep
from sprpcf.simulation.schema import Geometry


class FakeMphModel:
    def __init__(self, *, fail_ri: float | None = None):
        self.fail_ri = fail_ri
        self.current_ri = 1.33
        self.attempted = 0

    def parameter(self, key, value):
        if key == "analyte_ri":
            self.current_ri = float(value)

    def solve(self, study):
        assert study == "std1"
        self.attempted += 1
        if self.fail_ri is not None and self.current_ri == self.fail_ri:
            raise RuntimeError("FICTIONAL solver cannot compute this RI")

    def evaluate(self, expression):
        wavelength = np.linspace(550.0, 650.0, 101)
        resonance = 600.0 + 400.0 * (self.current_ri - 1.33)
        loss = 1.0 + 10.0 / (1.0 + ((wavelength - resonance) / 9.0) ** 2)
        return wavelength if expression == "lambda" else loss


def _install_fake(monkeypatch, *, fail_ri=None):
    import sprpcf.simulation.comsol_sweep as backend

    model = FakeMphModel(fail_ri=fail_ri)
    monkeypatch.setattr(
        backend, "_load_mph",
        lambda: SimpleNamespace(
            start=lambda: SimpleNamespace(load=lambda _: model)
        ),
    )
    return model


def _config(tmp_path, ri_values):
    cfg = tmp_path / "fixture.yaml"
    cfg.write_text(
        "study: std1\n"
        "wavelength_expression: lambda\nloss_expression: optical_loss\n"
        "wavelength_scale_to_nm: 1\nloss_scale_to_db_per_cm: 1\n"
        "expected_wavelength_nm: [500, 800]\n"
        "sweep:\n"
        "  d_over_lambda: [0.5]\n"
        "  pitch_um: [2.0]\n"
        "  metal_thickness_nm: [40.0]\n"
        "  channel_radius_um: [0.6]\n"
        "  analyte_ri: [" + ", ".join(str(x) for x in ri_values) + "]\n",
        encoding="utf-8",
    )
    return cfg


def _geometry(ri: float, pitch: float = 2.0):
    return Geometry(
        d_over_lambda=0.5, pitch_um=pitch, metal_thickness_nm=40.0,
        channel_radius_um=0.6, analyte_ri=ri,
    )


def test_overlapping_candidate_requests_keep_all_raw_solver_spectra(tmp_path, monkeypatch):
    """Identical target RI is legal across separate inverse-design requests."""
    fake = _install_fake(monkeypatch)
    cfg = _config(tmp_path, [1.33, 1.35])
    requests = [
        _geometry(1.33),
        _geometry(1.35),
        _geometry(1.33),  # repeated candidate; previously failed after all solves
        _geometry(1.37, pitch=2.2),
    ]
    result = run_comsol_geometries(tmp_path / "NOT_COMSOL.mph", cfg, requests)
    assert fake.attempted == 4
    assert len(result) == 4
    assert result["sample_id"].tolist() == [0, 1, 2, 3]
    assert result["status"].eq("ok").all()
    assert result["analyte_ri"].tolist() == [1.33, 1.35, 1.33, 1.37]
    assert result["wavelength_nm"].str.len().min() > 100
    assert result["loss_db_per_cm"].str.len().min() > 100
    # Arbitrary batches provide no frozen request grouping. No cross-request
    # finite differences or fabricated FOM can be assigned here.
    assert "sensitivity_nm_per_riu" not in result
    assert "fom_per_riu" not in result
    assert "sensitivity_diagnostic_status" not in result


def test_explicit_full_unique_ri_grid_produces_only_unqualified_diagnostics(
    tmp_path, monkeypatch,
):
    fake = _install_fake(monkeypatch)
    cfg = _config(tmp_path, [1.33, 1.35, 1.37])
    grid = run_comsol_sweep(tmp_path / "NOT_COMSOL.mph", cfg)
    assert fake.attempted == 3
    assert grid["status"].eq("ok").all()
    assert grid["sensitivity_diagnostic_status"].eq(
        "grid_diagnostic_only_unqualified"
    ).all()
    assert np.isfinite(grid["sensitivity_nm_per_riu"]).all()
    assert np.allclose(grid["sensitivity_nm_per_riu"], 400.0, atol=2.0)
    assert np.isfinite(grid["fom_per_riu"]).all()
    assert (grid["fom_per_riu"] > 0).all()


def test_duplicate_ri_in_explicit_config_keeps_original_outputs_and_no_derivative(
    tmp_path, monkeypatch,
):
    fake = _install_fake(monkeypatch)
    cfg = _config(tmp_path, [1.33, 1.35, 1.33])
    grid = run_comsol_sweep(tmp_path / "NOT_COMSOL.mph", cfg)
    assert fake.attempted == 3
    assert grid["status"].eq("ok").all()
    assert len(grid) == 3
    assert grid["sensitivity_diagnostic_status"].eq(
        "duplicate_ri_points_no_derivative"
    ).all()
    assert grid["sensitivity_nm_per_riu"].isna().all()
    assert grid["fom_per_riu"].isna().all()
    assert grid["lambda_res_nm"].notna().all()
    assert grid["loss_db_per_cm"].notna().all()


def test_failed_ri_withholds_diagnostic_for_entire_geometry_group(
    tmp_path, monkeypatch,
):
    fake = _install_fake(monkeypatch, fail_ri=1.35)
    cfg = _config(tmp_path, [1.33, 1.35, 1.37])
    grid = run_comsol_sweep(tmp_path / "NOT_COMSOL.mph", cfg)
    assert fake.attempted == 3
    assert grid["status"].tolist()[1].startswith("failed:")
    assert grid["status"].eq("ok").sum() == 2
    assert grid["sensitivity_nm_per_riu"].isna().all()
    assert grid["fom_per_riu"].isna().all()
    assert grid["sensitivity_diagnostic_status"].tolist() == [
        "incomplete_grid_solver_failure",
        "solver_failed",
        "incomplete_grid_solver_failure",
    ]
    assert grid.loc[grid["status"].eq("ok"), "wavelength_nm"].notna().all()


def test_single_sample_batch_never_infers_ri_derivative(tmp_path, monkeypatch):
    fake = _install_fake(monkeypatch)
    cfg = _config(tmp_path, [1.33])
    single = run_comsol_geometries(
        tmp_path / "NOT_COMSOL.mph", cfg, [_geometry(1.33)]
    )
    assert fake.attempted == 1
    assert single.iloc[0]["status"] == "ok"
    assert single.iloc[0]["fwhm_nm"] > 0
    assert "sensitivity_nm_per_riu" not in single


def test_explicit_single_ri_grid_has_declared_insufficient_status(
    tmp_path, monkeypatch,
):
    _install_fake(monkeypatch)
    cfg = _config(tmp_path, [1.33])
    grid = run_comsol_sweep(tmp_path / "NOT_COMSOL.mph", cfg)
    assert grid["sensitivity_diagnostic_status"].tolist() == [
        "insufficient_unique_ri_points"
    ]
    assert grid["sensitivity_nm_per_riu"].isna().all()
    assert grid["fom_per_riu"].isna().all()
