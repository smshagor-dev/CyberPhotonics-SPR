"""Fake COMSOL model and Lorentzian data test the engineering gate, NOT physics."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprpcf.simulation.metrics import extract_metrics
from sprpcf.simulation.model_gate import (
    audit_comsol_model_readiness, validate_live_pilot_report,
)
from sprpcf.simulation.verification_campaign import prepare_comsol_campaign
from sprpcf.utils.reproducibility import sha256_file


class FakeModel:
    def __init__(self, *, missing_parameter=False, study="std1", materials=True):
        self.missing_parameter = missing_parameter
        self.study = study
        self.has_materials = materials

    def parameters(self):
        values = {
            "d_over_lambda": "0.5", "pitch_um": "2[um]",
            "metal_thickness_nm": "40[nm]", "analyte_ri": "1.33",
            "channel_radius_um": "0.6[um]",
        }
        if self.missing_parameter:
            del values["channel_radius_um"]
        return values

    def studies(self):
        return [self.study]

    def geometries(self):
        return ["geom1"]

    def meshes(self):
        return ["mesh1"]

    def physics(self):
        return ["ewfd"]

    def materials(self):
        return ["Gold", "Silica"] if self.has_materials else []

    def problems(self):
        return []

    def version(self):
        return "TEST_ONLY_NOT_COMSOL"


class FakeClient:
    def __init__(self, model):
        self.model = model

    def load(self, _path):
        return self.model


def _inputs(tmp_path: Path, *, example_name: bool = False, scale: str = "1.0"):
    benchmark = tmp_path / "benchmark"
    benchmark.mkdir()
    model = tmp_path / "fake_model.mph"
    model.write_bytes(b"TEST MODEL FILE ONLY; NOT A REAL COMSOL MODEL")
    config = tmp_path / ("fake.example.yaml" if example_name else "test_settings.yaml")
    config.write_text(
        "study: std1\n"
        "wavelength_expression: optical_lambda\n"
        "loss_expression: attenuation\n"
        f"wavelength_scale_to_nm: {scale}\n"
        "loss_scale_to_db_per_cm: 1.0\n"
        "expected_wavelength_nm: [300, 2500]\n",
        encoding="utf-8",
    )
    common = {
        "design_seed": 3, "method": "tandem_latent", "target_id": 0,
        "target_group_id": "g1", "analyte_ri": 1.33,
        "d_over_lambda": 0.5, "pitch_um": 2.0,
        "metal_thickness_nm": 40.0, "channel_radius_um": 0.6,
    }
    requests = [{**common, "request_id": "fixture-req-0",
                 "target_sensitivity_nm_per_riu": 400.0,
                 "target_fom_per_riu": 22.0,
                 "target_lambda_res_nm": 600.0,
                 "fixed_geometry_ri_sweep_required": True,
                 "verification_status": "not_run"}]
    pd.DataFrame(requests).to_csv(benchmark / "PHYSICS_VERIFICATION_REQUESTS.csv", index=False)
    pd.DataFrame([common]).to_csv(benchmark / "selected_designs.csv", index=False)
    manifest = {
        "experiment": "matched_surrogate_query_inverse_design",
        "evidence_class": "software_only", "scientific_submission_ready": False,
        "outputs_sha256": {
            name: sha256_file(benchmark / name) for name in (
                "PHYSICS_VERIFICATION_REQUESTS.csv", "selected_designs.csv"
            )
        },
    }
    (benchmark / "INVERSE_BENCHMARK_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    plan = tmp_path / "plan"
    prepare_comsol_campaign(
        benchmark, model, config, plan,
        ri_span=0.04, ri_points=5, solver_version="TEST_FIXTURE_NOT_COMSOL",
        normalization_scales={
            "sensitivity_nm_per_riu": 100.0,
            "fom_per_riu": 10.0, "lambda_res_nm": 20.0,
        },
    )
    return plan, benchmark, model, config


def _fake_one_geometry(_model, _config, geometries):
    assert len(geometries) == 1
    item = geometries[0]
    assert item.analyte_ri == pytest.approx(1.33)
    wavelengths = np.linspace(550, 650, 101)
    loss = 1 + 10 / (1 + ((wavelengths - 600) / 9.0) ** 2)
    metrics = extract_metrics(wavelengths, loss)
    return pd.DataFrame([{
        "sample_id": 0, "status": "ok", **item.__dict__,
        "wavelength_nm": ",".join(f"{v:.6f}" for v in wavelengths),
        "loss_db_per_cm": ",".join(f"{v:.6f}" for v in loss),
        "lambda_res_nm": float(metrics.lambda_res_nm),
        "fwhm_nm": float(metrics.fwhm_nm),
    }])


def test_static_readiness_never_promotes_unexecuted_solver(tmp_path):
    args = _inputs(tmp_path)
    report = audit_comsol_model_readiness(*args, tmp_path / "out")
    assert report["static_valid"]
    assert not report["live_model_checked"]
    assert not report["pilot_solver_called"]
    assert not report["ready_for_full_solver_run"]
    assert not report["physical_correctness_verified"]
    assert report["observed_backend"] == "not_invoked"
    assert not (tmp_path / "out" / "pilot_single_ri_spectrum.csv").exists()


@pytest.mark.parametrize("mode", ["example", "invalid_scale"])
def test_template_or_bad_units_blocks_real_client_load(tmp_path, mode):
    args = _inputs(
        tmp_path, example_name=(mode == "example"),
        scale="0" if mode == "invalid_scale" else "1.0",
    )
    calls = []

    def forbidden():
        calls.append(1)
        raise AssertionError("Static checks must reject before runtime import")

    report = audit_comsol_model_readiness(
        *args, tmp_path / "out", live=True, pilot=True,
        client_factory=forbidden, geometry_runner=_fake_one_geometry,
    )
    assert not report["static_valid"]
    assert not report["live_model_checked"]
    assert not report["pilot_solver_called"]
    assert not report["ready_for_full_solver_run"]
    assert not calls
    assert report["errors"]


@pytest.mark.parametrize("failure", ["parameter", "study", "materials"])
def test_missing_required_model_components_blocks_pilot(tmp_path, failure):
    args = _inputs(tmp_path)
    fake = FakeModel(
        missing_parameter=failure == "parameter",
        study="incorrect-study" if failure == "study" else "std1",
        materials=failure != "materials",
    )
    report = audit_comsol_model_readiness(
        *args, tmp_path / "out", live=True, pilot=True,
        client_factory=lambda: FakeClient(fake),
        geometry_runner=lambda *a: (_ for _ in ()).throw(
            AssertionError("Never solve with missing model components")
        ),
    )
    assert report["static_valid"]
    assert report["live_model_checked"]
    assert not report["pilot_solver_called"]
    assert not report["ready_for_full_solver_run"]
    assert any(not c["passed"] for c in report["preflight_checks"])


def test_one_ri_fixture_pilot_is_consistent_but_never_real_comsol(tmp_path):
    args = _inputs(tmp_path)
    report = audit_comsol_model_readiness(
        *args, tmp_path / "out", live=True, pilot=True,
        client_factory=lambda: FakeClient(FakeModel()),
        geometry_runner=_fake_one_geometry,
    )
    assert report["pilot_solver_called"]
    assert report["pilot_numerically_consistent"]
    assert report["observed_backend"] == "software_fixture"
    assert report["ready_for_full_solver_run"] is False
    assert report["scientific_submission_ready"] is False
    assert report["physical_correctness_verified"] is False
    assert report["pilot_sample_id"] == 2  # centered 5-point RI grid
    assert report["pilot_request_id"] == "fixture-req-0"
    pilot = tmp_path / "out" / "pilot_single_ri_spectrum.csv"
    assert len(pd.read_csv(pilot)) == 1
    assert sha256_file(pilot) == report["pilot_spectrum_sha256"]


def test_internally_inconsistent_spectrum_does_not_pass_pilot(tmp_path):
    args = _inputs(tmp_path)

    def bad(_model, _config, geometries):
        data = _fake_one_geometry(_model, _config, geometries)
        data.loc[0, "fwhm_nm"] += 5
        return data

    report = audit_comsol_model_readiness(
        *args, tmp_path / "out", live=True, pilot=True,
        client_factory=lambda: FakeClient(FakeModel()), geometry_runner=bad,
    )
    assert report["pilot_solver_called"]
    assert not report["pilot_numerically_consistent"]
    assert not report["ready_for_full_solver_run"]
    assert any("spectrum" in e.lower() for e in report["errors"])


def test_missing_runtime_or_license_is_blocked_without_solver_output(tmp_path):
    args = _inputs(tmp_path)

    def fail():
        raise RuntimeError("licensed COMSOL runtime not available")

    report = audit_comsol_model_readiness(
        *args, tmp_path / "out", live=True, pilot=True, client_factory=fail,
    )
    assert not report["pilot_solver_called"]
    assert not report["ready_for_full_solver_run"]
    assert any("licensed COMSOL" in e for e in report["errors"])
    assert not (tmp_path / "out" / "pilot_single_ri_spectrum.csv").exists()


def test_modified_source_rejected_before_any_files_written(tmp_path):
    plan, benchmark, model, config = _inputs(tmp_path)
    model.write_bytes(model.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="model_sha256"):
        audit_comsol_model_readiness(
            plan, benchmark, model, config, tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_strict_cli_requires_real_pilot_not_static_fixture(tmp_path):
    args = _inputs(tmp_path)
    plan, benchmark, model, config = args
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_comsol_model.py"
    result = subprocess.run(
        [sys.executable, str(path),
         "--plan", str(plan), "--benchmark", str(benchmark),
         "--model", str(model), "--config", str(config),
         "--out", str(tmp_path / "out"), "--strict"],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2, result.stderr
    data = json.loads(
        (tmp_path / "out" / "COMSOL_MODEL_PILOT_READINESS.json").read_text(encoding="utf-8")
    )
    assert data["static_valid"]
    assert data["scientific_submission_ready"] is False
    assert data["ready_for_full_solver_run"] is False



def test_fixture_pilot_cannot_authorize_real_campaign(tmp_path):
    plan, benchmark, model, config = _inputs(tmp_path)
    out = tmp_path / "pilot"
    report = audit_comsol_model_readiness(
        plan, benchmark, model, config, out, live=True, pilot=True,
        client_factory=lambda: FakeClient(FakeModel()),
        geometry_runner=_fake_one_geometry,
    )
    assert report["pilot_numerically_consistent"]
    with pytest.raises(ValueError, match="observed_backend"):
        validate_live_pilot_report(
            out / "COMSOL_MODEL_PILOT_READINESS.json",
            plan, benchmark, model, config,
        )


def test_modified_frozen_model_rejects_previous_pilot(tmp_path):
    plan, benchmark, model, config = _inputs(tmp_path)
    out = tmp_path / "pilot"
    audit_comsol_model_readiness(
        plan, benchmark, model, config, out, live=True, pilot=True,
        client_factory=lambda: FakeClient(FakeModel()),
        geometry_runner=_fake_one_geometry,
    )
    model.write_bytes(model.read_bytes() + b"MODIFIED")
    with pytest.raises(ValueError, match="model_sha256"):
        validate_live_pilot_report(
            out / "COMSOL_MODEL_PILOT_READINESS.json",
            plan, benchmark, model, config,
        )


def test_real_cli_requires_actual_pilot_report_before_batch_execution(tmp_path):
    plan, benchmark, model, config = _inputs(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_comsol_verification_campaign.py"
    out = tmp_path / "never_run"
    args = [
        sys.executable, str(script), "run-batched",
        "--plan", str(plan), "--benchmark", str(benchmark),
        "--model", str(model), "--config", str(config),
        "--out", str(out), "--batch-size", "2",
    ]
    no_gate = subprocess.run(args, capture_output=True, text=True, check=False)
    assert no_gate.returncode == 2
    assert not out.exists()
    fake = tmp_path / "pilot"
    audit_comsol_model_readiness(
        plan, benchmark, model, config, fake, live=True, pilot=True,
        client_factory=lambda: FakeClient(FakeModel()),
        geometry_runner=_fake_one_geometry,
    )
    with_gate = subprocess.run(
        [*args, "--readiness-report", str(fake / "COMSOL_MODEL_PILOT_READINESS.json")],
        capture_output=True, text=True, check=False,
    )
    assert with_gate.returncode != 0
    assert "observed_backend" in with_gate.stderr
    assert not out.exists()
