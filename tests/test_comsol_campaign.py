"""Deliberately fake test inputs verify the COMSOL campaign control plane only.

No test runs COMSOL; the injected fixture is ALWAYS labeled test_fixture.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprpcf.simulation.metrics import extract_metrics
from sprpcf.simulation.verification_campaign import (
    execute_comsol_campaign,
    prepare_comsol_campaign,
    seal_comsol_campaign,
)
from sprpcf.utils.reproducibility import sha256_file


def _fixture(tmp_path: Path):
    root = tmp_path / "benchmark"
    root.mkdir()
    model = tmp_path / "fictional.mph"
    model.write_bytes(b"FAKE TEST MODEL - NEVER COMSOL")
    config = tmp_path / "fictional.yaml"
    config.write_text("study: std1\nwavelength_expression: lambda\nloss_expression: loss\n",
                      encoding="utf-8")
    requests = []
    selected = []
    for target_id in range(3):
        for method in ("tandem_latent", "uniform_random"):
            common = {
                "method": method, "design_seed": 7, "target_id": target_id,
                "target_group_id": f"geometry-{target_id}",
                "analyte_ri": 1.33,
                "pitch_um": 2.0 + target_id * 0.1,
                "d_over_lambda": 0.5,
                "metal_thickness_nm": 40.0,
                "channel_radius_um": 0.6,
            }
            requests.append({
                **common, "request_id": f"fake-request-{target_id}-{method}",
                "target_sensitivity_nm_per_riu": 500.0,
                "target_fom_per_riu": 30.0,
                "target_lambda_res_nm": 600.0,
                "verification_status": "not_run",
                "fixed_geometry_ri_sweep_required": True,
            })
            selected.append(common)
    pd.DataFrame(requests).to_csv(root / "PHYSICS_VERIFICATION_REQUESTS.csv", index=False)
    pd.DataFrame(selected).to_csv(root / "selected_designs.csv", index=False)
    manifest = {
        "experiment": "matched_surrogate_query_inverse_design",
        "scientific_submission_ready": False, "evidence_class": "software_only",
        "outputs_sha256": {
            name: sha256_file(root / name)
            for name in ("PHYSICS_VERIFICATION_REQUESTS.csv", "selected_designs.csv")
        },
    }
    (root / "INVERSE_BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root, model, config


def _plan(tmp_path: Path):
    root, model, config = _fixture(tmp_path)
    data = prepare_comsol_campaign(
        root, model, config, tmp_path / "plan",
        ri_span=0.04, ri_points=5, solver_version="FAKE-FOR-TESTS",
        normalization_scales={
            "sensitivity_nm_per_riu": 100.0,
            "fom_per_riu": 10.0,
            "lambda_res_nm": 20.0,
        },
    )
    return root, model, config, data


def _fake_solver(_model: Path, _config: Path, geometries, *, fail_first: bool = False):
    wavelengths = np.linspace(550, 650, 101)
    rows = []
    for i, geometry in enumerate(geometries):
        if fail_first and i == 0:
            rows.append({
                "sample_id": i, "status": "failed: mock eigenmode failure",
                **geometry.__dict__,
            })
            continue
        peak = 600 + 500 * (geometry.analyte_ri - 1.33)
        loss = 1 + 10 / (1 + ((wavelengths - peak) / 9) ** 2)
        metric = extract_metrics(wavelengths, loss)
        rows.append({
            "sample_id": i, "status": "ok", **geometry.__dict__,
            "wavelength_nm": ",".join(f"{v:.6f}" for v in wavelengths),
            "loss_db_per_cm": ",".join(f"{v:.6f}" for v in loss),
            "lambda_res_nm": metric.lambda_res_nm,
            "fwhm_nm": metric.fwhm_nm,
        })
    return pd.DataFrame(rows)


def test_plan_preregisters_every_request_and_ri_without_simulating(tmp_path: Path):
    _, _, _, data = _plan(tmp_path)
    assert data["planned_request_count"] == 6
    assert data["planned_solver_evaluations"] == 30
    assert data["physical_source_authenticity_verified"] is False
    frame = pd.read_csv(tmp_path / "plan" / "planned_sweeps.csv")
    assert frame["request_id"].nunique() == 6
    assert frame.groupby("request_id").size().eq(5).all()
    assert frame["sample_id"].tolist() == list(range(30))
    assert sha256_file(tmp_path / "plan" / "planned_sweeps.csv") == data["planned_sweeps_sha256"]


def test_mocked_execution_records_spectra_but_is_not_physical_evidence(tmp_path: Path):
    root, model, config, data = _plan(tmp_path)
    executed = execute_comsol_campaign(
        tmp_path / "plan", root, model, config, tmp_path / "run",
        runner=_fake_solver,
    )
    assert executed["real_comsol_backend_invoked"] is False
    assert executed["source_authenticity_verified"] is False
    assert executed["scientific_submission_ready"] is False
    assert executed["scheduled_solver_evaluations"] == 30
    assert executed["successful_solver_evaluations"] == 30
    sweeps = pd.read_csv(tmp_path / "run" / "sweeps.csv")
    assert sweeps["source"].eq("test_fixture").all()
    assert sweeps["evidence_class"].eq("software_only").all()
    assert sweeps["wavelength_nm"].notna().all()
    assert sha256_file(tmp_path / "run" / "sweeps.csv") == executed["sweeps_sha256"]
    native = tmp_path / "fictional_solver_log.txt"
    native.write_text("FAKE log never from COMSOL", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot seal injected test-runner"):
        seal_comsol_campaign(tmp_path / "plan", tmp_path / "run", root, model, config,
                             native, tmp_path / "missing-refinement", tmp_path / "sealed")
    assert not (tmp_path / "sealed").exists()


def test_solver_failure_keeps_all_frozen_requests_and_rows(tmp_path: Path):
    root, model, config, _ = _plan(tmp_path)
    executed = execute_comsol_campaign(
        tmp_path / "plan", root, model, config, tmp_path / "run",
        runner=lambda m, c, g: _fake_solver(m, c, g, fail_first=True),
    )
    assert executed["failed_solver_evaluations"] == 1
    assert executed["successful_solver_evaluations"] == 29
    sweeps = pd.read_csv(tmp_path / "run" / "sweeps.csv")
    assert len(sweeps) == 30
    failed = sweeps.loc[sweeps["status"] != "ok"]
    assert len(failed) == 1
    assert pd.isna(failed.iloc[0]["wavelength_nm"])
    assert "mock eigenmode" in failed.iloc[0]["status"]


def test_missing_comsol_runtime_halts_without_false_completion(tmp_path: Path, monkeypatch):
    import sprpcf.simulation.verification_campaign as module
    root, model, config, _ = _plan(tmp_path)

    def unavailable(*args, **kwargs):
        raise RuntimeError("Actual COMSOL runtime/license unavailable")

    monkeypatch.setattr(module, "run_comsol_geometries", unavailable)
    with pytest.raises(RuntimeError, match="COMSOL runtime/license unavailable"):
        execute_comsol_campaign(tmp_path / "plan", root, model, config, tmp_path / "run")
    assert not (tmp_path / "run" / "execution_manifest.json").exists()


def test_tampered_benchmark_or_plan_refuses_execution(tmp_path: Path):
    root, model, config, _ = _plan(tmp_path)
    original = root / "PHYSICS_VERIFICATION_REQUESTS.csv"
    original.write_text(original.read_text() + "\n")
    with pytest.raises(ValueError, match="frozen output hash mismatch"):
        execute_comsol_campaign(tmp_path / "plan", root, model, config,
                                tmp_path / "run", runner=_fake_solver)
    assert not (tmp_path / "run").exists()


def test_changed_model_or_config_refuses_execution(tmp_path: Path):
    root, model, config, _ = _plan(tmp_path)
    model.write_bytes(model.read_bytes() + b"TAMPER")
    with pytest.raises(ValueError, match="model_sha256"):
        execute_comsol_campaign(tmp_path / "plan", root, model, config,
                                tmp_path / "run", runner=_fake_solver)
    assert not (tmp_path / "run").exists()


def test_mismatched_solver_geometry_or_incomplete_return_rejected(tmp_path: Path):
    root, model, config, _ = _plan(tmp_path)

    def altered(m, c, g):
        df = _fake_solver(m, c, g)
        df.loc[0, "pitch_um"] = 3.5
        return df

    with pytest.raises(ValueError, match="frozen geometry"):
        execute_comsol_campaign(tmp_path / "plan", root, model, config,
                                tmp_path / "bad", runner=altered)

    def missing(m, c, g):
        return _fake_solver(m, c, g).iloc[:-1]

    with pytest.raises(ValueError, match="exactly one row"):
        execute_comsol_campaign(tmp_path / "plan", root, model, config,
                                tmp_path / "missing", runner=missing)


def test_rejects_invalid_scales_and_no_ri_protocol(tmp_path: Path):
    root, model, config = _fixture(tmp_path)
    kwargs = {"ri_span": 0.04, "ri_points": 5, "solver_version": "TEST",
              "normalization_scales": {
                  "sensitivity_nm_per_riu": 0.0,
                  "fom_per_riu": 10.0,
                  "lambda_res_nm": 20.0,
              }}
    with pytest.raises(ValueError, match="positive"):
        prepare_comsol_campaign(root, model, config, tmp_path / "plan", **kwargs)
    kwargs["normalization_scales"]["sensitivity_nm_per_riu"] = 100.0
    kwargs["ri_points"] = 4
    with pytest.raises(ValueError, match="odd integer"):
        prepare_comsol_campaign(root, model, config, tmp_path / "plan", **kwargs)
    assert not (tmp_path / "plan").exists()


def test_cli_plan_only_does_not_require_comsol_runtime(tmp_path: Path):
    root, model, config = _fixture(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_comsol_verification_campaign.py"
    result = subprocess.run(
        [sys.executable, str(script), "plan",
         "--benchmark", str(root), "--model", str(model), "--config", str(config),
         "--out", str(tmp_path / "plan"), "--ri-span", "0.04", "--ri-points", "5",
         "--sensitivity-scale", "100", "--fom-scale", "10", "--resonance-scale", "20",
         "--solver-version", "DUMMY-NOT-COMSOL"],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    result_json = json.loads(result.stdout)
    assert result_json["operation"] == "plan"
    assert result_json["scientific_submission_ready"] is False
    assert (tmp_path / "plan" / "campaign_plan.json").is_file()
