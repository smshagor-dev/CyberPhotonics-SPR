"""Stage 11: entirely synthetic COMSOL RI-response guard regressions.

These injected tests use fake .mph bytes and Lorentzian curves. They are NOT
evidence for any real optical sensitivity, experiment, or journal results.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sprpcf.simulation.metrics import extract_metrics
from sprpcf.simulation.model_gate import audit_comsol_model_readiness
from sprpcf.simulation.ri_response_gate import (
    probe_comsol_ri_response, validate_ri_response_report,
)
from sprpcf.simulation.verification_campaign import prepare_comsol_campaign
from sprpcf.utils.reproducibility import sha256_file


def _files(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    root = tmp_path / "benchmark"
    root.mkdir()
    model = tmp_path / "FAKE_NOT_COMSOL.mph"
    model.write_bytes(b"FAKE DEMONSTRATION ONLY")
    config = tmp_path / "non_template_config.yaml"
    config.write_text(
        "study: std1\nwavelength_expression: lambda\n"
        "loss_expression: optical_loss\nwavelength_scale_to_nm: 1.0\n"
        "loss_scale_to_db_per_cm: 1.0\nexpected_wavelength_nm: [300, 2500]\n",
        encoding="utf-8",
    )
    common = {
        "design_seed": 7, "method": "uniform_random", "target_id": 0,
        "target_group_id": "group_A",
        "pitch_um": 2.0, "d_over_lambda": 0.5,
        "metal_thickness_nm": 40.0, "channel_radius_um": 0.6,
        "analyte_ri": 1.33,
    }
    request = {
        **common, "request_id": "synthetic_request_zero",
        "target_sensitivity_nm_per_riu": 400.0,
        "target_fom_per_riu": 20.0,
        "target_lambda_res_nm": 600.0,
        "verification_status": "not_run",
        "fixed_geometry_ri_sweep_required": True,
    }
    pd.DataFrame([request]).to_csv(root / "PHYSICS_VERIFICATION_REQUESTS.csv", index=False)
    pd.DataFrame([common]).to_csv(root / "selected_designs.csv", index=False)
    (root / "INVERSE_BENCHMARK_MANIFEST.json").write_text(json.dumps({
        "experiment": "matched_surrogate_query_inverse_design",
        "evidence_class": "software_only", "scientific_submission_ready": False,
        "outputs_sha256": {
            name: sha256_file(root / name) for name in (
                "PHYSICS_VERIFICATION_REQUESTS.csv", "selected_designs.csv"
            )
        },
    }), encoding="utf-8")
    plan = tmp_path / "plan"
    prepare_comsol_campaign(
        root, model, config, plan,
        ri_span=0.04, ri_points=5, solver_version="FAKE-NOT-COMSOL",
        normalization_scales={
            "sensitivity_nm_per_riu": 100.0,
            "fom_per_riu": 10.0, "lambda_res_nm": 20.0,
        },
    )
    pilot = tmp_path / "fake_pilot"
    pilot.mkdir()
    (pilot / "COMSOL_MODEL_PILOT_READINESS.json").write_text(
        json.dumps({"software_fixture": True}), encoding="utf-8",
    )
    return plan, root, model, config, pilot / "COMSOL_MODEL_PILOT_READINESS.json"


def _fake_solver(model, config, geometries, *, kind: str = "valid"):
    assert len(geometries) == 4
    wavelength = np.linspace(550.0, 650.0, 101)
    result = []
    for index, geometry in enumerate(geometries):
        center = 600.0 + 400.0 * (geometry.analyte_ri - 1.33)
        if kind == "no_coupling":
            center = 600
        elif kind == "nonmonotonic" and index == 1:
            center = 585
        elif kind == "unstable_repeat" and index == 3:
            center = 603
        loss = 1 + 10.0 / (1.0 + ((wavelength - center) / 9.0) ** 2)
        metrics = extract_metrics(wavelength, loss)
        status = "failed: synthetic eigenmode solver error" if kind == "failure" and index == 2 else "ok"
        entry = {"sample_id": index, **geometry.__dict__, "status": status}
        if status == "ok":
            entry.update({
                "wavelength_nm": ",".join(f"{v:.6f}" for v in wavelength),
                "loss_db_per_cm": ",".join(f"{v:.6f}" for v in loss),
                "lambda_res_nm": float(metrics.lambda_res_nm),
                "fwhm_nm": float(metrics.fwhm_nm),
            })
        result.append(entry)
    return pd.DataFrame(result)


def _probe(tmp_path: Path, *, kind: str = "valid"):
    files = _files(tmp_path)
    output = tmp_path / "response"
    result = probe_comsol_ri_response(
        *files, output, min_shift_nm=1.0, max_repeat_nm=0.5,
        min_diagnostic_r2=0.90,
        runner=lambda m, c, g: _fake_solver(m, c, g, kind=kind),
        check_pilot=lambda *args: {"operational_pilot_verified": True},
    )
    return files, output, result


def test_mock_four_call_response_is_consistent_but_never_real_physics(tmp_path: Path):
    files, output, result = _probe(tmp_path)
    assert result["solver_called"]
    assert result["numerical_ri_response_consistent"]
    assert result["ready_for_full_solver_run"] is False
    assert result["observed_backend"] == "software_fixture"
    assert result["scientific_submission_ready"] is False
    assert result["physical_correctness_verified"] is False
    assert len(result["frozen_samples"]) == 4
    assert [x["frozen_sample_id"] for x in result["frozen_samples"]] == [0, 2, 4, 2]
    assert len(pd.read_csv(output / "RI_PROBE_SPECTRA.csv")) == 4
    assert (output / "RI_PROBE_PROTOCOL.json").is_file()
    assert result["diagnostic"]["low_to_high_resonance_shift_nm"] == pytest.approx(16.0, abs=0.1)
    with pytest.raises(ValueError, match="observed_backend"):
        validate_ri_response_report(
            output / "COMSOL_RI_RESPONSE_READINESS.json", *files[:4], files[4],
        )


@pytest.mark.parametrize("kind,expected", [
    ("no_coupling", "shift"),
    ("nonmonotonic", "monotonically"),
    ("unstable_repeat", "Repeated center"),
    ("failure", "failed or were incomplete"),
])
def test_bad_ri_probe_never_claims_operational_readiness(tmp_path: Path, kind, expected):
    _, output, result = _probe(tmp_path, kind=kind)
    assert result["numerical_ri_response_consistent"] is False
    assert result["ready_for_full_solver_run"] is False
    assert result["solver_called"] is True
    assert any(expected.lower() in x.lower() for x in result["errors"])
    assert (output / "COMSOL_RI_RESPONSE_READINESS.json").is_file()
    assert (output / "RI_PROBE_PROTOCOL.json").is_file()


def test_invalid_numeric_thresholds_refused_before_executing_any_solver(tmp_path: Path):
    plan, bench, model, cfg, pilot = _files(tmp_path)
    for values in [
        {"min_shift_nm": 0, "max_repeat_nm": 0.5},
        {"min_shift_nm": 1.0, "max_repeat_nm": -0.1},
        {"min_shift_nm": 1.0, "max_repeat_nm": 0.5, "min_diagnostic_r2": 1.1},
    ]:
        with pytest.raises(ValueError):
            probe_comsol_ri_response(
                plan, bench, model, cfg, pilot, tmp_path / "out",
                runner=_fake_solver,
                check_pilot=lambda *a: {"operational_pilot_verified": True},
                **values,
            )
    assert not (tmp_path / "out").exists()


def test_original_pilot_genuine_source_gate_fails_before_solving(tmp_path: Path):
    files = _files(tmp_path)
    called = []

    def forbidden(*args):
        called.append(1)
        raise AssertionError("must not execute when prior pilot was software-only")

    with pytest.raises(ValueError, match="Missing genuine pilot readiness"):
        probe_comsol_ri_response(
            *files, tmp_path / "out", min_shift_nm=1.0, max_repeat_nm=0.5,
            runner=forbidden,
        )
    assert not called
    assert not (tmp_path / "out").exists()


def test_tampered_frozen_model_does_not_start_ri_probe(tmp_path: Path):
    plan, bench, model, cfg, pilot = _files(tmp_path)
    model.write_bytes(model.read_bytes() + b"ALTERED")
    with pytest.raises(ValueError, match="model_sha256"):
        probe_comsol_ri_response(
            plan, bench, model, cfg, pilot, tmp_path / "out",
            min_shift_nm=1.0, max_repeat_nm=0.5,
            runner=_fake_solver,
            check_pilot=lambda *a: {"operational_pilot_verified": True},
        )
    assert not (tmp_path / "out").exists()


def test_preregistered_thresholds_and_solver_spectra_are_sha256_bound(tmp_path: Path):
    _, output, result = _probe(tmp_path)
    manifest_path = output / "COMSOL_RI_RESPONSE_READINESS.json"
    protocol = output / "RI_PROBE_PROTOCOL.json"
    assert sha256_file(protocol) == result["preregistered_protocol_sha256"]
    protocol.write_text(protocol.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["observed_backend"] = "comsol_mph_api"
    data["ready_for_full_solver_run"] = True
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    # The protocol and status cannot be forged into source-verified physical
    # evidence merely by changing a JSON flag.
    with pytest.raises(ValueError, match="pre-run protocol hash mismatch"):
        validate_ri_response_report(
            manifest_path, *_files_for_validator(tmp_path),
        )


def _files_for_validator(tmp_path: Path):
    return (tmp_path / "plan", tmp_path / "benchmark",
            tmp_path / "FAKE_NOT_COMSOL.mph", tmp_path / "non_template_config.yaml",
            tmp_path / "fake_pilot" / "COMSOL_MODEL_PILOT_READINESS.json")


def test_production_cli_requires_ri_probe_report_before_any_run(tmp_path: Path):
    plan, bench, model, cfg, pilot = _files(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_comsol_verification_campaign.py"
    base = [
        sys.executable, str(script), "run-batched",
        "--plan", str(plan), "--benchmark", str(bench), "--model", str(model),
        "--config", str(cfg), "--out", str(tmp_path / "no_run"),
        "--readiness-report", str(pilot),
    ]
    no_response = subprocess.run(base, capture_output=True, text=True, check=False)
    assert no_response.returncode == 2
    assert not (tmp_path / "no_run").exists()
    fake = tmp_path / "RI_fake.json"
    fake.write_text("{}", encoding="utf-8")
    with_response = subprocess.run(
        [*base, "--ri-response-report", str(fake)],
        capture_output=True, text=True, check=False,
    )
    assert with_response.returncode != 0
    assert not (tmp_path / "no_run").exists()
