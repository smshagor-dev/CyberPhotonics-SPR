"""Synthetic tests for a *numerical* reload-vs-reuse COMSOL mode audit.

All solver calls below are TEST-ONLY injected substitutes. No real COMSOL
license, loaded model, physics or genuine sensor output is present.
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
from sprpcf.simulation.resumable_campaign import (
    inspect_campaign_checkpoint, run_checkpointed_campaign,
)
from sprpcf.simulation.session_provenance import bind_native_session_log
from sprpcf.validation.comsol_mode_equivalence import (
    compare_comsol_mode_results, freeze_mode_comparison_protocol,
)
from test_resumable_campaign import _campaign

TOLERANCES = {
    "max_wavelength_grid_delta_nm": 0.01,
    "max_resonance_delta_nm": 0.15,
    "max_fwhm_delta_nm": 0.15,
    "max_loss_delta_db_per_cm": 0.002,
}


def _solver(_model, _config, geoms, *, shift_nm=0.0, bad=False):
    wave = np.linspace(550.0, 650.0, 101)
    rows = []
    for i, geom in enumerate(geoms):
        if bad and i == 0:
            rows.append({
                "sample_id": i, "status": "failed: deliberate synthetic solver failure",
                **geom.__dict__,
            })
            continue
        peak = 600.0 + 400.0 * (geom.analyte_ri - 1.33) + shift_nm
        loss = 1 + 10 / (1 + ((wave - peak) / 9.0) ** 2)
        metrics = extract_metrics(wave, loss)
        rows.append({
            "sample_id": i, "status": "ok", **geom.__dict__,
            "wavelength_nm": ",".join(format(x, ".17g") for x in wave),
            "loss_db_per_cm": ",".join(format(x, ".17g") for x in loss),
            "lambda_res_nm": metrics.lambda_res_nm, "fwhm_nm": metrics.fwhm_nm,
        })
    return pd.DataFrame(rows)


def _two_synthetic_runs(tmp_path, monkeypatch, *, reuse_shift_nm=0.0,
                        reuse_has_failure=False):
    import sprpcf.simulation.resumable_campaign as campaign

    inputs = _campaign(tmp_path)
    reload_dir, reuse_dir = tmp_path / "reload", tmp_path / "reuse"
    monkeypatch.setattr(
        campaign, "run_comsol_geometries",
        lambda m, c, g: _solver(m, c, g),
    )
    run_checkpointed_campaign(*inputs, reload_dir, batch_size=7)

    class FakeLoadedMph:
        def __init__(self, _model, _config):
            self.closed = False

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.closed = True

        def run(self, m, c, geoms):
            return _solver(
                m, c, geoms, shift_nm=reuse_shift_nm,
                bad=reuse_has_failure,
            )

    monkeypatch.setattr(campaign, "ComsolModelSession", FakeLoadedMph)
    run_checkpointed_campaign(
        *inputs, reuse_dir, batch_size=7, reuse_loaded_model=True,
    )
    assert inspect_campaign_checkpoint(*inputs, reload_dir)["model_session_mode"] == "per_batch_load"
    assert inspect_campaign_checkpoint(*inputs, reuse_dir)["model_session_mode"] == "reused_loaded_model"
    return inputs, reload_dir, reuse_dir


def _protocol(inputs, out):
    return freeze_mode_comparison_protocol(
        *inputs, out, tolerances=TOLERANCES
    )


def _compare(inputs, protocol_dir, reload, reuse, out):
    return compare_comsol_mode_results(
        *inputs, protocol_dir / "COMSOL_MODE_COMPARISON_PROTOCOL.json",
        reload, reuse, out,
    )


def test_matched_synthetic_spectra_are_numerically_equal_but_unqualified(
    tmp_path, monkeypatch,
):
    inputs, reload, reuse = _two_synthetic_runs(tmp_path, monkeypatch)
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    report = _compare(inputs, protocol, reload, reuse, tmp_path / "audit")
    assert report["paired_evaluations"] == 30
    assert report["paired_passed"] == 30
    assert report["paired_failed"] == 0
    assert report["all_paired_spectra_numerically_equivalent"] is True
    assert report["native_session_logs_complete"] is False
    assert report["operational_evidence_complete"] is False
    assert report["source_authenticity_verified"] is False
    assert report["physical_correctness_verified"] is False
    assert report["scientific_submission_ready"] is False
    rows = pd.read_csv(tmp_path / "audit" / "COMSOL_MODE_EQUIVALENCE_ROWS.csv")
    assert len(rows) == 30
    assert rows["passed"].all()
    assert rows["max_loss_delta_db_per_cm"].max() == 0
    assert rows["resonance_delta_nm"].max() == 0


def test_reused_model_drift_detected_at_all_paired_samples(tmp_path, monkeypatch):
    inputs, reload, reuse = _two_synthetic_runs(
        tmp_path, monkeypatch, reuse_shift_nm=0.55,
    )
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    report = _compare(inputs, protocol, reload, reuse, tmp_path / "audit")
    assert report["paired_evaluations"] == 30
    assert report["paired_failed"] > 0
    assert not report["all_paired_spectra_numerically_equivalent"]
    assert not report["operational_evidence_complete"]
    rows = pd.read_csv(tmp_path / "audit" / "COMSOL_MODE_EQUIVALENCE_ROWS.csv")
    assert rows["failure_reason"].str.contains("tolerance_exceeded").any()
    assert rows["max_loss_delta_db_per_cm"].max() > TOLERANCES["max_loss_delta_db_per_cm"]


def test_failed_solver_rows_always_fail_numeric_equivalence(tmp_path, monkeypatch):
    inputs, reload, reuse = _two_synthetic_runs(
        tmp_path, monkeypatch, reuse_has_failure=True,
    )
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    report = _compare(inputs, protocol, reload, reuse, tmp_path / "audit")
    assert report["paired_failed"] == 5  # first row of each seven-point batch
    assert not report["all_paired_spectra_numerically_equivalent"]
    rows = pd.read_csv(tmp_path / "audit" / "COMSOL_MODE_EQUIVALENCE_ROWS.csv")
    assert (rows["failure_reason"] == "one_or_both_solvers_failed").sum() == 5


def test_cannot_swap_run_modes_or_reuse_same_directory(tmp_path, monkeypatch):
    inputs, reload, reuse = _two_synthetic_runs(tmp_path, monkeypatch)
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    with pytest.raises(ValueError, match="Wrong COMSOL reload/reuse lifecycle"):
        _compare(inputs, protocol, reuse, reload, tmp_path / "no_audit")
    with pytest.raises(ValueError, match="two distinct execution"):
        _compare(inputs, protocol, reload, reload, tmp_path / "no_audit")
    assert not (tmp_path / "no_audit").exists()


def test_tampered_checkpoint_or_model_rejected_before_output(tmp_path, monkeypatch):
    inputs, reload, reuse = _two_synthetic_runs(tmp_path, monkeypatch)
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    sweep = reuse / "sweeps.csv"
    sweep.write_bytes(sweep.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="immutable checkpoint evidence"):
        _compare(inputs, protocol, reload, reuse, tmp_path / "no_output")
    assert not (tmp_path / "no_output").exists()


@pytest.mark.parametrize("invalid", [
    {"max_wavelength_grid_delta_nm": 0.0},
    {"max_loss_delta_db_per_cm": float("nan")},
    {"max_fwhm_delta_nm": float("inf")},
    {"max_resonance_delta_nm": -0.5},
])
def test_bad_threshold_refused_before_writing_protocol(tmp_path, invalid):
    inputs = _campaign(tmp_path)
    thresholds = {**TOLERANCES, **invalid}
    with pytest.raises(ValueError, match="finite positive"):
        freeze_mode_comparison_protocol(
            *inputs, tmp_path / "invalid", tolerances=thresholds,
        )
    assert not (tmp_path / "invalid").exists()


def test_source_plan_mutation_refuses_protocol_or_comparison(tmp_path, monkeypatch):
    inputs, reload, reuse = _two_synthetic_runs(tmp_path, monkeypatch)
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    model = inputs[2]
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="model_sha256"):
        _compare(inputs, protocol, reload, reuse, tmp_path / "never")
    assert not (tmp_path / "never").exists()


def test_logs_are_separate_and_required_for_operational_evidence(
    tmp_path, monkeypatch,
):
    inputs, reload, reuse = _two_synthetic_runs(tmp_path, monkeypatch)
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    for label, path in (("reload", reload), ("reuse", reuse)):
        checked = inspect_campaign_checkpoint(*inputs, path)
        checkpoint = json.loads((path / "checkpoint_manifest.json").read_text())
        for session in checked["session_evidence"]:
            log = tmp_path / f"{label}_{session['session_id']}.log"
            log.write_text("THIS IS A SYNTHETIC LOG, NOT REAL COMSOL " +
                           label + session["session_id"], encoding="utf-8")
            bind_native_session_log(
                path, session["session_id"], log, checkpoint,
                checked["chunk_evidence"],
            )
    report = _compare(inputs, protocol, reload, reuse, tmp_path / "audit")
    assert report["all_paired_spectra_numerically_equivalent"] is True
    assert report["native_session_logs_complete"] is True
    assert report["operational_evidence_complete"] is True
    # A fake log passes local byte-integrity checks, NOT source authentication.
    assert report["source_authenticity_verified"] is False
    assert report["scientific_submission_ready"] is False


def test_cli_strict_requires_both_numeric_match_and_logs(tmp_path, monkeypatch):
    inputs, reload, reuse = _two_synthetic_runs(tmp_path, monkeypatch)
    protocol = tmp_path / "protocol"
    _protocol(inputs, protocol)
    plan, benchmark, model, config = inputs
    script = Path(__file__).resolve().parents[1] / "scripts" / "audit_comsol_model_modes.py"
    result = subprocess.run([
        sys.executable, str(script), "audit",
        "--plan", str(plan), "--benchmark", str(benchmark),
        "--model", str(model), "--config", str(config),
        "--protocol", str(protocol / "COMSOL_MODE_COMPARISON_PROTOCOL.json"),
        "--reload-run", str(reload), "--reuse-run", str(reuse),
        "--out", str(tmp_path / "audit"), "--strict",
    ], check=False, capture_output=True, text=True)
    assert result.returncode == 2, result.stderr
    report = json.loads(
        (tmp_path / "audit" / "COMSOL_MODE_EQUIVALENCE_AUDIT.json").read_text()
    )
    assert report["all_paired_spectra_numerically_equivalent"] is True
    assert report["native_session_logs_complete"] is False
    assert report["scientific_submission_ready"] is False
