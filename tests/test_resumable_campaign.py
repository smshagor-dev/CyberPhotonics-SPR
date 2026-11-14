"""Crash/resume and journal-integrity tests, using deliberately FAKE spectra.

No COMSOL program or license is needed; results are always software fixtures
and cannot be sealed as authentic physics.
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
from sprpcf.simulation.verification_campaign import (
    prepare_comsol_campaign, seal_comsol_campaign,
)
from sprpcf.utils.reproducibility import sha256_file


def _campaign(tmp_path: Path):
    root = tmp_path / "benchmark"
    root.mkdir()
    model = tmp_path / "fake_test_model.mph"
    model.write_bytes(b"NOT COMSOL; TEST FILE ONLY")
    config = tmp_path / "fixture.yaml"
    config.write_text("study: test-only\n", encoding="utf-8")
    requests, selected = [], []
    for i in range(3):
        for method in ("tandem_latent", "uniform_random"):
            common = {
                "design_seed": 7, "method": method, "target_id": i,
                "target_group_id": "geometry-" + str(i),
                "pitch_um": 1.8 + i * 0.1, "d_over_lambda": 0.5,
                "metal_thickness_nm": 40.0, "channel_radius_um": 0.6,
                "analyte_ri": 1.33,
            }
            requests.append({
                **common, "request_id": f"fake-{method}-{i}",
                "target_sensitivity_nm_per_riu": 500.0,
                "target_fom_per_riu": 25.0, "target_lambda_res_nm": 600.0,
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
            n: sha256_file(root / n)
            for n in ("PHYSICS_VERIFICATION_REQUESTS.csv", "selected_designs.csv")
        },
    }
    (root / "INVERSE_BENCHMARK_MANIFEST.json").write_text(json.dumps(manifest), encoding="utf-8")
    plan = tmp_path / "plan"
    prepare_comsol_campaign(
        root, model, config, plan,
        ri_span=0.04, ri_points=5, solver_version="FAKE TEST VERSION",
        normalization_scales={
            "sensitivity_nm_per_riu": 100.0, "fom_per_riu": 10.0,
            "lambda_res_nm": 20.0,
        },
    )
    return plan, root, model, config


def _fake_solver(_model, _config, geometries, *, fail_first: bool = False):
    wavelengths = np.linspace(550, 650, 101)
    rows = []
    for i, geom in enumerate(geometries):
        if fail_first and i == 0:
            rows.append({
                "sample_id": i, "status": "failed: mock eigenmode unavailable",
                **geom.__dict__,
            })
            continue
        resonance = 600.0 + 400.0 * (geom.analyte_ri - 1.33)
        losses = 1 + 10 / (1 + ((wavelengths - resonance) / 9.0) ** 2)
        metrics = extract_metrics(wavelengths, losses)
        rows.append({
            "sample_id": i, "status": "ok", **geom.__dict__,
            "wavelength_nm": ",".join(f"{value:.6f}" for value in wavelengths),
            "loss_db_per_cm": ",".join(f"{value:.6f}" for value in losses),
            "lambda_res_nm": metrics.lambda_res_nm, "fwhm_nm": metrics.fwhm_nm,
        })
    return pd.DataFrame(rows)


def _run(inputs, output, **kwargs):
    return run_checkpointed_campaign(*inputs, output, **kwargs)


def test_batched_fixture_matches_full_stage7_output_contract_and_cannot_seal(tmp_path):
    plan, root, model, config = _campaign(tmp_path)
    out = tmp_path / "execution"
    result = _run((plan, root, model, config), out, batch_size=7, runner=_fake_solver)
    assert result["execution_mode"] == "checkpointed_batches"
    assert result["scheduled_solver_evaluations"] == 30
    assert result["successful_solver_evaluations"] == 30
    assert result["failed_solver_evaluations"] == 0
    assert result["real_comsol_backend_invoked"] is False
    assert result["scientific_submission_ready"] is False
    assert len(result["chunk_evidence"]) == 5
    frame = pd.read_csv(out / "sweeps.csv")
    assert len(frame) == 30
    assert frame["source"].eq("test_fixture").all()
    assert frame["evidence_class"].eq("software_only").all()
    assert result["sweeps_sha256"] == sha256_file(out / "sweeps.csv")
    inspected = inspect_campaign_checkpoint(plan, root, model, config, out)
    assert inspected["contract_valid"] and inspected["remaining_solver_evaluations"] == 0
    solver_log = tmp_path / "dummy.log"
    solver_log.write_text("NO COMSOL EVER RAN", encoding="utf-8")
    with pytest.raises(ValueError, match="Cannot seal injected test-runner"):
        seal_comsol_campaign(
            plan, out, root, model, config, solver_log,
            tmp_path / "missing_refinement", tmp_path / "sealed",
        )


def test_interruption_resumes_only_pending_batches_and_reports_progress(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"
    calls = []

    def break_after_first(model, config, geometries):
        calls.append(len(geometries))
        if len(calls) == 2:
            raise RuntimeError("simulated process interruption")
        return _fake_solver(model, config, geometries)

    with pytest.raises(RuntimeError, match="simulated process interruption"):
        _run(inputs, out, batch_size=7, runner=break_after_first)
    assert calls == [7, 7]
    assert not (out / "execution_manifest.json").exists()
    assert not (out / ".active-comsol-campaign.lock").exists()
    progress = inspect_campaign_checkpoint(*inputs, out)
    assert progress["committed_solver_evaluations"] == 7
    assert progress["remaining_solver_evaluations"] == 23
    assert progress["committed_batches"] == 1
    resumed_calls = []

    def resume_solver(model, config, geometries):
        resumed_calls.append(len(geometries))
        return _fake_solver(model, config, geometries)

    outcome = _run(inputs, out, batch_size=7, resume=True, runner=resume_solver)
    assert resumed_calls == [7, 7, 7, 2]
    assert outcome["scheduled_solver_evaluations"] == 30
    assert len(outcome["chunk_evidence"]) == 5
    assert inspect_campaign_checkpoint(*inputs, out)["remaining_solver_evaluations"] == 0
    with pytest.raises(FileExistsError, match="Already finalized"):
        _run(inputs, out, batch_size=7, resume=True, runner=resume_solver)


def test_modified_completed_checkpoint_refuses_resume_before_new_solver_calls(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"

    def interrupted(model, config, geometries):
        if (out / "chunks" / "batch_00000000_00000007.json").is_file():
            raise RuntimeError("planned midstream failure")
        return _fake_solver(model, config, geometries)

    with pytest.raises(RuntimeError):
        _run(inputs, out, batch_size=7, runner=interrupted)
    first = out / "chunks" / "batch_00000000_00000007.csv"
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    attempted = []

    def should_not_run(*args):
        attempted.append(1)
        raise AssertionError("Tampered batch must not be trusted")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _run(inputs, out, batch_size=7, resume=True, runner=should_not_run)
    assert not attempted


def test_rejects_reconfigured_resume_and_backend_source_switch(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"

    def interrupt(*args):
        raise RuntimeError("runtime unavailable")

    with pytest.raises(RuntimeError):
        _run(inputs, out, batch_size=7, runner=interrupt)
    with pytest.raises(ValueError, match="Cannot resume changed"):
        _run(inputs, out, batch_size=8, resume=True, runner=_fake_solver)
    # A fixture checkpoint may not be resumed as an actual COMSOL campaign.
    with pytest.raises(ValueError, match="Cannot resume changed"):
        _run(inputs, out, batch_size=7, resume=True)
    model = inputs[2]
    model.write_bytes(model.read_bytes() + b"TAMPER")
    with pytest.raises(ValueError, match="model_sha256"):
        _run(inputs, out, batch_size=7, resume=True, runner=_fake_solver)


def test_solver_failures_remain_in_frozen_output_denominator(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"
    report = _run(
        inputs, out, batch_size=7,
        runner=lambda m, c, g: _fake_solver(m, c, g, fail_first=True),
    )
    assert report["scheduled_solver_evaluations"] == 30
    assert report["failed_solver_evaluations"] == 5
    assert report["successful_solver_evaluations"] == 25
    sweep = pd.read_csv(out / "sweeps.csv")
    assert len(sweep) == 30
    assert sweep["status"].str.startswith("failed:").sum() == 5
    assert sweep.loc[sweep["status"] != "ok", "lambda_res_nm"].isna().all()


def test_stale_lock_requires_explicit_acknowledgement(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"

    def fail(*args):
        raise RuntimeError("software interrupt")

    with pytest.raises(RuntimeError):
        _run(inputs, out, batch_size=7, runner=fail)
    lock = out / ".active-comsol-campaign.lock"
    lock.write_text("ORPHAN LOCK", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Campaign lock exists"):
        _run(inputs, out, batch_size=7, resume=True, runner=_fake_solver)
    result = _run(inputs, out, batch_size=7, resume=True,
                  recover_stale_lock=True, runner=_fake_solver)
    assert result["scheduled_solver_evaluations"] == 30
    assert not lock.exists()


def test_orphan_checkpoint_without_matching_hash_manifest_is_rejected(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"

    def fail(*args):
        raise RuntimeError("software interrupt")

    with pytest.raises(RuntimeError):
        _run(inputs, out, batch_size=7, runner=fail)
    (out / "chunks" / "batch_00000000_00000007.csv").write_text(
        "UNCOMMITTED BYTES", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Incomplete checkpoint pair"):
        _run(inputs, out, batch_size=7, resume=True, runner=_fake_solver)


def test_readonly_inspect_cli_does_not_try_to_run_comsol(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "execution"
    _run(inputs, out, batch_size=8, runner=_fake_solver)
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_comsol_verification_campaign.py"
    plan, root, model, config = inputs
    run = subprocess.run(
        [sys.executable, str(script), "inspect",
         "--plan", str(plan), "--benchmark", str(root), "--model", str(model),
         "--config", str(config), "--out", str(out)],
        capture_output=True, text=True, check=False,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout)
    assert result["operation"] == "inspect"
    assert result["committed_solver_evaluations"] == 30
    assert result["remaining_solver_evaluations"] == 0
    assert result["scientific_submission_ready"] is False


def test_sealing_rechecks_all_chunks_and_final_merged_spectra(
    tmp_path, monkeypatch: pytest.MonkeyPatch,
):
    """Mocked 'real' path is scoped to this test, NOT an authentic COMSOL run."""
    import sprpcf.simulation.resumable_campaign as resumable
    import sprpcf.simulation.verification_campaign as legacy

    inputs = _campaign(tmp_path)
    plan, benchmark, model, config = inputs
    monkeypatch.setattr(resumable, "run_comsol_geometries", _fake_solver)
    out = tmp_path / "checkpointed"
    report = _run(inputs, out, batch_size=7)
    assert report["real_comsol_backend_invoked"] is True  # backend entrypoint was patched in test
    assert report["source_authenticity_verified"] is False

    native_log = tmp_path / "fictional_comsol_test.log"
    native_log.write_text("FAKE NATIVE LOG: NO COMSOL WAS ACTUALLY RUN", encoding="utf-8")
    inspected = inspect_campaign_checkpoint(plan, benchmark, model, config, out)
    checkpoint = json.loads((out / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    for item in inspected["session_evidence"]:
        bind_native_session_log(
            out, item["session_id"], native_log, checkpoint, inspected["chunk_evidence"],
        )
    refinement = tmp_path / "fake_refinement"
    refinement.mkdir()
    (refinement / "refinement_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        legacy, "audit_refinement_archive",
        lambda root: {
            "convergence_consistent": True,
            "file_sha256": {"model": sha256_file(model)},
        },
    )
    sealed = seal_comsol_campaign(
        plan, out, benchmark, model, config, native_log, refinement,
        tmp_path / "test_seal",
    )
    assert sealed["physical_correctness_verified"] is False
    assert sealed["source_authenticity_verified"] is False
    first_chunk = out / "chunks" / "batch_00000000_00000007.csv"
    first_chunk.write_text(first_chunk.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpoint SHA-256 mismatch"):
        seal_comsol_campaign(
            plan, out, benchmark, model, config, native_log, refinement,
            tmp_path / "refused_seal",
        )


def test_final_sweeps_rehash_cannot_bypass_checkpointed_seal(tmp_path, monkeypatch):
    import sprpcf.simulation.resumable_campaign as resumable
    import sprpcf.simulation.verification_campaign as legacy

    plan, benchmark, model, config = _campaign(tmp_path)
    monkeypatch.setattr(resumable, "run_comsol_geometries", _fake_solver)
    out = tmp_path / "checkpointed"
    run_checkpointed_campaign(plan, benchmark, model, config, out, batch_size=8)
    # Change only the combined CSV, then update its digest in the top-level
    # execution manifest. The immutable batch journal must still reject it.
    sweeps = out / "sweeps.csv"
    data = pd.read_csv(sweeps)
    data.loc[0, "pitch_um"] += 0.3
    data.to_csv(sweeps, index=False)
    manifest = out / "execution_manifest.json"
    state = json.loads(manifest.read_text(encoding="utf-8"))
    state["sweeps_sha256"] = sha256_file(sweeps)
    manifest.write_text(json.dumps(state), encoding="utf-8")
    native_log = tmp_path / "mock.log"
    native_log.write_text("NOT A REAL COMSOL LOG", encoding="utf-8")
    with pytest.raises(ValueError, match="Checkpointed COMSOL run chunks disagree"):
        seal_comsol_campaign(
            plan, out, benchmark, model, config, native_log,
            tmp_path / "missing_refinement", tmp_path / "never_sealed",
        )



def test_resumed_mock_backend_requires_distinct_native_log_for_every_session(
    tmp_path, monkeypatch,
):
    """Simulated COMSOL entrypoint: NO REAL COMSOL EXECUTED in this regression."""
    import sprpcf.simulation.resumable_campaign as resumable
    import sprpcf.simulation.verification_campaign as legacy
    from sprpcf.simulation.session_provenance import (
        audit_native_session_logs, audit_sealed_session_logs,
    )

    plan, benchmark, model, config = _campaign(tmp_path)
    out = tmp_path / "checkpointed"
    invocations = []

    def interrupted_source(m, c, geometries):
        invocations.append(len(geometries))
        if len(invocations) == 2:
            raise RuntimeError("simulated COMSOL session loss")
        return _fake_solver(m, c, geometries)

    monkeypatch.setattr(resumable, "run_comsol_geometries", interrupted_source)
    with pytest.raises(RuntimeError, match="simulated COMSOL session loss"):
        run_checkpointed_campaign(plan, benchmark, model, config, out, batch_size=7)
    interrupted_state = inspect_campaign_checkpoint(plan, benchmark, model, config, out)
    assert interrupted_state["committed_solver_evaluations"] == 7
    assert len(interrupted_state["session_evidence"]) == 1
    assert interrupted_state["session_evidence"][0]["status"] == "interrupted"
    monkeypatch.setattr(resumable, "run_comsol_geometries", _fake_solver)
    run = run_checkpointed_campaign(plan, benchmark, model, config, out,
                                   batch_size=7, resume=True)
    assert len(run["session_evidence"]) == 2
    assert {s["status"] for s in run["session_evidence"]} == {"completed", "interrupted"}
    inspected = inspect_campaign_checkpoint(plan, benchmark, model, config, out)
    contract = json.loads((out / "checkpoint_manifest.json").read_text(encoding="utf-8"))
    assert {x["session_id"] for x in inspected["chunk_evidence"]} == {
        x["session_id"] for x in inspected["session_evidence"]
    }
    first = inspected["session_evidence"][0]
    first_log = tmp_path / "first_native_fixture.log"
    first_log.write_text("FAKE LOG - FIRST COMSOL SESSION NEVER RAN", encoding="utf-8")
    bind_native_session_log(out, first["session_id"], first_log, contract,
                           inspected["chunk_evidence"])
    with pytest.raises(ValueError, match="Missing or symlinked solver session evidence"):
        audit_native_session_logs(out, contract, inspected["chunk_evidence"])
    second = inspected["session_evidence"][1]
    second_log = tmp_path / "second_native_fixture.log"
    second_log.write_text("FAKE LOG - SECOND COMSOL SESSION NEVER RAN", encoding="utf-8")
    bind_native_session_log(out, second["session_id"], second_log, contract,
                           inspected["chunk_evidence"])
    evidence = audit_native_session_logs(out, contract, inspected["chunk_evidence"])
    assert len(evidence) == 2
    assert all(x["source_authenticity_verified"] is False for x in evidence)
    refinement = tmp_path / "fixture_refinement"
    refinement.mkdir()
    (refinement / "refinement_manifest.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(legacy, "audit_refinement_archive", lambda root: {
        "convergence_consistent": True,
        "file_sha256": {"model": sha256_file(model)},
    })
    sealed = tmp_path / "sealed"
    manifest = seal_comsol_campaign(
        plan, out, benchmark, model, config, second_log, refinement, sealed,
    )
    assert manifest["session_log_provenance_required"] is True
    assert manifest["execution_mode"] == "checkpointed_batches"
    attested = audit_sealed_session_logs(sealed, manifest)
    assert attested["session_count"] == 2
    assert attested["source_authenticity_verified"] is False
    assert attested["physical_correctness_verified"] is False
    first_copied = sealed / "session_logs" / (first["session_id"] + ".log")
    first_copied.write_bytes(first_copied.read_bytes() + b"\nTAMPER")
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        audit_sealed_session_logs(sealed, manifest)


def test_missing_session_native_log_blocks_seal_before_refinement(tmp_path, monkeypatch):
    import sprpcf.simulation.resumable_campaign as resumable
    from sprpcf.simulation.session_provenance import (
        audit_native_session_logs, bind_native_session_log,
    )
    plan, benchmark, model, config = _campaign(tmp_path)
    monkeypatch.setattr(resumable, "run_comsol_geometries", _fake_solver)
    out = tmp_path / "checkpointed"
    run_checkpointed_campaign(plan, benchmark, model, config, out, batch_size=7)
    inspected = inspect_campaign_checkpoint(plan, benchmark, model, config, out)
    contract = json.loads((out / "checkpoint_manifest.json").read_text())
    first_session = inspected["session_evidence"][0]["session_id"]
    log = tmp_path / "native_fixture.log"
    log.write_text("FAKE - NEVER COMSOL", encoding="utf-8")
    with pytest.raises(ValueError, match="missing"):
        seal_comsol_campaign(
            plan, out, benchmark, model, config, log,
            tmp_path / "missing_refinement", tmp_path / "sealed",
        )
    receipt = bind_native_session_log(out, first_session, log, contract,
                                      inspected["chunk_evidence"])
    assert receipt["source_authenticity_verified"] is False
    with pytest.raises(FileExistsError, match="already bound"):
        bind_native_session_log(out, first_session, log, contract,
                               inspected["chunk_evidence"])
    (out / "native_solver_logs" / (first_session + ".log")).write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="receipt mismatch"):
        audit_native_session_logs(out, contract, inspected["chunk_evidence"])


def test_explicit_crash_closure_cannot_override_active_worker_lock(tmp_path):
    from sprpcf.simulation.session_provenance import (
        close_crashed_solver_session, start_solver_session, inspect_session_ledger,
    )
    from sprpcf.simulation.resumable_campaign import _checkpoint_contract
    plan, benchmark, model, config = _campaign(tmp_path)
    contract = _checkpoint_contract(
        plan, json.loads((plan / "campaign_plan.json").read_text(encoding="utf-8")),
        7, True,
    )
    run_dir = tmp_path / "crashed"
    run_dir.mkdir()
    sid = start_solver_session(run_dir, contract, 0)
    lock = run_dir / ".active-comsol-campaign.lock"
    lock.write_text("fake live worker", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Active worker lock"):
        close_crashed_solver_session(run_dir, sid)
    lock.unlink()
    status = close_crashed_solver_session(run_dir, sid)
    assert status["status"] == "manually_closed_after_crash"
    inv = inspect_session_ledger(run_dir, contract, [])
    assert len(inv) == 1
    assert inv[0]["status"] == "manually_closed_after_crash"
    with pytest.raises(ValueError, match="already ended"):
        close_crashed_solver_session(run_dir, sid)


def test_cli_session_binding_and_audit_never_loads_comsol(tmp_path, monkeypatch):
    import sprpcf.simulation.resumable_campaign as resumable
    plan, benchmark, model, config = _campaign(tmp_path)
    monkeypatch.setattr(resumable, "run_comsol_geometries", _fake_solver)
    out = tmp_path / "checkpointed"
    run_checkpointed_campaign(plan, benchmark, model, config, out, batch_size=8)
    inspected = inspect_campaign_checkpoint(plan, benchmark, model, config, out)
    sid = inspected["session_evidence"][0]["session_id"]
    native_log = tmp_path / "not_real_solver.log"
    native_log.write_text("FICTIONAL LOG", encoding="utf-8")
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_comsol_verification_campaign.py"
    # A new process cannot inherit the test-only fake COMSOL function, but
    # bind-log and audit-logs only load the frozen evidence files.
    arguments = [
        "--plan", str(plan), "--benchmark", str(benchmark),
        "--model", str(model), "--config", str(config), "--out", str(out),
    ]
    bind = subprocess.run(
        [sys.executable, str(script), "bind-log", *arguments,
         "--session-id", sid, "--native-solver-log", str(native_log)],
        capture_output=True, text=True, check=False,
    )
    assert bind.returncode == 0, bind.stderr
    assert json.loads(bind.stdout)["source_authenticity_verified"] is False
    audit = subprocess.run(
        [sys.executable, str(script), "audit-logs", *arguments],
        capture_output=True, text=True, check=False,
    )
    assert audit.returncode == 0, audit.stderr
    assert json.loads(audit.stdout)["session_count"] == 1
