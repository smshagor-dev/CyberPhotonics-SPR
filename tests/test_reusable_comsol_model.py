"""Reusable COMSOL model lifecycle tests.

ALL runs use an injected, fictional MPh implementation and Lorentzian spectra.
They cannot establish genuine COMSOL execution, physics or journal evidence.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sprpcf.simulation.comsol_sweep import ComsolModelSession
from sprpcf.simulation.resumable_campaign import (
    inspect_campaign_checkpoint, run_checkpointed_campaign,
)
from test_resumable_campaign import _campaign, _fake_solver


class DummyModel:
    def __init__(self, parent):
        self.parent = parent
        self.ri = 1.33

    def parameter(self, key, value):
        if key == "analyte_ri":
            self.ri = float(value)

    def solve(self, study):
        assert study == "test-only"
        self.parent.solve_count += 1

    def evaluate(self, expression):
        w = np.linspace(550.0, 650.0, 101)
        center = 600.0 + 400.0 * (self.ri - 1.33)
        losses = 1 + 10 / (1 + ((w - center) / 9.0) ** 2)
        return w if expression == "lambda" else losses


class DummyClient:
    def __init__(self, tracker):
        self.tracker = tracker

    def load(self, path):
        self.tracker.model_load_count += 1
        assert Path(path).name == "fake_test_model.mph"
        return DummyModel(self.tracker)

    def remove(self, model):
        assert isinstance(model, DummyModel)
        self.tracker.remove_count += 1


class Counter:
    def __init__(self):
        self.start_count = 0
        self.model_load_count = 0
        self.remove_count = 0
        self.solve_count = 0


def _patch_mph(monkeypatch, tracker):
    import sprpcf.simulation.comsol_sweep as sweep

    def start():
        tracker.start_count += 1
        return DummyClient(tracker)

    monkeypatch.setattr(sweep, "_load_mph", lambda: SimpleNamespace(start=start))


def test_four_batches_load_one_comsol_model_and_close_only_own_model(tmp_path, monkeypatch):
    inputs = _campaign(tmp_path)
    tracker = Counter()
    _patch_mph(monkeypatch, tracker)
    out = tmp_path / "run"
    result = run_checkpointed_campaign(
        *inputs, out, batch_size=8, reuse_loaded_model=True,
    )
    assert result["scheduled_solver_evaluations"] == 30
    assert result["successful_solver_evaluations"] == 30
    assert result["failed_solver_evaluations"] == 0
    assert result["model_session_mode"] == "reused_loaded_model"
    assert tracker.start_count == 1
    assert tracker.model_load_count == 1
    assert tracker.remove_count == 1
    assert tracker.solve_count == 30
    assert len(result["chunk_evidence"]) == 4
    assert result["native_solver_log_captured"] is False
    assert result["scientific_submission_ready"] is False
    inspected = inspect_campaign_checkpoint(*inputs, out)
    assert inspected["model_session_mode"] == "reused_loaded_model"
    assert inspected["remaining_solver_evaluations"] == 0
    contract = json.loads((out / "checkpoint_manifest.json").read_text())
    assert contract["model_session_mode"] == "reused_loaded_model"
    assert len(list((out / "solver_sessions").glob("*.start.json"))) == 1


def test_resume_creates_new_model_session_and_does_not_repeat_committed_samples(
    tmp_path, monkeypatch,
):
    inputs = _campaign(tmp_path)
    tracker = Counter()
    _patch_mph(monkeypatch, tracker)
    original = ComsolModelSession.run
    attempts = []

    def interrupt_after_committed(self, model_path, config_path, geometries):
        attempts.append(len(geometries))
        if len(attempts) == 2:
            raise RuntimeError("software-only interrupted COMSOL session")
        return original(self, model_path, config_path, geometries)

    monkeypatch.setattr(ComsolModelSession, "run", interrupt_after_committed)
    out = tmp_path / "run"
    with pytest.raises(RuntimeError, match="interrupted COMSOL session"):
        run_checkpointed_campaign(
            *inputs, out, batch_size=7, reuse_loaded_model=True,
        )
    assert attempts == [7, 7]
    assert tracker.model_load_count == 1
    assert tracker.remove_count == 1
    assert tracker.solve_count == 7
    partial = inspect_campaign_checkpoint(*inputs, out)
    assert partial["committed_solver_evaluations"] == 7
    assert partial["remaining_solver_evaluations"] == 23
    assert partial["model_session_mode"] == "reused_loaded_model"
    assert len(partial["session_evidence"]) == 1
    assert partial["session_evidence"][0]["status"] == "interrupted"
    with pytest.raises(ValueError, match="Cannot resume changed"):
        run_checkpointed_campaign(*inputs, out, batch_size=7, resume=True)
    # Wrong mode must be refused before another expensive model load.
    assert tracker.model_load_count == 1
    monkeypatch.setattr(ComsolModelSession, "run", original)
    done = run_checkpointed_campaign(
        *inputs, out, batch_size=7, reuse_loaded_model=True, resume=True,
    )
    assert done["scheduled_solver_evaluations"] == 30
    assert tracker.model_load_count == 2
    assert tracker.remove_count == 2
    assert tracker.start_count == 2
    assert tracker.solve_count == 30  # prior seven were not rerun
    assert len(done["session_evidence"]) == 2
    assert [s["status"] for s in done["session_evidence"]] == [
        "interrupted", "completed",
    ]
    assert done["scientific_submission_ready"] is False


def test_reusable_mode_refuses_injected_software_runner(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "no_fake_comsol"
    with pytest.raises(ValueError, match="genuine mph backend"):
        run_checkpointed_campaign(
            *inputs, out, batch_size=7, reuse_loaded_model=True,
            runner=_fake_solver,
        )
    assert not out.exists()


def test_single_model_session_rejects_path_switch_and_use_after_close(tmp_path, monkeypatch):
    inputs = _campaign(tmp_path)
    tracker = Counter()
    _patch_mph(monkeypatch, tracker)
    model, config = inputs[2], inputs[3]
    with ComsolModelSession(model, config) as session:
        with pytest.raises(ValueError, match="cannot switch"):
            session.run(tmp_path / "other.mph", config, [])
        with pytest.raises(ValueError, match="cannot switch"):
            session.run(model, tmp_path / "other.yaml", [])
    assert tracker.remove_count == 1
    with pytest.raises(RuntimeError, match="No active loaded"):
        session.run(model, config, [])


def test_per_batch_legacy_fixture_contract_unchanged(tmp_path):
    inputs = _campaign(tmp_path)
    out = tmp_path / "legacy"
    result = run_checkpointed_campaign(
        *inputs, out, batch_size=13, runner=_fake_solver,
    )
    contract = json.loads((out / "checkpoint_manifest.json").read_text())
    assert "model_session_mode" not in contract
    assert result["model_session_mode"] == "per_batch_load"
    assert inspect_campaign_checkpoint(*inputs, out)["model_session_mode"] == "per_batch_load"
    assert result["real_comsol_backend_invoked"] is False
