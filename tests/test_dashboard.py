from __future__ import annotations

import subprocess

import pytest


def test_dashboard_tabs_and_command_flags() -> None:
    from sprpcf.ui.dashboard import DASHBOARD_TABS, build_streamlit_command

    assert DASHBOARD_TABS == [
        "Physics-Informed Inverse Design",
        "Explainable AI",
        "Active Learning",
        "Edge Denoising",
    ]
    command = build_streamlit_command(port=8502, host="127.0.0.1")
    assert command[1:4] == ["-m", "streamlit", "run"]
    assert "--server.port" in command
    assert "8502" in command
    assert "--server.address" in command
    assert "127.0.0.1" in command
    assert command[4:8] == ["--server.port", "8502", "--server.address", "127.0.0.1"]
    assert command[8:10] == ["--browser.gatherUsageStats", "false"]


def test_main_dashboard_cli_invokes_streamlit(monkeypatch) -> None:
    import os
    import main

    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)

    main.main(["dashboard", "--port", "8503", "--host", "127.0.0.1"])

    assert len(calls) == 1
    assert calls[0][1:4] == ["-m", "streamlit", "run"]
    assert "8503" in calls[0]
    assert "127.0.0.1" in calls[0]
    assert os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] == "false"


def test_dashboard_surrogate_outputs_are_stable() -> None:
    from sprpcf.ui.dashboard import (
        dispersion_curves,
        edge_latency_snapshot,
        predict_inverse_geometry,
        synthetic_feature_importance,
    )

    prediction = predict_inverse_geometry(900.0, 45.0, 650.0)
    assert prediction.feasible is True
    importance = synthetic_feature_importance(prediction)
    assert set(importance.columns) == {"feature", "importance"}
    assert importance["importance"].sum() == pytest.approx(1.0)

    curves = dispersion_curves(points=12)
    assert {"wavelength_nm", "epsilon_au_real", "epsilon_au_imag", "silica_n"}.issubset(curves.columns)

    snapshot = edge_latency_snapshot(noisy=curves["silica_n"].to_numpy("float32"), model_path=None)
    assert snapshot["used_tflite"] is False
    assert snapshot["latency_ms"] >= 0.0
