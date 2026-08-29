from __future__ import annotations

import subprocess

import main


def test_main_without_subcommand_launches_control_center(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    main.main([])

    assert len(calls) == 1
    assert calls[0][1:4] == ["-m", "streamlit", "run"]
    assert calls[0][-1].replace("\\", "/").endswith("src/sprpcf/dashboard/app.py")
    assert "8501" in calls[0]
    assert "localhost" in calls[0]


def test_explicit_dashboard_subcommand_uses_same_control_center(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, check):
        calls.append(command)
        assert check is True
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    main.main(["dashboard", "--port", "8510", "--host", "127.0.0.1"])

    assert len(calls) == 1
    assert calls[0][-1].replace("\\", "/").endswith("src/sprpcf/dashboard/app.py")
    assert "8510" in calls[0]
    assert "127.0.0.1" in calls[0]
