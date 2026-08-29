from __future__ import annotations

import argparse

import main


def test_main_without_subcommand_launches_native_desktop(monkeypatch) -> None:
    calls: list[object] = []

    def fake_desktop(args=None):
        calls.append(args)

    monkeypatch.setattr(main, "launch_desktop", fake_desktop)
    main.main([])

    assert calls == [None]


def test_dashboard_alias_launches_native_desktop(monkeypatch) -> None:
    calls: list[object] = []

    def fake_desktop(args=None):
        calls.append(args)

    monkeypatch.setattr(main, "launch_desktop", fake_desktop)
    main.main(["dashboard"])

    assert len(calls) == 1
    assert isinstance(calls[0], argparse.Namespace)
    assert calls[0].command == "dashboard"


def test_gui_subcommand_launches_same_native_desktop(monkeypatch) -> None:
    calls: list[object] = []

    monkeypatch.setattr(main, "launch_desktop", lambda args=None: calls.append(args))
    main.main(["gui"])

    assert len(calls) == 1
    assert calls[0].command == "gui"
