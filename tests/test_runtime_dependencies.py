from __future__ import annotations

import sys
from pathlib import Path

from sprpcf.runtime.dependencies import (
    DEPENDENCY_GROUPS,
    candidate_project_pythons,
    dependency_report,
    probe_modules,
    recommended_install_command,
)


def test_dependency_groups_cover_all_optional_runtime_surfaces() -> None:
    assert {"desktop", "edge", "dashboard", "hardware", "onnx", "comsol", "xai"}.issubset(DEPENDENCY_GROUPS)
    assert "PySide6" in DEPENDENCY_GROUPS["desktop"].modules
    assert "tensorflow" in DEPENDENCY_GROUPS["edge"].modules
    assert DEPENDENCY_GROUPS["desktop"].required is True


def test_probe_modules_reports_success_and_missing_modules() -> None:
    assert probe_modules(sys.executable, ["json", "pathlib"]) == []
    missing = probe_modules(sys.executable, ["definitely_missing_sprpcf_test_module"])
    assert missing
    assert missing[0]["module"] == "definitely_missing_sprpcf_test_module"


def test_dependency_report_contains_actionable_install_commands() -> None:
    report = {row["key"]: row for row in dependency_report()}
    assert "pip install -e" in str(report["desktop"]["install"])
    assert ".[edge]" in str(report["edge"]["install"])
    assert isinstance(report["desktop"]["missing"], list)


def test_candidate_project_pythons_always_includes_current_interpreter(tmp_path: Path) -> None:
    candidates = candidate_project_pythons(tmp_path)
    assert Path(sys.executable) in candidates


def test_install_hint_targets_requested_extra() -> None:
    base = recommended_install_command("desktop", "python")
    edge = recommended_install_command("edge", "python")
    assert 'pip install -e "."' in base
    assert 'pip install -e ".[edge]"' in edge
