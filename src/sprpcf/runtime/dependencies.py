from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DependencyGroup:
    key: str
    label: str
    modules: tuple[str, ...]
    extra: str | None = None
    required: bool = False


DEPENDENCY_GROUPS: dict[str, DependencyGroup] = {
    "desktop": DependencyGroup(
        "desktop",
        "Native desktop",
        ("sprpcf", "numpy", "pandas", "pyarrow", "scipy", "sklearn", "matplotlib", "torch", "yaml", "PySide6"),
        required=True,
    ),
    "edge": DependencyGroup(
        "edge",
        "Edge training / TFLite",
        ("tensorflow",),
        extra="edge",
    ),
    "dashboard": DependencyGroup(
        "dashboard",
        "Legacy Streamlit dashboard",
        ("streamlit",),
        extra="dashboard",
    ),
    "hardware": DependencyGroup(
        "hardware",
        "Serial hardware",
        ("serial",),
        extra="hardware",
    ),
    "onnx": DependencyGroup(
        "onnx",
        "ONNX export/runtime",
        ("onnx", "onnxscript", "onnxruntime"),
        extra="onnx",
    ),
    "comsol": DependencyGroup(
        "comsol",
        "COMSOL bridge",
        ("mph",),
        extra="comsol",
    ),
    "xai": DependencyGroup(
        "xai",
        "SHAP explainability",
        ("shap",),
        extra="xai",
    ),
}


def _import_probe_code(modules: Iterable[str]) -> str:
    quoted = json.dumps(list(modules))
    return (
        "import importlib, json; "
        f"mods={quoted}; "
        "missing=[]; "
        "\nfor name in mods:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append({'module': name, 'error': f'{type(exc).__name__}: {exc}'})\n"
        "print(json.dumps(missing))"
    )


def probe_modules(
    python_executable: str | Path,
    modules: Iterable[str],
    *,
    env: dict[str, str] | None = None,
    timeout: float = 45.0,
) -> list[dict[str, str]]:
    requested = tuple(dict.fromkeys(str(name) for name in modules if str(name).strip()))
    if not requested:
        return []
    try:
        completed = subprocess.run(
            [str(python_executable), "-c", _import_probe_code(requested)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [{"module": "<python>", "error": f"{type(exc).__name__}: {exc}"}]

    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or f"exit code {completed.returncode}"
        return [{"module": "<python>", "error": detail}]
    try:
        payload = json.loads(completed.stdout.strip() or "[]")
    except json.JSONDecodeError:
        return [{"module": "<probe>", "error": completed.stdout.strip() or "invalid probe output"}]
    if not isinstance(payload, list):
        return [{"module": "<probe>", "error": "invalid probe payload"}]
    return [
        {"module": str(item.get("module", "unknown")), "error": str(item.get("error", "import failed"))}
        for item in payload
        if isinstance(item, dict)
    ]


def missing_modules(modules: Iterable[str]) -> list[str]:
    missing: list[str] = []
    for module in modules:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, AttributeError, ValueError):
            spec = None
        if spec is None:
            missing.append(module)
    return missing


def recommended_install_command(group_key: str, python_executable: str | Path | None = None) -> str:
    group = DEPENDENCY_GROUPS[group_key]
    python = str(python_executable or sys.executable)
    target = "." if group.extra is None else f".[{group.extra}]"
    return f'"{python}" -m pip install -e "{target}"'


def dependency_report() -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    for group in DEPENDENCY_GROUPS.values():
        missing = missing_modules(group.modules)
        report.append(
            {
                "key": group.key,
                "label": group.label,
                "required": group.required,
                "available": not missing,
                "missing": missing,
                "install": recommended_install_command(group.key),
            }
        )
    return report


def candidate_project_pythons(project_root: Path) -> list[Path]:
    candidates = [
        Path(sys.executable),
        project_root / ".venv311" / "Scripts" / "python.exe",
        project_root / ".venv" / "Scripts" / "python.exe",
        project_root / ".venv311" / "bin" / "python",
        project_root / ".venv" / "bin" / "python",
    ]
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists() or candidate == Path(sys.executable):
            unique.append(candidate)
    return unique
