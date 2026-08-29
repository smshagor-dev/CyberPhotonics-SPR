"""Native desktop control center for CyberPhotonics-SPR.

The desktop stack is imported lazily so missing GUI dependencies can be
recovered through an existing project virtual environment before Qt imports.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from sprpcf.runtime.dependencies import (
    DEPENDENCY_GROUPS,
    candidate_project_pythons,
    missing_modules,
    probe_modules,
    recommended_install_command,
)


__all__ = ["ResponsiveControlCenter", "launch_desktop"]


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _PROJECT_ROOT / "src"
_REEXEC_GUARD = "SPRPCF_DESKTOP_REEXEC"


def _project_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(_SRC_ROOT) if not pythonpath else f"{_SRC_ROOT}{os.pathsep}{pythonpath}"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    return env


def _desktop_missing() -> list[str]:
    return missing_modules(DEPENDENCY_GROUPS["desktop"].modules)


def _ready_alternate_python() -> Path | None:
    current = Path(sys.executable).resolve()
    env = _project_env()
    for candidate in candidate_project_pythons(_PROJECT_ROOT):
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if resolved == current:
            continue
        if not probe_modules(candidate, DEPENDENCY_GROUPS["desktop"].modules, env=env):
            return candidate
    return None


def _repair_message(missing: list[str]) -> str:
    install = recommended_install_command("desktop", sys.executable)
    names = ", ".join(missing) if missing else "one or more desktop dependencies"
    return (
        "CyberPhotonics-SPR desktop environment is incomplete.\n"
        f"Missing: {names}\n"
        f"Repair the active environment with:\n  {install}\n"
        "Then run: python main.py"
    )


def _load_shell():
    try:
        from . import shell
    except ModuleNotFoundError as exc:
        missing = [exc.name or "unknown module"]
        raise SystemExit(_repair_message(missing)) from None
    return shell


def launch_desktop() -> int:
    missing = _desktop_missing()
    if missing and os.environ.get(_REEXEC_GUARD) != "1":
        alternate = _ready_alternate_python()
        if alternate is not None:
            env = _project_env()
            env[_REEXEC_GUARD] = "1"
            completed = subprocess.run(
                [str(alternate), str(_PROJECT_ROOT / "main.py"), "gui"],
                cwd=_PROJECT_ROOT,
                env=env,
                check=False,
            )
            return int(completed.returncode)
    if missing:
        raise SystemExit(_repair_message(missing))
    return _load_shell().launch_desktop()


def __getattr__(name: str) -> Any:
    if name == "ResponsiveControlCenter":
        return _load_shell().ResponsiveControlCenter
    if name == "launch_desktop":
        return launch_desktop
    raise AttributeError(name)
