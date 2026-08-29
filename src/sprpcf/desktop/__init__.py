"""Native desktop control center for CyberPhotonics-SPR.

The desktop stack is imported lazily so a missing GUI dependency produces a
clear repair instruction instead of an opaque ModuleNotFoundError traceback.
"""

from __future__ import annotations

from typing import Any


__all__ = ["ResponsiveControlCenter", "launch_desktop"]


def _load_shell():
    try:
        from . import shell
    except ModuleNotFoundError as exc:
        missing = exc.name or "unknown module"
        raise RuntimeError(
            "CyberPhotonics-SPR desktop dependencies are incomplete. "
            f"Missing module: {missing}. Repair the active environment with: "
            'python -m pip install -e "."'
        ) from exc
    return shell


def launch_desktop() -> int:
    return _load_shell().launch_desktop()


def __getattr__(name: str) -> Any:
    if name == "ResponsiveControlCenter":
        return _load_shell().ResponsiveControlCenter
    if name == "launch_desktop":
        return launch_desktop
    raise AttributeError(name)
