from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import venv
from pathlib import Path

CRITICAL_IMPORTS = (
    "sprpcf",
    "sprpcf.simulation.comsol_sweep",
    "sprpcf.ml.multiobjective",
    "sprpcf.validation.closed_loop",
    "sprpcf.validation.campaign",
    "sprpcf.validation.preflight",
    "sprpcf.edge.hardware",
    "sprpcf.evidence.qualification",
    "sprpcf.publication.evidence",
    "sprpcf.publication.submission",
    "sprpcf.publication.finalization",
    "sprpcf.utils.readiness",
)


def _python_path(venv_dir: Path) -> Path:
    if (venv_dir / "Scripts" / "python.exe").is_file():
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _pip_path(venv_dir: Path) -> Path:
    if (venv_dir / "Scripts" / "pip.exe").is_file():
        return venv_dir / "Scripts" / "pip.exe"
    return venv_dir / "bin" / "pip"


def smoke_test_wheel(wheel: Path, expected_version: str | None = None) -> dict[str, str]:
    wheel = wheel.resolve()
    if not wheel.is_file() or wheel.suffix != ".whl":
        raise ValueError(f"Wheel not found: {wheel}")

    with tempfile.TemporaryDirectory(prefix="sprpcf-wheel-smoke-") as temp:
        root = Path(temp)
        venv_dir = root / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True, clear=True).create(venv_dir)
        python = _python_path(venv_dir)
        pip = _pip_path(venv_dir)
        subprocess.run(
            [str(pip), "install", "--no-deps", "--force-reinstall", str(wheel)],
            check=True,
            cwd=root,
        )

        imports = "; ".join(f"import {name}" for name in CRITICAL_IMPORTS)
        code = (
            "import importlib.metadata as metadata, json, pathlib, sprpcf; "
            + imports
            + "; print(json.dumps({'version': metadata.version('sprpcf'), "
            "'module_file': str(pathlib.Path(sprpcf.__file__).resolve())}))"
        )
        result = subprocess.run(
            [str(python), "-I", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            cwd=root,
        )
        payload = json.loads(result.stdout.strip())
        version = str(payload["version"])
        module_file = str(payload["module_file"])
        if expected_version is not None and version != expected_version:
            raise RuntimeError(f"Installed wheel version {version!r} != expected {expected_version!r}")
        if "site-packages" not in module_file.replace("\\", "/"):
            raise RuntimeError(f"Wheel smoke import did not resolve from site-packages: {module_file}")
        return {"version": version, "module_file": module_file, "wheel": str(wheel)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and smoke-import a built CyberPhotonics-SPR wheel.")
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--expected-version")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    report = smoke_test_wheel(wheel=args.wheel, expected_version=args.expected_version)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
