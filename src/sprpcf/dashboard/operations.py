from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
SRC_ROOT = PROJECT_ROOT / "src"


@dataclass(frozen=True)
class TaskResult:
    name: str
    command: list[str]
    returncode: int
    output: str
    elapsed_sec: float

    @property
    def success(self) -> bool:
        return self.returncode == 0

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "success": self.success}


@dataclass(frozen=True)
class ArtifactStatus:
    label: str
    path: str
    exists: bool
    size_bytes: int
    modified_utc: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalize_dashboard_arguments(subcommand: str, arguments: Iterable[str]) -> list[str]:
    """Normalize friendly dashboard values to the CLI/runtime contract."""
    values = [str(value) for value in arguments]
    device_flags = {"--device"} if subcommand == "train-edge" else set()
    if subcommand == "run-pipeline":
        device_flags.add("--edge-device")

    for index, value in enumerate(values[:-1]):
        if value in device_flags and values[index + 1].strip().lower() in {"gpu", "cuda", "gpu:0", "/gpu:0"}:
            values[index + 1] = "/GPU:0"
    return values


def _can_import(python_executable: Path | str, module: str) -> bool:
    command = [str(python_executable), "-c", f"import {module}"]
    return subprocess.call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0


def _can_import_all(python_executable: Path | str, modules: Iterable[str]) -> bool:
    return all(_can_import(python_executable, module) for module in modules)


def project_python_executable(required_module: str | Iterable[str] = "sprpcf") -> str:
    """Return a Python executable that can import the project package."""
    modules = (required_module,) if isinstance(required_module, str) else tuple(required_module)
    venv311_python = PROJECT_ROOT / ".venv311" / "Scripts" / "python.exe"
    if venv311_python.exists() and _can_import_all(venv311_python, modules):
        return str(venv311_python)
    if _can_import_all(sys.executable, modules):
        return sys.executable
    return sys.executable


def project_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("PYTHONIOENCODING", "utf-8")
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(SRC_ROOT) if not pythonpath else f"{SRC_ROOT}{os.pathsep}{pythonpath}"
    return env


def build_cli_command(subcommand: str, arguments: Iterable[str] = ()) -> list[str]:
    """Build a dashboard-to-CLI command without shell interpolation."""
    if not subcommand or subcommand.startswith("-"):
        raise ValueError("subcommand must be a non-empty command name")
    normalized = _normalize_dashboard_arguments(subcommand, arguments)
    return [project_python_executable(), "-u", str(MAIN_SCRIPT), subcommand, *normalized]


def run_cli_task(
    name: str,
    subcommand: str,
    arguments: Iterable[str] = (),
    on_output: Callable[[str], None] | None = None,
) -> TaskResult:
    """Run one orchestrator task in an isolated child process and stream combined output."""
    command = build_cli_command(subcommand, arguments)
    env = project_subprocess_env()

    creationflags = 0
    if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        creationflags = subprocess.CREATE_NO_WINDOW

    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
        creationflags=creationflags,
    )
    lines: list[str] = []
    if process.stdout is not None:
        for raw_line in iter(process.stdout.readline, ""):
            line = raw_line.rstrip("\r\n")
            lines.append(line)
            if on_output is not None:
                on_output(line)
        process.stdout.close()
    returncode = process.wait()
    elapsed = time.perf_counter() - started
    return TaskResult(
        name=name,
        command=command,
        returncode=returncode,
        output="\n".join(lines),
        elapsed_sec=float(elapsed),
    )


def _artifact(label: str, path: Path) -> ArtifactStatus:
    resolved = path if path.is_absolute() else PROJECT_ROOT / path
    exists = resolved.exists() and resolved.is_file()
    stat = resolved.stat() if exists else None
    modified = None
    if stat is not None:
        modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds")
    try:
        display_path = str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        display_path = str(resolved)
    return ArtifactStatus(
        label=label,
        path=display_path,
        exists=exists,
        size_bytes=int(stat.st_size) if stat is not None else 0,
        modified_utc=modified,
    )


def artifact_inventory(
    dataset_path: Path = Path("data/processed/synthetic.parquet"),
    model_dir: Path = Path("models"),
    hil_report: Path = Path("reports/phase4_hil_benchmark.json"),
) -> list[ArtifactStatus]:
    return [
        _artifact("Dataset", dataset_path),
        _artifact("Tandem checkpoint", model_dir / "tandem.pt"),
        _artifact("Inverse ONNX", model_dir / "inverse_pcf_spr.onnx"),
        _artifact("Edge denoiser", model_dir / "edge_denoiser.keras"),
        _artifact("RI predictor", model_dir / "edge_ri_predictor.keras"),
        _artifact("INT8 denoiser", model_dir / "edge_denoiser_quantized.tflite"),
        _artifact("INT8 RI predictor", model_dir / "edge_ri_predictor_quantized.tflite"),
        _artifact("HIL report", hil_report),
    ]


def capability_inventory() -> list[dict[str, object]]:
    checks = [
        ("Dashboard", "streamlit", True),
        ("Inverse training", "torch", True),
        ("Edge training", "tensorflow", False),
        ("TFLite runtime", "ai_edge_litert", False),
        ("ONNX export/runtime", "onnx", False),
        ("COMSOL bridge", "mph", False),
        ("Hardware serial", "serial", False),
        ("SHAP explainability", "shap", False),
    ]
    return [
        {
            "capability": label,
            "module": module,
            "required_for_dashboard": required,
            "available": importlib.util.find_spec(module) is not None,
        }
        for label, module, required in checks
    ]


def human_bytes(size: int) -> str:
    value = float(max(size, 0))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"
