from __future__ import annotations

import sys
from pathlib import Path

import pytest

from sprpcf.dashboard.operations import MAIN_SCRIPT, artifact_inventory, build_cli_command, human_bytes


def test_build_cli_command_uses_current_python_without_shell() -> None:
    command = build_cli_command("generate-data", ["--samples", "12", "--seed", "7"])
    assert command[:3] == [sys.executable, "-u", str(MAIN_SCRIPT)]
    assert command[3:] == ["generate-data", "--samples", "12", "--seed", "7"]
    with pytest.raises(ValueError):
        build_cli_command("--bad")


def test_build_cli_command_normalizes_edge_gpu_aliases() -> None:
    edge = build_cli_command("train-edge", ["--device", "gpu", "--epochs", "2"])
    assert edge[edge.index("--device") + 1] == "/GPU:0"

    pipeline = build_cli_command("run-pipeline", ["--edge-device", "cuda", "--device", "cuda"])
    assert pipeline[pipeline.index("--edge-device") + 1] == "/GPU:0"
    assert pipeline[pipeline.index("--device") + 1] == "cuda"


def test_artifact_inventory_reports_real_files(tmp_path: Path) -> None:
    dataset = tmp_path / "synthetic.parquet"
    models = tmp_path / "models"
    report = tmp_path / "hil.json"
    dataset.write_bytes(b"dataset")
    models.mkdir()
    (models / "tandem.pt").write_bytes(b"checkpoint")
    (models / "edge_denoiser_quantized.tflite").write_bytes(b"tflite")
    report.write_text("{}", encoding="utf-8")

    rows = {item.label: item for item in artifact_inventory(dataset, models, report)}
    assert rows["Dataset"].exists is True
    assert rows["Tandem checkpoint"].exists is True
    assert rows["INT8 denoiser"].exists is True
    assert rows["INT8 RI predictor"].exists is False
    assert rows["HIL report"].exists is True
    assert rows["Dataset"].size_bytes == len(b"dataset")


def test_human_bytes_is_compact() -> None:
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(1024) == "1.0 KB"
    assert human_bytes(1024 * 1024) == "1.0 MB"
