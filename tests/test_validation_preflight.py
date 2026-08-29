from __future__ import annotations

from pathlib import Path

import yaml

from sprpcf.validation.preflight import build_campaign_preflight, campaign_preflight_markdown


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _config(tmp_path: Path, *, placeholders: bool) -> Path:
    checkpoint = _write(tmp_path / "models" / "tandem.pt")
    targets = _write(tmp_path / "data" / "targets.csv", "target_id\n0\n")
    base_data = _write(tmp_path / "data" / "dataset.csv", "sample_id\n0\n")
    model = _write(tmp_path / "physics" / "sensor.mph")
    sweep = _write(tmp_path / "physics" / "sweep.yaml", "study: std1\n")
    protocol = _write(tmp_path / "experiment" / "protocol.md", "# Protocol\n")
    calibration = _write(tmp_path / "experiment" / "calibration.json", "{}\n")
    deployed = _write(tmp_path / "device" / "denoiser.tflite")

    payload = {
        "schema_version": 1,
        "campaign_id": "real-validation-test",
        "registry": str(tmp_path / "evidence" / "evidence_registry.json"),
        "reviewer_package": str(tmp_path / "reviewer_package"),
        "submission_package": str(tmp_path / "submission_package"),
        "comsol": {
            "checkpoint": str(checkpoint),
            "targets": str(targets),
            "base_data": str(base_data),
            "model": "REPLACE_WITH_VALIDATED_MODEL.mph" if placeholders else str(model),
            "config": str(sweep),
            "output_dir": str(tmp_path / "comsol"),
            "passes": 32,
            "ri_span": 0.04,
            "ri_points": 5,
            "seed": 7,
        },
        "experiment": {
            "raw_data": [str(tmp_path / "experiment" / "raw_frames.jsonl")],
            "protocol": str(protocol),
            "calibration": str(calibration),
            "instrument_id": "REPLACE_WITH_INSTRUMENT_ID" if placeholders else "SPEC-001",
            "acquired_at": (
                "REPLACE_WITH_TIMEZONE_AWARE_ISO8601"
                if placeholders
                else "2026-08-29T10:30:00+03:00"
            ),
        },
        "device": {
            "benchmark": str(tmp_path / "device" / "benchmark.json"),
            "model": str(deployed),
            "device_name": "REPLACE_WITH_EXACT_DEVICE_NAME" if placeholders else "Target Device A",
            "os_name": "REPLACE_WITH_EXACT_OS_IMAGE" if placeholders else "Linux image 1",
            "runtime": "LiteRT",
        },
    }
    path = tmp_path / ("placeholder.yaml" if placeholders else "ready.yaml")
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_preflight_blocks_placeholders_without_claiming_evidence(tmp_path: Path) -> None:
    report = build_campaign_preflight(_config(tmp_path, placeholders=True))

    assert report["ready"] is False
    assert report["stage_ready"]["comsol"] is False
    assert report["stage_ready"]["experiment"] is False
    assert report["stage_ready"]["device"] is False
    failed = {row["name"] for row in report["required_failures"]}
    assert "comsol:model" in failed
    assert "experiment:instrument_id" in failed
    assert "experiment:acquired_at" in failed
    assert "device:device_name" in failed
    assert "exact-device evidence" in report["scientific_boundary"]


def test_preflight_passes_when_real_execution_inputs_are_prepared(tmp_path: Path) -> None:
    report = build_campaign_preflight(_config(tmp_path, placeholders=False))

    assert report["ready"] is True
    assert report["required_failures"] == []
    assert report["stage_ready"] == {"comsol": True, "experiment": True, "device": True}
    markdown = campaign_preflight_markdown(report)
    assert "Real Validation Campaign Preflight" in markdown
    assert "Ready: **YES**" in markdown


def test_preflight_rejects_even_ri_sweep_and_naive_timestamp(tmp_path: Path) -> None:
    path = _config(tmp_path, placeholders=False)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    payload["comsol"]["ri_points"] = 4
    payload["experiment"]["acquired_at"] = "2026-08-29T10:30:00"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    report = build_campaign_preflight(path)
    failed = {row["name"] for row in report["required_failures"]}
    assert report["ready"] is False
    assert "comsol:ri_points" in failed
    assert "experiment:acquired_at" in failed
