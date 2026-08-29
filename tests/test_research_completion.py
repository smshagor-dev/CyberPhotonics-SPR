from __future__ import annotations

from pathlib import Path

import yaml

from sprpcf.validation.campaign import initialize_campaign
from sprpcf.validation.completion import qualify_available_evidence, research_completion_status


def _campaign_config(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "campaign_id": "completion-test",
        "registry": str(tmp_path / "evidence" / "registry.json"),
        "reviewer_package": str(tmp_path / "reviewer"),
        "submission_package": str(tmp_path / "submission"),
        "comsol": {
            "checkpoint": str(tmp_path / "model.pt"),
            "targets": str(tmp_path / "targets.csv"),
            "base_data": str(tmp_path / "data.parquet"),
            "model": str(tmp_path / "physics.mph"),
            "config": str(tmp_path / "sweep.yaml"),
            "output_dir": str(tmp_path / "comsol"),
        },
        "experiment": {
            "raw_data": [str(tmp_path / "raw.jsonl")],
            "protocol": str(tmp_path / "protocol.md"),
            "calibration": str(tmp_path / "calibration.json"),
            "instrument_id": "TEST-SPEC",
            "acquired_at": "2026-08-29T10:30:00+03:00",
            "measurement_manifest": str(tmp_path / "measurements.yaml"),
            "analysis_output_dir": str(tmp_path / "analysis"),
        },
        "device": {
            "benchmark": str(tmp_path / "benchmark.json"),
            "model": str(tmp_path / "model.tflite"),
            "device_name": "TEST-DEVICE",
            "os_name": "TEST-OS",
            "runtime": "LiteRT",
        },
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_completion_status_uses_work_names_and_keeps_missing_evidence_blocked(tmp_path: Path) -> None:
    campaign_dir = tmp_path / "campaign"
    initialize_campaign(_campaign_config(tmp_path), campaign_dir)

    report = research_completion_status(campaign_dir, repo_root=".")
    names = [row["name"] for row in report["works"]]

    assert names == [
        "Real COMSOL Validation",
        "Experimental Sensor Validation",
        "Exact-Device Benchmark",
        "Evidence-Aware Finalization",
        "Paper Results Finalization",
        "Stable Release",
    ]
    assert all("M9" not in name for name in names)
    assert report["complete"] is False
    assert report["works"][-1]["status"] == "blocked"

    qualification = qualify_available_evidence(campaign_dir)
    assert qualification["validation"]["ok"] is False
    assert {row["status"] for row in qualification["actions"]} == {"pending"}
