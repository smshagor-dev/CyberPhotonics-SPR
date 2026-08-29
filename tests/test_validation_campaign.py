from __future__ import annotations

from pathlib import Path

import yaml

from sprpcf.validation.campaign import campaign_status, initialize_campaign, stable_release_gate


def _config(tmp_path: Path) -> Path:
    payload = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "registry": str(tmp_path / "evidence_registry.json"),
        "reviewer_package": str(tmp_path / "reviewer"),
        "submission_package": str(tmp_path / "submission"),
        "comsol": {
            "checkpoint": str(tmp_path / "model.pt"),
            "targets": str(tmp_path / "targets.csv"),
            "base_data": str(tmp_path / "data.csv"),
            "model": str(tmp_path / "physics.mph"),
            "config": str(tmp_path / "sweep.yaml"),
            "output_dir": str(tmp_path / "comsol"),
        },
        "experiment": {
            "raw_data": [str(tmp_path / "raw.jsonl")],
            "protocol": str(tmp_path / "protocol.md"),
            "calibration": str(tmp_path / "calibration.json"),
            "instrument_id": "TEST-INSTRUMENT",
            "acquired_at": "2026-08-29T10:30:00+03:00",
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


def test_campaign_init_is_plan_only_and_hash_bound(tmp_path: Path) -> None:
    out = tmp_path / "campaign"
    manifest = initialize_campaign(_config(tmp_path), out)

    assert manifest["stages"]["comsol"]["status"] == "pending"
    assert "does not create COMSOL" in manifest["scientific_boundary"]
    assert (out / "RUNBOOK.md").is_file()
    assert (out / "EXPERIMENT_PROTOCOL_TEMPLATE.md").is_file()
    assert (out / "CALIBRATION_RECORD_TEMPLATE.json").is_file()
    assert (out / "DEVICE_METADATA_TEMPLATE.json").is_file()

    status = campaign_status(out)
    assert status["complete"] is False
    assert status["snapshot_integrity"] is True
    assert set(status["missing_evidence_classes"]) == {
        "comsol_physics",
        "experimental_sensor",
        "device_benchmark",
    }

    snapshot = out / "campaign_config.snapshot.yaml"
    snapshot.write_text(snapshot.read_text(encoding="utf-8") + "\n# tamper\n", encoding="utf-8")
    tampered = campaign_status(out)
    assert tampered["snapshot_integrity"] is False
    assert tampered["complete"] is False


def test_stable_gate_blocks_rc_and_missing_physical_evidence(tmp_path: Path) -> None:
    out = tmp_path / "campaign"
    initialize_campaign(_config(tmp_path), out)

    gate = stable_release_gate(out, repo_root=".", expected_version="1.0.0rc1")
    assert gate["ready_for_stable_release"] is False
    assert any("prerelease" in blocker for blocker in gate["blockers"])
    assert any("campaign is incomplete" in blocker for blocker in gate["blockers"])
