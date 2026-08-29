from __future__ import annotations

import json
from pathlib import Path

import pytest

from sprpcf.evidence.qualification import (
    qualify_comsol_iteration,
    qualify_device_benchmark,
    qualify_experimental_sensor,
    validate_evidence_registry,
    write_evidence_record,
)
from sprpcf.utils.readiness import build_readiness_report
from sprpcf.utils.reproducibility import sha256_file


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_comsol_qualification_requires_real_backend_hashes_and_immutable_class(tmp_path: Path) -> None:
    iteration = tmp_path / "iteration"
    model = _write(tmp_path / "model.mph", "dummy model bytes")
    config = _write(tmp_path / "config.yaml", "study: std1\n")
    simulation = _write(iteration / "simulation_results.csv", "sample_id,status\n0,ok\n")
    verification = _write(iteration / "verification.csv", "target_id,accepted,reason\n0,true,\n")
    _write(simulation.with_suffix(".csv.meta.json"), json.dumps({"evidence_class": "comsol_physics"}))
    manifest = {
        "schema_version": 1,
        "backend": "comsol",
        "evidence_class": "comsol_physics",
        "selected_targets": 1,
        "accepted_targets": 1,
        "ri_points": 5,
        "seed": 7,
        "inputs": {
            "comsol_model": str(model),
            "comsol_model_sha256": sha256_file(model),
            "comsol_config": str(config),
            "comsol_config_sha256": sha256_file(config),
        },
        "outputs": {
            "simulation_results": str(simulation),
            "verification": str(verification),
        },
    }
    _write(iteration / "iteration_manifest.json", json.dumps(manifest))
    registry = tmp_path / "evidence_registry.json"

    record = qualify_comsol_iteration(iteration, registry_path=registry)
    assert record["evidence_class"] == "comsol_physics"
    report = write_evidence_record(registry, record)
    assert report["ok"] is True

    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    registry_payload["records"][0]["evidence_class"] = "experimental_sensor"
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    edited = validate_evidence_registry(registry)
    assert edited["ok"] is False
    assert any("record_id mismatch" in error for error in edited["errors"])
    write_evidence_record(registry, record)

    manifest["backend"] = "synthetic"
    _write(iteration / "iteration_manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="backend='comsol'"):
        qualify_comsol_iteration(iteration, registry_path=registry)


def test_registry_drives_full_readiness_and_detects_artifact_tampering(tmp_path: Path) -> None:
    registry = tmp_path / "evidence_registry.json"

    raw = _write(tmp_path / "measured.jsonl", '{"wavelength_nm":[500,501],"loss":[1,2]}\n')
    protocol = _write(tmp_path / "protocol.md", "# Measurement protocol\n")
    calibration = _write(tmp_path / "calibration.json", '{"dark":"recorded","reference":"recorded"}\n')
    experimental = qualify_experimental_sensor(
        [raw],
        protocol_path=protocol,
        calibration_path=calibration,
        instrument_id="SPEC-UNIT-001",
        acquired_at="2026-08-29T10:30:00+03:00",
        registry_path=registry,
    )
    write_evidence_record(registry, experimental)

    benchmark = _write(
        tmp_path / "benchmark.json",
        json.dumps(
            {
                "iterations": 100,
                "latency_ms_p50": 2.0,
                "latency_ms_p95": 3.0,
                "latency_ms_p99": 4.0,
                "throughput_fps": 400.0,
            }
        ),
    )
    deployed_model = _write(tmp_path / "model.tflite", "quantized model")
    device = qualify_device_benchmark(
        benchmark,
        model_path=deployed_model,
        device_name="Test target device",
        os_name="Test OS",
        runtime="LiteRT",
        accelerator="CPU",
        registry_path=registry,
    )
    write_evidence_record(registry, device)

    iteration = tmp_path / "iteration"
    comsol_model = _write(tmp_path / "physics.mph", "validated model placeholder")
    comsol_config = _write(tmp_path / "physics.yaml", "study: std1\n")
    simulation = _write(iteration / "simulation_results.csv", "sample_id,status\n0,ok\n")
    verification = _write(iteration / "verification.csv", "target_id,accepted,reason\n0,false,tolerance\n")
    manifest = {
        "backend": "comsol",
        "evidence_class": "comsol_physics",
        "selected_targets": 1,
        "accepted_targets": 0,
        "ri_points": 5,
        "seed": 7,
        "inputs": {
            "comsol_model": str(comsol_model),
            "comsol_model_sha256": sha256_file(comsol_model),
            "comsol_config": str(comsol_config),
            "comsol_config_sha256": sha256_file(comsol_config),
        },
        "outputs": {"simulation_results": str(simulation), "verification": str(verification)},
    }
    _write(iteration / "iteration_manifest.json", json.dumps(manifest))
    comsol = qualify_comsol_iteration(iteration, registry_path=registry)
    write_evidence_record(registry, comsol)

    registry_report = validate_evidence_registry(registry)
    assert registry_report["ok"] is True
    assert set(registry_report["evidence_classes"]) == {
        "comsol_physics",
        "experimental_sensor",
        "device_benchmark",
    }

    readiness = build_readiness_report(
        ".",
        profile="full",
        expected_version="1.0.0rc1",
        evidence_registry=registry,
    )
    assert readiness["ready"] is True
    assert readiness["missing_full_evidence"] == []

    raw.write_text("tampered\n", encoding="utf-8")
    tampered = validate_evidence_registry(registry)
    assert tampered["ok"] is False
    assert any("hash mismatch" in error for error in tampered["errors"])

    blocked = build_readiness_report(
        ".",
        profile="full",
        expected_version="1.0.0rc1",
        evidence_registry=registry,
    )
    assert blocked["ready"] is False
    assert any(item["name"] == "evidence_registry" for item in blocked["required_failures"])
