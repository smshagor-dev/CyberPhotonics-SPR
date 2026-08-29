from __future__ import annotations

import json
from pathlib import Path

import pytest

from sprpcf.evidence.qualification import (
    qualify_comsol_iteration,
    qualify_device_benchmark,
    qualify_experimental_sensor,
    write_evidence_record,
)
from sprpcf.publication.finalization import (
    build_evidence_finalization_package,
    validate_finalization_package,
)
from sprpcf.utils.reproducibility import sha256_file


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _add_experiment(registry: Path, root: Path) -> None:
    raw = _write(root / "experiment" / "raw.jsonl", '{"axis":[1,2,3,4],"signal":[1,1,1,1]}\n')
    protocol = _write(root / "experiment" / "protocol.md", "# Protocol\nMeasured fixture for qualification tests.\n")
    calibration = _write(root / "experiment" / "calibration.json", '{"dark":"recorded","reference":"recorded"}\n')
    record = qualify_experimental_sensor(
        [raw],
        protocol_path=protocol,
        calibration_path=calibration,
        instrument_id="TEST-SPEC-001",
        acquired_at="2026-08-29T10:30:00+03:00",
        registry_path=registry,
    )
    write_evidence_record(registry, record)


def _add_device(registry: Path, root: Path) -> None:
    benchmark = _write(
        root / "device" / "benchmark.json",
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
    model = _write(root / "device" / "model.tflite", "test model bytes")
    record = qualify_device_benchmark(
        benchmark,
        model_path=model,
        device_name="Test target device",
        os_name="Test OS image",
        runtime="LiteRT",
        accelerator="CPU",
        registry_path=registry,
    )
    write_evidence_record(registry, record)


def _add_comsol(registry: Path, root: Path) -> None:
    iteration = root / "comsol" / "iteration"
    model = _write(root / "comsol" / "sensor.mph", "test model bytes")
    config = _write(root / "comsol" / "config.yaml", "study: std1\n")
    simulation = _write(iteration / "simulation_results.csv", "sample_id,status\n0,ok\n")
    verification = _write(iteration / "verification.csv", "target_id,accepted,reason\n0,true,\n")
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
    record = qualify_comsol_iteration(iteration, registry_path=registry)
    write_evidence_record(registry, record)


def test_partial_registry_builds_delta_and_blocks_stable_release(tmp_path: Path) -> None:
    registry = tmp_path / "evidence" / "registry.json"
    _add_experiment(registry, tmp_path)

    out = tmp_path / "finalization"
    manifest = build_evidence_finalization_package(out, evidence_registry=registry, version="1.0.0rc1")
    validation = validate_finalization_package(out)

    assert validation["ok"] is True
    assert manifest["present_physical_classes"] == ["experimental_sensor"]
    assert set(manifest["missing_physical_classes"]) == {"comsol_physics", "device_benchmark"}
    assert manifest["ready_for_stable_release"] is False
    assert manifest["stable_version"] is False
    blockers = {row["gate"] for row in manifest["blockers"]}
    assert "evidence:comsol_physics" in blockers
    assert "evidence:device_benchmark" in blockers
    assert "stable_version" in blockers

    delta = json.loads((out / "EVIDENCE_DELTA.json").read_text(encoding="utf-8"))
    supported = {row["claim"] for row in delta["claims"] if row["status"] == "supported"}
    assert "Experimental sensor performance is measured" in supported
    assert "Numerical PCF-SPR physics performance is independently verified" not in supported


def test_all_qualified_physical_classes_still_do_not_release_prerelease_version(tmp_path: Path) -> None:
    registry = tmp_path / "evidence" / "registry.json"
    _add_experiment(registry, tmp_path)
    _add_device(registry, tmp_path)
    _add_comsol(registry, tmp_path)

    out = tmp_path / "finalization"
    manifest = build_evidence_finalization_package(out, evidence_registry=registry, version="1.0.0rc1")

    assert set(manifest["present_physical_classes"]) == {
        "comsol_physics",
        "experimental_sensor",
        "device_benchmark",
    }
    assert manifest["missing_physical_classes"] == []
    assert manifest["full_readiness"] is True
    assert manifest["stable_version"] is False
    assert manifest["ready_for_stable_release"] is False
    assert {row["gate"] for row in manifest["blockers"]} == {"stable_version"}


def test_finalization_detects_tampering_and_replace_is_scoped(tmp_path: Path) -> None:
    registry = tmp_path / "evidence" / "registry.json"
    _add_experiment(registry, tmp_path)
    out = tmp_path / "finalization"
    build_evidence_finalization_package(out, evidence_registry=registry, version="1.0.0rc1")

    delta = out / "EVIDENCE_DELTA.md"
    delta.write_text(delta.read_text(encoding="utf-8") + "tampered\n", encoding="utf-8")
    assert validate_finalization_package(out)["ok"] is False

    build_evidence_finalization_package(
        out,
        evidence_registry=registry,
        version="1.0.0rc1",
        replace=True,
    )
    assert validate_finalization_package(out)["ok"] is True

    foreign = tmp_path / "foreign"
    _write(foreign / "keep.txt", "do not remove")
    with pytest.raises(ValueError, match="Refusing to replace"):
        build_evidence_finalization_package(
            foreign,
            evidence_registry=registry,
            version="1.0.0rc1",
            replace=True,
        )
