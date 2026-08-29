from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sprpcf.evidence.qualification import (
    qualify_comsol_iteration,
    qualify_device_benchmark,
    qualify_experimental_sensor,
    write_evidence_record,
)
from sprpcf.publication.finalization import build_evidence_finalization_package
from sprpcf.publication.results import build_paper_results_package, validate_paper_results_package
from sprpcf.utils.reproducibility import sha256_file
from sprpcf.utils.stable_release import build_stable_release_plan
from sprpcf.validation.experiment import analyze_experimental_measurements


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


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
        "outputs": {"simulation_results": str(simulation), "verification": str(verification)},
    }
    _write(iteration / "iteration_manifest.json", json.dumps(manifest))
    write_evidence_record(registry, qualify_comsol_iteration(iteration, registry_path=registry))


def _add_experiment_and_analysis(registry: Path, root: Path) -> Path:
    protocol = _write(root / "experiment" / "protocol.md", "# Protocol\nTest protocol.\n")
    calibration = _write(root / "experiment" / "calibration.json", '{"dark":"recorded"}\n')
    manifest_rows = []
    raw_files: list[Path] = []
    for ri, center in ((1.33, 620.0), (1.35, 636.0), (1.37, 652.0)):
        for replicate, shift in ((1, -0.1), (2, 0.1)):
            raw = _write(root / "experiment" / f"raw_{ri}_{replicate}.jsonl", '{"measured":true}\n')
            raw_files.append(raw)
            wavelength = np.linspace(600.0, 680.0, 321)
            loss = 2.0 + 8.0 * np.exp(-0.5 * ((wavelength - (center + shift)) / 4.0) ** 2)
            calibrated = root / "experiment" / f"calibrated_{ri}_{replicate}.csv"
            pd.DataFrame({"wavelength_nm": wavelength, "loss_db_per_cm": loss}).to_csv(calibrated, index=False)
            manifest_rows.append(
                {
                    "analyte_ri": ri,
                    "replicate": str(replicate),
                    "raw_path": raw.name,
                    "path": calibrated.name,
                }
            )
    measurement_manifest = root / "experiment" / "measurements.yaml"
    measurement_manifest.write_text(
        yaml.safe_dump({"schema_version": 1, "experiment_id": "test", "spectra": manifest_rows}, sort_keys=False),
        encoding="utf-8",
    )
    record = qualify_experimental_sensor(
        raw_files,
        protocol_path=protocol,
        calibration_path=calibration,
        instrument_id="TEST-SPEC",
        acquired_at="2026-08-29T10:30:00+03:00",
        registry_path=registry,
    )
    write_evidence_record(registry, record)
    analysis_dir = root / "experiment" / "analysis"
    analyze_experimental_measurements(measurement_manifest, analysis_dir)
    return analysis_dir


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
                "peak_memory_bytes": 123456,
            }
        ),
    )
    model = _write(root / "device" / "model.tflite", "test model")
    record = qualify_device_benchmark(
        benchmark,
        model_path=model,
        device_name="Test Device",
        os_name="Test OS",
        runtime="LiteRT",
        accelerator="CPU",
        registry_path=registry,
    )
    write_evidence_record(registry, record)


def test_paper_results_require_hash_linked_experimental_analysis(tmp_path: Path) -> None:
    registry = tmp_path / "evidence" / "registry.json"
    _add_comsol(registry, tmp_path)
    analysis = _add_experiment_and_analysis(registry, tmp_path)
    _add_device(registry, tmp_path)

    out = tmp_path / "paper_results"
    manifest = build_paper_results_package(out, evidence_registry=registry, experimental_analysis_dir=analysis)
    validation = validate_paper_results_package(out)

    assert validation["ok"] is True
    assert manifest["ready_for_manuscript_results"] is True
    assert manifest["experiment"]["analysis_bound_to_qualified_raw"] is True
    assert manifest["experiment"]["metrics"]["sensitivity_nm_per_riu"] == pytest.approx(800.0, rel=1e-3)
    assert (out / "TABLE_COMSOL_EVIDENCE.csv").is_file()
    assert (out / "TABLE_DEVICE_BENCHMARK.csv").is_file()
    assert (out / "figures" / "resonance_vs_ri.png").is_file()


def test_stable_promotion_plan_allows_only_version_blocker_on_ready_rc(tmp_path: Path) -> None:
    registry = tmp_path / "evidence" / "registry.json"
    _add_comsol(registry, tmp_path)
    analysis = _add_experiment_and_analysis(registry, tmp_path)
    _add_device(registry, tmp_path)

    finalization = tmp_path / "finalization"
    final = build_evidence_finalization_package(
        finalization,
        evidence_registry=registry,
        repo_root=".",
        version="1.0.0rc1",
    )
    assert final["full_readiness"] is True
    assert final["ready_for_stable_release"] is False
    assert {row["gate"] for row in final["blockers"]} == {"stable_version"}

    results = tmp_path / "paper_results"
    build_paper_results_package(results, evidence_registry=registry, experimental_analysis_dir=analysis)
    plan = build_stable_release_plan(
        repo_root=".",
        finalization_dir=finalization,
        paper_results_dir=results,
        target_version="1.0.0",
    )
    assert plan["finalization"]["rc_promotion_ready"] is True
    assert plan["ready_for_promotion"] is True
    assert plan["blockers"] == []
