from __future__ import annotations

import json
from pathlib import Path

from sprpcf.utils.readiness import build_readiness_report, readiness_markdown


def test_release_readiness_passes_for_repository() -> None:
    report = build_readiness_report(".", profile="release", expected_version="1.0.0rc1")
    assert report["ready"] is True
    assert report["required_failures"] == []
    names = {item["name"] for item in report["checks"]}
    assert "system_file:configs/comsol_sweep.example.yaml" in names
    assert "release_metadata" in names


def test_full_readiness_blocks_without_external_evidence() -> None:
    report = build_readiness_report(".", profile="full", expected_version="1.0.0rc1")
    assert report["ready"] is False
    assert set(report["missing_full_evidence"]) == {
        "comsol_physics",
        "experimental_sensor",
        "device_benchmark",
    }
    failed = {item["name"] for item in report["required_failures"]}
    assert "evidence:comsol_physics" in failed
    assert "evidence:experimental_sensor" in failed
    assert "evidence:device_benchmark" in failed


def test_full_readiness_accepts_explicit_evidence_manifests(tmp_path: Path) -> None:
    reviewer = tmp_path / "reviewer"
    submission = tmp_path / "submission"
    reviewer.mkdir()
    submission.mkdir()
    (reviewer / "manifest.json").write_text(
        json.dumps(
            {
                "evidence_classes": [
                    "software_only",
                    "reproducibility",
                    "comsol_physics",
                    "experimental_sensor",
                    "device_benchmark",
                ]
            }
        ),
        encoding="utf-8",
    )
    (submission / "submission_manifest.json").write_text(
        json.dumps(
            {
                "readiness": {
                    "numerical_physics_evidence": True,
                    "experimental_sensor_evidence": True,
                    "target_device_benchmark_evidence": True,
                }
            }
        ),
        encoding="utf-8",
    )

    report = build_readiness_report(
        ".",
        profile="full",
        expected_version="1.0.0rc1",
        reviewer_package=reviewer,
        submission_package=submission,
    )
    assert report["ready"] is True
    assert report["missing_full_evidence"] == []


def test_readiness_markdown_preserves_claim_boundary() -> None:
    report = build_readiness_report(".", profile="release", expected_version="1.0.0rc1")
    text = readiness_markdown(report)
    assert "System Readiness" in text
    assert "never upgrades synthetic" in text
    assert "comsol_physics" in text
