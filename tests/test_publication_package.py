from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sprpcf.publication.evidence import EvidenceSource, build_reviewer_package


def _claim_status(manifest: dict, prefix: str) -> str:
    for row in manifest["claims"]:
        if row["claim"].startswith(prefix):
            return row["status"]
    raise AssertionError(f"Claim not found: {prefix}")


def test_reviewer_package_keeps_evidence_classes_separate(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "summary.json").write_text('{"demo": true}\n', encoding="utf-8")
    (validation / "validation_report.md").write_text("# validation\n", encoding="utf-8")

    closed_loop = tmp_path / "closed_loop"
    closed_loop.mkdir()
    (closed_loop / "iteration_manifest.json").write_text(
        '{"backend": "comsol", "evidence_class": "comsol_physics"}\n',
        encoding="utf-8",
    )
    (closed_loop / "verification.csv").write_text("accepted\ntrue\n", encoding="utf-8")

    hardware = tmp_path / "hardware"
    hardware.mkdir()
    (hardware / "benchmark.json").write_text('{"device": "test-target"}\n', encoding="utf-8")

    out = tmp_path / "reviewer"
    manifest = build_reviewer_package(
        out,
        repo_root=tmp_path,
        include_release_metadata=False,
        sources=(
            EvidenceSource("validation", validation, "software_only", "Validation"),
            EvidenceSource("closed_loop", closed_loop, None, "Closed loop"),
            EvidenceSource("hardware", hardware, "device_benchmark", "Device benchmark"),
        ),
    )

    assert manifest["evidence_classes"] == ["comsol_physics", "device_benchmark", "software_only"]
    assert _claim_status(manifest, "Synthetic/software") == "supported"
    assert _claim_status(manifest, "Numerical PCF-SPR") == "supported"
    assert _claim_status(manifest, "Experimental sensor") == "not supplied"
    assert _claim_status(manifest, "Target-device") == "supported"
    assert _claim_status(manifest, "Confidence/ranking") == "not claimed"


def test_closed_loop_auto_classification_does_not_promote_synthetic_evidence(tmp_path: Path) -> None:
    closed_loop = tmp_path / "closed_loop"
    closed_loop.mkdir()
    (closed_loop / "iteration_manifest.json").write_text(
        '{"backend": "synthetic", "evidence_class": "software_only"}\n',
        encoding="utf-8",
    )
    (closed_loop / "verification.csv").write_text("accepted\ntrue\n", encoding="utf-8")

    manifest = build_reviewer_package(
        tmp_path / "reviewer",
        repo_root=tmp_path,
        include_release_metadata=False,
        sources=(EvidenceSource("closed_loop", closed_loop),),
    )

    assert manifest["evidence_classes"] == ["software_only"]
    assert _claim_status(manifest, "Numerical PCF-SPR") == "not supplied"
    assert _claim_status(manifest, "Experimental sensor") == "not supplied"


def test_package_excludes_model_binaries_and_hashes_packaged_files(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "table.csv").write_text("x\n1\n", encoding="utf-8")
    (evidence / "model.pt").write_bytes(b"not-for-reviewer-package")

    out = tmp_path / "reviewer"
    manifest = build_reviewer_package(
        out,
        repo_root=tmp_path,
        include_release_metadata=False,
        sources=(EvidenceSource("validation", evidence, "software_only"),),
    )

    packaged_paths = {row["package_path"] for row in manifest["artifacts"]}
    assert "evidence/validation/table.csv" in packaged_paths
    assert all(not path.endswith(".pt") for path in packaged_paths)

    table = out / "evidence" / "validation" / "table.csv"
    expected = hashlib.sha256(table.read_bytes()).hexdigest()
    table_row = next(row for row in manifest["artifacts"] if row["package_path"].endswith("table.csv"))
    assert table_row["sha256"] == expected
    assert expected in (out / "checksums.sha256").read_text(encoding="utf-8")


def test_reviewer_package_rejects_nonempty_output(tmp_path: Path) -> None:
    out = tmp_path / "reviewer"
    out.mkdir()
    (out / "old.txt").write_text("stale", encoding="utf-8")

    with pytest.raises(FileExistsError):
        build_reviewer_package(out, sources=(), repo_root=tmp_path, include_release_metadata=False)


def test_hardware_defaults_to_software_only_without_explicit_measured_class(tmp_path: Path) -> None:
    hardware = tmp_path / "hardware"
    hardware.mkdir()
    (hardware / "benchmark.json").write_text('{"latency_ms": 1.0}\n', encoding="utf-8")

    manifest = build_reviewer_package(
        tmp_path / "reviewer",
        repo_root=tmp_path,
        include_release_metadata=False,
        sources=(EvidenceSource("hardware", hardware),),
    )

    assert manifest["evidence_classes"] == ["software_only"]
    assert _claim_status(manifest, "Experimental sensor") == "not supplied"
    assert _claim_status(manifest, "Target-device") == "not supplied"
