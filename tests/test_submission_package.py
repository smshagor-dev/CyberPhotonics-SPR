from __future__ import annotations

import json
from pathlib import Path

from sprpcf.publication.evidence import EvidenceSource, build_reviewer_package
from sprpcf.publication.submission import build_submission_package, validate_submission_package


def test_submission_package_generates_tables_figures_and_gap_flags(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "summary.json").write_text(
        json.dumps(
            {
                "fixed_geometry_sweeps": {
                    "geometry_sweeps": 4,
                    "median_linearity_r2": 0.99,
                    "sensitivity_nm_per_riu": {"mean": 1200.0},
                },
                "ridge_baseline": {"r2": 0.8},
            }
        ),
        encoding="utf-8",
    )
    (validation / "validation_report.md").write_text("# validation\n", encoding="utf-8")
    (validation / "resonance_shift.png").write_bytes(b"synthetic-figure")

    reviewer = tmp_path / "reviewer"
    build_reviewer_package(
        reviewer,
        repo_root=tmp_path,
        include_release_metadata=False,
        sources=(EvidenceSource("validation", validation, "software_only", "Validation"),),
    )

    out = tmp_path / "submission"
    manifest = build_submission_package(
        out,
        reviewer_package_dir=reviewer,
        version="1.0.0rc1",
        repo_root=tmp_path,
        validation_dir=validation,
        journal="Example Journal",
    )

    assert manifest["readiness"]["software_validation_evidence"] is True
    assert manifest["readiness"]["numerical_physics_evidence"] is False
    assert manifest["readiness"]["experimental_sensor_evidence"] is False
    assert len(manifest["figures"]) == 1
    assert manifest["figures"][0]["evidence_class"] == "software_only"
    assert (out / "SUPPLEMENTARY_INFORMATION.md").is_file()
    assert (out / "MANUSCRIPT_CHECKLIST.md").is_file()
    assert (out / "TABLE_S1_VALIDATION_METRICS.csv").is_file()
    assert "1200" in (out / "TABLE_S1_VALIDATION_METRICS.csv").read_text(encoding="utf-8")
    assert validate_submission_package(out)["ok"] is True


def test_submission_checksum_validation_detects_tampering(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")

    reviewer = tmp_path / "reviewer"
    build_reviewer_package(
        reviewer,
        repo_root=tmp_path,
        include_release_metadata=False,
        sources=(EvidenceSource("validation", evidence, "software_only"),),
    )
    out = tmp_path / "submission"
    build_submission_package(
        out,
        reviewer_package_dir=reviewer,
        version="1.0.0rc1",
        repo_root=tmp_path,
    )
    assert validate_submission_package(out)["ok"] is True

    (out / "README_FIRST.md").write_text("tampered\n", encoding="utf-8")
    validation = validate_submission_package(out)
    assert validation["ok"] is False
    assert any("Checksum mismatch: README_FIRST.md" in error for error in validation["errors"])
