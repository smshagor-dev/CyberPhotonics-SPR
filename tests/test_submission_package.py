from __future__ import annotations

import json
from pathlib import Path

from sprpcf.publication.evidence import EvidenceSource, build_reviewer_package
from sprpcf.publication.submission import _metric_unit, build_submission_package, validate_submission_package


def test_metric_unit_prioritizes_statistic_semantics() -> None:
    assert _metric_unit("dataset_constraints.d_over_lambda_above_max_rate") == "dimensionless"
    assert _metric_unit("fixed_geometry_sweeps.sensitivity_nm_per_riu.confidence") == "dimensionless"
    assert _metric_unit("ridge_baseline.fom_per_riu_r2") == "dimensionless"
    assert _metric_unit("ridge_baseline.lambda_res_nm_r2") == "dimensionless"
    assert _metric_unit("ridge_baseline.lambda_res_nm_rmse") == "nm"
    assert _metric_unit("ridge_baseline.sensitivity_nm_per_riu_mae") == "nm/RIU"


def test_submission_package_generates_tables_figures_and_gap_flags(tmp_path: Path) -> None:
    validation = tmp_path / "validation"
    validation.mkdir()
    (validation / "summary.json").write_text(
        json.dumps(
            {
                "fixed_geometry_sweeps": {
                    "geometry_sweeps": 4,
                    "median_linearity_r2": 0.99,
                    "sensitivity_nm_per_riu": {"mean": 1200.0, "confidence": 0.95},
                },
                "ridge_baseline": {"lambda_res_nm_r2": 0.98, "lambda_res_nm_rmse": 1.2},
                "dataset_constraints": {"d_over_lambda_above_max_rate": 0.0},
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
    table = (out / "TABLE_S1_VALIDATION_METRICS.csv").read_text(encoding="utf-8")
    assert "1200" in table
    assert "d_over_lambda_above_max_rate,0,dimensionless" in table
    assert "sensitivity_nm_per_riu.confidence,0.95,dimensionless" in table
    assert "lambda_res_nm_r2,0.98,dimensionless" in table
    assert "lambda_res_nm_rmse,1.2,nm" in table
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
    build_submission_package(out, reviewer_package_dir=reviewer, version="1.0.0rc1", repo_root=tmp_path)
    assert validate_submission_package(out)["ok"] is True

    (out / "README_FIRST.md").write_text("tampered\n", encoding="utf-8")
    validation = validate_submission_package(out)
    assert validation["ok"] is False
    assert any("Checksum mismatch: README_FIRST.md" in error for error in validation["errors"])
